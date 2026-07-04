"""Prefill worker: runs forward pass, holds KV cache, transfers via NCCL.

Each prefill worker runs on a dedicated GPU. It loads the model at startup,
joins the NCCL process group, registers with the scheduler, then serves
EnqueuePrefill RPCs. Internally it processes requests from a queue,
coordinates KV transfer, and reports completion back to the scheduler.
"""

import math
import queue
import threading
import time
from collections import defaultdict

from dataclasses import dataclass

import grpc
import torch

from sangam.logger import init_logger
from sangam.metrics.constants import WorkerStateTimeline
from sangam.kv_cache.kv_cache_transfer import send_paged_kv_layer_async
from sangam.model.model_runner import (
    MixedBatchItem,
    pack_mixed_batch,
    run_mixed_paged_forward,
)
from sangam.kv_cache.paged_kv_cache import RequestKVState
from sangam.metrics.utils.cuda_timer import DurationTimer
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.sampler import Sampler, SamplingRequest
from sangam.sampling_parameters import SamplingParameters
from sangam.types import PrefillQueuePolicy, WorkerType
from sangam.worker.base_worker import BaseWorker, create_scheduler_callback_stub
from sangam.batch import Batch, BatchRequestUpdate
from sangam.worker.gpu_queue import GpuWorkQueue
from sangam.worker.model_operation_metrics import (
    batch_operation_metric_values,
    create_operation_metrics_context,
    resolve_profile_layer_id,
)
from sangam.worker.prefill_queue_policy import compute_prefill_queue_priority
from sangam.worker.shared_gpu_resources import (
    WorkerSharedGpuResources,
    create_worker_shared_gpu_resources,
)
from sangam.worker.worker_config import PrefillWorkerConfig

logger = init_logger(__name__)


@dataclass
class PrefillBatchItem:
    """Parsed fields from an EnqueuePrefillRequest for GPU processing."""

    req_id: str
    sequence: list[int]
    block_start: int
    block_end: int
    request_seed: int
    sampling_parameters: SamplingParameters
    mask_id: int
    block_index: int = 0
    streaming_decode_worker_id: str | None = None
    streaming_decode_dst_rank: int | None = None
    streaming_decode_worker_address: str | None = None

    @classmethod
    def from_proto(cls, req: sangam_pb2.EnqueuePrefillRequest) -> "PrefillBatchItem":
        streaming_target = (
            req.streaming_decode_target
            if req.HasField("streaming_decode_target")
            else None
        )
        return cls(
            req_id=req.request_id,
            sequence=list(req.sequence_ids),
            block_start=req.block_start,
            block_end=req.block_end,
            block_index=req.block_index,
            request_seed=req.request_seed,
            sampling_parameters=SamplingParameters.from_proto(
                req.sampling_parameters if req.HasField("sampling_parameters") else None
            ),
            mask_id=req.mask_id,
            streaming_decode_worker_id=(
                streaming_target.worker_id if streaming_target is not None else None
            ),
            streaming_decode_dst_rank=(
                streaming_target.dst_rank if streaming_target is not None else None
            ),
            streaming_decode_worker_address=(
                streaming_target.worker_address
                if streaming_target is not None
                else None
            ),
        )

    @property
    def is_streaming(self) -> bool:
        return (
            self.streaming_decode_dst_rank is not None
            and self.streaming_decode_worker_address is not None
            and self.streaming_decode_worker_id is not None
        )


@dataclass
class PrefillResult:
    """Result from GPU prefill for a single request."""

    sampled_sequence: list[int]
    num_unmasked_tokens: int
    kv_state: RequestKVState
    cuda_event: torch.cuda.Event | None  # recorded after forward pass; None on CPU


@dataclass
class RetainedPrefillState:
    """KV retained on the prefill worker after prefill completes."""

    request_id: str
    block_index: int
    request_seed: int
    sequence: list[int]
    kv_state: RequestKVState
    cuda_event: torch.cuda.Event | None  # ordering dependency for send queues


@dataclass
class StreamingKVTransferContext:
    request_id: str
    block_index: int
    dst_rank: int
    decode_worker_id: str
    decode_worker_address: str
    page_ids: list[int]
    num_layers: int
    layer_events: list[torch.cuda.Event | None]
    layer_ready: queue.Queue[int | None]
    done_event: threading.Event


class PrefillWorkerServicer(sangam_pb2_grpc.PrefillWorkerServiceServicer):
    """gRPC servicer for prefill operations."""

    def __init__(
        self,
        config: PrefillWorkerConfig,
        *,
        model: torch.nn.Module,
        device: torch.device,
        gpu_resources: WorkerSharedGpuResources,
    ):
        self.model = model
        self.device = device
        self._config = config
        self.worker_id = config.worker_id
        self.dist_rank = config.dist_rank
        self._kv_page_size = gpu_resources.kv_pool.page_size
        self._kv_max_pages = gpu_resources.kv_pool.max_pages
        self._max_prefill_tokens_per_batch = config.max_prefill_tokens_per_batch
        self._prefill_queue_policy = PrefillQueuePolicy(config.prefill_queue_policy)
        self._op_metrics_enabled = (
            config.enable_metrics and config.enable_operation_metrics
        )
        self._op_metrics_layer_id = resolve_profile_layer_id(
            enabled=self._op_metrics_enabled,
            requested_layer_id=config.op_metrics_layer_id,
            num_layers=self.model.num_layers,
        )

        self._sampler = Sampler()
        self._gpu_resources = gpu_resources
        self._flashinfer_workspace = gpu_resources.flashinfer_workspace
        self._prefill_kv_pool = gpu_resources.kv_pool

        # GPU work queue — serializes compute (forward pass + sampling)
        self._gpu_queue = GpuWorkQueue(device=self.device)
        self._gpu_queue.start()

        # Per-destination communication queues preserve CUDA-thread ownership for NCCL
        # while allowing transfers to different decode workers to proceed in parallel.
        # Created eagerly so concurrent coordinators never race on lazy init.
        self._send_queues: dict[int, GpuWorkQueue] = {
            rank: GpuWorkQueue(device=self.device) for rank in range(config.world_size)
        }
        for send_queue in self._send_queues.values():
            send_queue.start()
        self._send_stream_pool: dict[int, torch.cuda.Stream | None] = {
            rank: (
                torch.cuda.Stream(device=self.device)
                if self.device.type == "cuda"
                else None
            )
            for rank in range(config.world_size)
        }

        # Internal request queue for autonomous processing
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()

        self._active_batch_size: int = 0
        self._retained_lock = threading.Lock()
        self._retained_requests: dict[str, RetainedPrefillState] = {}

        # Per-destination locks — serialise gRPC+NCCL coordination per decode
        # worker so that send/recv order on a (src, dst) pair always matches.
        self._dest_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)

        # Scheduler callback stub
        _, self._scheduler_stub = create_scheduler_callback_stub(
            config.scheduler_address,
            config.max_grpc_message_length,
        )

        # Cached decode worker stubs (address -> stub)
        self._decode_stubs: dict[str, sangam_pb2_grpc.DecodeWorkerServiceStub] = {}
        self._decode_stub_lock = threading.Lock()

        # Drain event — set to initiate graceful shutdown
        self._draining = threading.Event()

        # Start processing loop
        self._process_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="prefill-process"
        )
        self._process_thread.start()

    def _report_batch(self, batch: Batch) -> None:
        self._scheduler_stub.ReportBatchMetrics(batch.to_proto())

    def _create_operation_metrics_context(self):
        return create_operation_metrics_context(
            enabled=getattr(self, "_op_metrics_enabled", False),
            profile_layer_id=getattr(self, "_op_metrics_layer_id", None),
            num_layers=getattr(self.model, "num_layers", 1),
        )

    def _run_mixed_forward(self, packed_batch):
        op_metrics_context = self._create_operation_metrics_context()
        try:
            forward_result = run_mixed_paged_forward(
                self.model,
                packed_batch,
                op_metrics_context=op_metrics_context,
            )
        except TypeError as exc:
            if "op_metrics_context" not in str(exc):
                raise
            forward_result = run_mixed_paged_forward(self.model, packed_batch)
        return forward_result

    def _current_kv_page_stats(self) -> tuple[int, int, int]:
        total = self._prefill_kv_pool.max_pages
        free = self._prefill_kv_pool.allocator.num_free
        used = self._prefill_kv_pool.allocator.num_used
        return total, used, free

    def _get_decode_stub(self, address: str) -> sangam_pb2_grpc.DecodeWorkerServiceStub:
        stub = self._decode_stubs.get(address)
        if stub is not None:
            return stub
        with self._decode_stub_lock:
            stub = self._decode_stubs.get(address)
            if stub is None:
                channel = grpc.insecure_channel(
                    address,
                    options=[
                        (
                            "grpc.max_send_message_length",
                            self._config.max_grpc_message_length,
                        ),
                        (
                            "grpc.max_receive_message_length",
                            self._config.max_grpc_message_length,
                        ),
                    ],
                )
                stub = sangam_pb2_grpc.DecodeWorkerServiceStub(channel)
                self._decode_stubs[address] = stub
        return stub

    def _get_send_queue(self, dst_rank: int) -> GpuWorkQueue:
        return self._send_queues[dst_rank]

    def _get_send_stream(self, dst_rank: int) -> torch.cuda.Stream | None:
        return self._send_stream_pool[dst_rank]

    def _build_state_snapshot(
        self, state: WorkerStateTimeline
    ) -> sangam_pb2.WorkerStateSnapshot:
        kv_total_pages, kv_used_pages, kv_free_pages = self._current_kv_page_stats()
        return sangam_pb2.WorkerStateSnapshot(
            state=getattr(sangam_pb2, f"WORKER_STATE_{state.value.upper()}"),
            timestamp=time.time(),
            waiting_queue_depth=self._request_queue.qsize(),
            active_batch_size=self._active_batch_size
            if state is WorkerStateTimeline.BUSY
            else 0,
            kv_total_pages=kv_total_pages,
            kv_used_pages=kv_used_pages,
            kv_free_pages=kv_free_pages,
        )

    def _current_worker_state(self) -> WorkerStateTimeline:
        if self._active_batch_size > 0:
            return WorkerStateTimeline.BUSY
        if self._request_queue.qsize() > 0:
            return WorkerStateTimeline.QUEUED
        return WorkerStateTimeline.IDLE

    def shutdown(self) -> None:
        draining = getattr(self, "_draining", None)
        if draining is not None:
            draining.set()

        process_thread = getattr(self, "_process_thread", None)
        if (
            process_thread is not None
            and process_thread is not threading.current_thread()
        ):
            process_thread.join()

        for send_queue in getattr(self, "_send_queues", {}).values():
            send_queue.shutdown()

        gpu_queue = getattr(self, "_gpu_queue", None)
        if gpu_queue is not None:
            gpu_queue.shutdown()

    # ----- gRPC handlers -----

    def EnqueuePrefill(
        self, request: sangam_pb2.EnqueuePrefillRequest, context
    ) -> sangam_pb2.EnqueuePrefillResponse:
        """Enqueue a prefill request under the configured queue policy."""
        if self._draining.is_set():
            return sangam_pb2.EnqueuePrefillResponse(success=False)
        self._request_queue.put(
            (self._prefill_queue_priority(request), request.request_id, request)
        )
        snapshot = self._build_state_snapshot(self._current_worker_state())
        return sangam_pb2.EnqueuePrefillResponse(success=True, accepted_state=snapshot)

    def _prefill_queue_priority(
        self, request: sangam_pb2.EnqueuePrefillRequest
    ) -> tuple[int, float, str] | tuple[float, str]:
        return compute_prefill_queue_priority(request, self._prefill_queue_policy)

    def ReturnQueuedPrefills(
        self, request: sangam_pb2.ReturnQueuedPrefillsRequest, context
    ) -> sangam_pb2.ReturnQueuedPrefillsResponse:
        """Drain the waiting queue and return all unstarted requests for redistribution."""
        returned: list[sangam_pb2.EnqueuePrefillRequest] = []
        while True:
            try:
                _, _, req = self._request_queue.get_nowait()
                returned.append(req)
            except queue.Empty:
                break
        return sangam_pb2.ReturnQueuedPrefillsResponse(requests=returned)

    def TriggerKVTransfer(
        self, request: sangam_pb2.TriggerKVTransferRequest, context
    ) -> sangam_pb2.TriggerKVTransferResponse:
        with self._retained_lock:
            retained = self._retained_requests.get(request.request_id)
        if retained is None or retained.block_index != request.block_index:
            return sangam_pb2.TriggerKVTransferResponse(success=False)
        threading.Thread(
            target=self._coordinate_kv_transfer,
            args=(request,),
            daemon=True,
        ).start()
        return sangam_pb2.TriggerKVTransferResponse(success=True)

    # ----- Autonomous processing loop -----

    def _can_add_to_batch(
        self,
        req: sangam_pb2.EnqueuePrefillRequest,
        total_tokens: int,
        pages_so_far: int,
    ) -> bool:
        """Check whether a request fits in the current batch."""
        req_tokens = len(req.sequence_ids)
        req_pages = math.ceil(req_tokens / self._kv_page_size)
        if total_tokens + req_tokens > self._max_prefill_tokens_per_batch:
            return False
        if self._prefill_kv_pool.allocator.num_free < pages_so_far + req_pages:
            return False
        return True

    def _processing_loop(self) -> None:
        """Process prefill requests, batching by token budget and available KV pages."""
        while not self._draining.is_set():
            try:
                _, _, first_req = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            dequeue_time = time.time()

            batch: list[tuple] = [(first_req, dequeue_time)]
            total_tokens = len(first_req.sequence_ids)
            pages_so_far = math.ceil(total_tokens / self._kv_page_size)

            while total_tokens < self._max_prefill_tokens_per_batch:
                try:
                    _, _, req = self._request_queue.get_nowait()
                except queue.Empty:
                    break
                if not self._can_add_to_batch(req, total_tokens, pages_so_far):
                    self._request_queue.put(
                        (
                            self._prefill_queue_priority(req),
                            req.request_id,
                            req,
                        )
                    )
                    break
                t = time.time()
                req_tokens = len(req.sequence_ids)
                total_tokens += req_tokens
                pages_so_far += math.ceil(req_tokens / self._kv_page_size)
                batch.append((req, t))

            self._active_batch_size = len(batch)
            try:
                self._process_batch(batch)
            finally:
                self._active_batch_size = 0

    def prepare_phase_change(self) -> list[sangam_pb2.EnqueuePrefillRequest]:
        """Drain all state for role conversion. Returns un-processed queued requests."""

        # 1. Stop accepting new prefill requests
        self._draining.set()

        # 2. Wait for processing loop to finish current batch and exit
        self._process_thread.join()

        # 3. Drain queued requests (never processed — return to caller/scheduler)
        returned_requests: list[sangam_pb2.EnqueuePrefillRequest] = []
        while True:
            try:
                _, _, req = self._request_queue.get_nowait()
                returned_requests.append(req)
            except queue.Empty:
                break

        # 4. Wait for ALL in-flight KV transfers to complete.
        #    TriggerKVTransfer spawns a daemon thread running
        #    _coordinate_kv_transfer. Wait until _retained_requests is
        #    empty — each completed transfer calls _free_retained_request.
        deadline = time.time() + 300  # 5 min safety timeout
        while True:
            with self._retained_lock:
                if not self._retained_requests:
                    break
            if time.time() > deadline:
                logger.warning(
                    "Timed out waiting for %d retained KV transfers to complete",
                    len(self._retained_requests),
                )
                break
            time.sleep(self._config.poll_interval)

        # 5. Free any remaining retained state (should only happen on timeout)
        with self._retained_lock:
            for retained in self._retained_requests.values():
                self._prefill_kv_pool.free(retained.kv_state.page_ids)
            self._retained_requests.clear()

        # 6. Shut down communication and GPU queues
        for send_queue in self._send_queues.values():
            send_queue.shutdown()
        self._gpu_queue.shutdown()
        return returned_requests

    def _process_batch(self, batch: list[tuple]) -> None:
        """Process one or more prefill requests in one batched GPU forward pass."""
        batch_items: list[PrefillBatchItem] = []
        queue_wait_durations: list[float] = []
        for enqueue_req, dequeue_time in batch:
            wait = 0.0
            if enqueue_req.prefill_enqueue_time > 0:
                wait = max(0.0, dequeue_time - enqueue_req.prefill_enqueue_time)
            queue_wait_durations.append(wait)
            batch_items.append(PrefillBatchItem.from_proto(enqueue_req))

        total_tokens = sum(len(item.sequence) for item in batch_items)
        reqs = [enqueue_req for enqueue_req, _ in batch]

        try:
            prefill_batch_start_time = time.time()
            (
                results,
                sampling_duration,
                operation_metrics_seconds,
            ) = self._gpu_queue.submit(self._do_run_batched_prefill, batch_items)
            batch_end_time = time.time()
            (
                batch_op_attn_time,
                batch_op_mlp_time,
                batch_op_qkv_time,
            ) = batch_operation_metric_values(
                enabled=getattr(self, "_op_metrics_enabled", False),
                operation_metrics_seconds=operation_metrics_seconds,
            )

            prefill_duration = max(0.0, batch_end_time - prefill_batch_start_time)
            request_updates: list[BatchRequestUpdate] = []
            total_unmasked = 0
            for enqueue_req, batch_item, result, wait in zip(
                reqs, batch_items, results, queue_wait_durations, strict=True
            ):
                total_unmasked += result.num_unmasked_tokens
                block_tokens = result.sampled_sequence[
                    enqueue_req.block_start : enqueue_req.block_end
                ]
                block_completed = all(
                    token_id != enqueue_req.mask_id for token_id in block_tokens
                )
                if batch_item.is_streaming:
                    self._update_retained_request(
                        enqueue_req.request_id,
                        result.sampled_sequence,
                        result.cuda_event,
                    )
                elif block_completed:
                    self._prefill_kv_pool.free(result.kv_state.page_ids)
                else:
                    with self._retained_lock:
                        self._retained_requests[enqueue_req.request_id] = (
                            RetainedPrefillState(
                                request_id=enqueue_req.request_id,
                                block_index=enqueue_req.block_index,
                                request_seed=enqueue_req.request_seed,
                                sequence=result.sampled_sequence,
                                kv_state=result.kv_state,
                                cuda_event=result.cuda_event,
                            )
                        )
                request_updates.append(
                    BatchRequestUpdate(
                        request_id=enqueue_req.request_id,
                        block_index=enqueue_req.block_index,
                        success=True,
                        updated_sequence=result.sampled_sequence,
                        num_unmasked_tokens=result.num_unmasked_tokens,
                        num_forward_evals_in_batch_phase=1,
                        request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                        prefill_duration=prefill_duration,
                        prefill_queue_wait_duration=wait,
                        block_completed=block_completed,
                    )
                )
            kv_total_pages, kv_used_pages, kv_free_pages = self._current_kv_page_stats()
            self._report_batch(
                Batch(
                    worker_id=self.worker_id,
                    worker_type=WorkerType.PREFILL.value,
                    batch_size=len(batch),
                    prompt_len=total_tokens,
                    gen_len=0,
                    batch_start_time=prefill_batch_start_time,
                    batch_end_time=batch_end_time,
                    kv_total_pages=kv_total_pages,
                    kv_used_pages=kv_used_pages,
                    kv_free_pages=kv_free_pages,
                    num_unmasked_tokens=total_unmasked,
                    batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    sampling_duration=sampling_duration,
                    batch_op_attn_time=batch_op_attn_time,
                    batch_op_mlp_time=batch_op_mlp_time,
                    batch_op_qkv_time=batch_op_qkv_time,
                    request_updates=request_updates,
                    worker_state_after=self._build_state_snapshot(
                        self._current_worker_state()
                    ),
                )
            )
            logger.debug(
                f"Batched prefill complete: {len(batch)} requests, "
                f"{total_tokens} tokens, prefill={prefill_duration:.3f}s"
            )

        except Exception:
            logger.exception(f"Batched prefill failed for {len(batch)} requests")
            failure_time = time.time()
            self._report_batch(
                Batch(
                    worker_id=self.worker_id,
                    worker_type=WorkerType.PREFILL.value,
                    batch_size=len(reqs),
                    prompt_len=total_tokens,
                    gen_len=0,
                    batch_start_time=failure_time,
                    batch_end_time=failure_time,
                    kv_total_pages=self._prefill_kv_pool.max_pages,
                    kv_used_pages=self._prefill_kv_pool.allocator.num_used,
                    kv_free_pages=self._prefill_kv_pool.allocator.num_free,
                    num_unmasked_tokens=0,
                    batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    batch_op_attn_time=(
                        0.0 if getattr(self, "_op_metrics_enabled", False) else None
                    ),
                    batch_op_mlp_time=(
                        0.0 if getattr(self, "_op_metrics_enabled", False) else None
                    ),
                    batch_op_qkv_time=(
                        0.0 if getattr(self, "_op_metrics_enabled", False) else None
                    ),
                    request_updates=[
                        BatchRequestUpdate(
                            request_id=enqueue_req.request_id,
                            block_index=enqueue_req.block_index,
                            success=False,
                            updated_sequence=list(enqueue_req.sequence_ids),
                            num_unmasked_tokens=0,
                            num_forward_evals_in_batch_phase=0,
                            request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                        )
                        for enqueue_req in reqs
                    ],
                    worker_state_after=self._build_state_snapshot(
                        self._current_worker_state()
                    ),
                )
            )

    def _update_retained_request(
        self,
        request_id: str,
        sequence: list[int],
        cuda_event: torch.cuda.Event | None,
    ) -> None:
        with self._retained_lock:
            retained = self._retained_requests.get(request_id)
            if retained is not None:
                retained.sequence = sequence
                retained.cuda_event = cuda_event

    def _make_streaming_callback(
        self,
        contexts: list[StreamingKVTransferContext],
    ):
        def _callback(
            layer_idx: int,
            _kv_tensor: torch.Tensor,
            _requests: list[RequestKVState],
        ) -> None:
            event = torch.cuda.Event() if self.device.type == "cuda" else None
            if event is not None:
                event.record()
            for ctx in contexts:
                ctx.layer_events[layer_idx] = event
                ctx.layer_ready.put(layer_idx)

        return _callback

    def _coordinate_kv_transfer(
        self,
        request: sangam_pb2.TriggerKVTransferRequest,
    ) -> None:
        transfer_start_time = time.time()
        try:
            with self._retained_lock:
                retained = self._retained_requests[request.request_id]
            decode_stub = self._get_decode_stub(request.decode_worker_address)
            decode_request = getattr(request, "decode_request", None)
            if decode_request is None:
                decode_request = sangam_pb2.EnqueueDecodeRequest(
                    request_id=request.request_id,
                    sequence_ids=retained.sequence,
                    block_start=0,
                    block_end=0,
                    block_index=request.block_index,
                    request_seed=retained.request_seed,
                    mask_id=0,
                    arrival_time=0.0,
                    decode_enqueue_time=0.0,
                )
                decode_request.sampling_parameters.CopyFrom(
                    SamplingParameters.default().to_proto()
                )
            receive_request = sangam_pb2.ReceiveKVCacheRequest(
                request_id=request.request_id,
                src_rank=self.dist_rank,
                num_layers=self.model.num_layers,
                num_kv_heads=self.model.num_kv_heads,
                head_dim=self.model.head_dim,
                seq_length=len(retained.sequence),
                block_start=decode_request.block_start,
                block_end=decode_request.block_end,
                block_index=decode_request.block_index,
                sequence_ids=decode_request.sequence_ids,
                request_seed=decode_request.request_seed,
                mask_id=decode_request.mask_id,
                arrival_time=decode_request.arrival_time,
                decode_enqueue_time=decode_request.decode_enqueue_time,
                streaming=False,
                auto_enqueue_decode=True,
            )
            receive_request.sampling_parameters.CopyFrom(
                decode_request.sampling_parameters
            )
            recv_response = None
            recv_exception = None

            def _do_recv() -> None:
                nonlocal recv_response, recv_exception
                try:
                    recv_response = decode_stub.ReceiveKVCache(
                        receive_request, timeout=self._config.kv_transfer_timeout_s
                    )
                except Exception as exc:
                    recv_exception = exc

            # Serialize the full gRPC+NCCL coordination per destination rank.
            # Without this lock, concurrent coordinators for the same (src, dst)
            # pair can interleave so that gRPC enqueues recvs in a different
            # order than the per-destination send queue, causing NCCL FIFO mismatch.
            with self._dest_locks[request.decode_dst_rank]:
                recv_thread = threading.Thread(target=_do_recv, daemon=True)
                recv_thread.start()
                send_item = self._get_send_queue(request.decode_dst_rank).submit_async(
                    self._do_transfer_retained_kv,
                    request.request_id,
                    request.decode_dst_rank,
                )
                # Wait for the recv coordinator first: it owns the gRPC
                # call and posts the NCCL recvs. If the decode side fails
                # to allocate, it returns success=False before any recv
                # is posted, leaving the matching NCCL send permanently
                # unmatched.
                recv_thread.join(timeout=self._config.kv_transfer_timeout_s)
                if recv_thread.is_alive():
                    raise RuntimeError(
                        "Timed out waiting for decode worker ReceiveKVCache "
                        f"to rank {request.decode_dst_rank}"
                    )
                if recv_exception is not None:
                    # Wait for the already-queued send to complete so it
                    # doesn't outlive this coordinator and orphan the
                    # GPU send queue. With the drain on the decode side,
                    # this is a no-op in the happy case; a timeout here
                    # would indicate a separate NCCL-level hang.
                    send_item.done_event.wait(
                        timeout=self._config.kv_transfer_timeout_s
                    )
                    raise recv_exception
                if recv_response is None or not recv_response.success:
                    # Decode side rejected (e.g. pool exhausted) but is
                    # contractually obliged to drain the matching NCCL
                    # recv so our send completes. Wait to confirm and to
                    # avoid orphaning the GPU send queue if that drain
                    # contract is ever broken.
                    send_item.done_event.wait(
                        timeout=self._config.kv_transfer_timeout_s
                    )
                    raise RuntimeError("Decode worker rejected KV transfer")
                # Recv succeeded: matching NCCL ops completed on both
                # sides. The send item should be done; bound the wait
                # to surface any latent bug.
                if not send_item.done_event.wait(
                    timeout=self._config.kv_transfer_timeout_s
                ):
                    raise RuntimeError(
                        f"NCCL send to rank {request.decode_dst_rank} "
                        "did not complete after successful recv"
                    )
                if send_item.exception is not None:
                    raise send_item.exception
            if recv_exception is not None:
                raise recv_exception
            if not recv_response.success:
                raise RuntimeError("Decode worker rejected KV transfer")
            self._free_retained_request(request.request_id)
            transfer_end_time = time.time()
            self._scheduler_stub.ReportKVTransfer(
                sangam_pb2.KVTransferReport(
                    worker_id=self.worker_id,
                    request_id=request.request_id,
                    block_index=request.block_index,
                    success=True,
                    transfer_start_time=transfer_start_time,
                    transfer_end_time=transfer_end_time,
                )
            )
        except Exception as exc:
            logger.exception(
                "[%s] Deferred KV transfer failed for block %s",
                request.request_id,
                request.block_index,
            )
            self._free_retained_request(request.request_id)
            transfer_end_time = time.time()
            self._scheduler_stub.ReportKVTransfer(
                sangam_pb2.KVTransferReport(
                    worker_id=self.worker_id,
                    request_id=request.request_id,
                    block_index=request.block_index,
                    success=False,
                    transfer_start_time=transfer_start_time,
                    transfer_end_time=transfer_end_time,
                    error_message=str(exc),
                )
            )

    def _free_retained_request(self, request_id: str) -> None:
        with self._retained_lock:
            retained = self._retained_requests.pop(request_id, None)
        if retained is not None:
            self._prefill_kv_pool.free(retained.kv_state.page_ids)

    def _do_streaming_transfer(
        self,
        ctx: StreamingKVTransferContext,
        first_layer_idx: int,
    ) -> float:
        send_stream = self._get_send_stream(ctx.dst_rank)
        works: list[torch.distributed.Work] = []
        transfer_start_time: float | None = None
        layer_idx: int | None = first_layer_idx
        processed_layers = 0
        while layer_idx is not None:
            if transfer_start_time is None:
                transfer_start_time = time.time()
            event = ctx.layer_events[layer_idx]
            if send_stream is not None and event is not None:
                send_stream.wait_event(event)
            works.extend(
                send_paged_kv_layer_async(
                    kv_layer=self._prefill_kv_pool.kv_data[layer_idx],
                    page_ids=ctx.page_ids,
                    dst_rank=ctx.dst_rank,
                    stream=send_stream,
                )
            )
            processed_layers += 1
            if processed_layers == ctx.num_layers:
                break
            try:
                layer_idx = ctx.layer_ready.get(
                    timeout=self._config.streaming_layer_ready_timeout_s
                )
            except queue.Empty as exc:
                raise RuntimeError(
                    "Timed out waiting for next streaming KV layer"
                ) from exc
            if layer_idx is None:
                raise RuntimeError(
                    "Streaming prefill aborted before all layers were sent"
                )
        if send_stream is not None:
            send_stream.synchronize()
        for work in works:
            work.wait()
        return transfer_start_time or time.time()

    def _coordinate_streaming_kv_transfer(
        self,
        ctx: StreamingKVTransferContext,
        receive_request: sangam_pb2.ReceiveKVCacheRequest,
    ) -> None:
        recv_response = None
        recv_exception = None

        def _do_recv() -> None:
            nonlocal recv_response, recv_exception
            try:
                recv_response = self._get_decode_stub(
                    ctx.decode_worker_address
                ).ReceiveKVCache(receive_request)
            except Exception as exc:
                recv_exception = exc

        try:
            try:
                first_layer_idx = ctx.layer_ready.get(
                    timeout=self._config.streaming_layer_ready_timeout_s
                )
            except queue.Empty as exc:
                raise RuntimeError(
                    "Timed out waiting for first streaming KV layer"
                ) from exc
            if first_layer_idx is None:
                raise RuntimeError("Streaming transfer cancelled before first layer")
            logger.debug(
                "[%s] Streaming KV transfer started for block %s to %s",
                ctx.request_id,
                ctx.block_index,
                ctx.decode_worker_id,
            )
            with self._dest_locks[ctx.dst_rank]:
                recv_thread = threading.Thread(target=_do_recv, daemon=True)
                recv_thread.start()
                try:
                    transfer_start_time = self._get_send_queue(ctx.dst_rank).submit(
                        self._do_streaming_transfer,
                        ctx,
                        first_layer_idx,
                    )
                except Exception:
                    recv_thread.join(timeout=self._config.streaming_recv_join_timeout_s)
                    raise
                recv_thread.join(timeout=self._config.streaming_recv_join_timeout_s)
                if recv_thread.is_alive():
                    raise RuntimeError(
                        "Timed out waiting for decode worker streaming recv to finish"
                    )
            if recv_exception is not None:
                raise recv_exception
            if recv_response is None or not recv_response.success:
                raise RuntimeError("Decode worker rejected streaming KV transfer")
            transfer_end_time = time.time()
            logger.debug(
                "[%s] Streaming KV transfer finished for block %s to %s",
                ctx.request_id,
                ctx.block_index,
                ctx.decode_worker_id,
            )
            self._scheduler_stub.ReportKVTransfer(
                sangam_pb2.KVTransferReport(
                    worker_id=self.worker_id,
                    request_id=ctx.request_id,
                    block_index=ctx.block_index,
                    success=True,
                    transfer_start_time=transfer_start_time,
                    transfer_end_time=transfer_end_time,
                )
            )
        except Exception as exc:
            logger.exception(
                "[%s] Streaming KV transfer failed for block %s",
                ctx.request_id,
                ctx.block_index,
            )
            transfer_end_time = time.time()
            self._scheduler_stub.ReportKVTransfer(
                sangam_pb2.KVTransferReport(
                    worker_id=self.worker_id,
                    request_id=ctx.request_id,
                    block_index=ctx.block_index,
                    success=False,
                    transfer_start_time=transfer_end_time,
                    transfer_end_time=transfer_end_time,
                    error_message=str(exc),
                )
            )
        finally:
            self._free_retained_request(ctx.request_id)
            ctx.done_event.set()

    def _do_transfer_retained_kv(self, request_id: str, dst_rank: int) -> None:
        with self._retained_lock:
            retained = self._retained_requests[request_id]
            page_ids = list(retained.kv_state.page_ids)
            cuda_event = retained.cuda_event
        send_stream = self._get_send_stream(dst_rank)

        # Ensure forward-pass kernels (default stream, _gpu_queue thread) are
        # visible to the send stream before reading KV pages on this thread.
        if send_stream is not None and cuda_event is not None:
            send_stream.wait_event(cuda_event)

        works: list[torch.distributed.Work] = []
        for layer_kv in self._prefill_kv_pool.kv_data:
            works.extend(
                send_paged_kv_layer_async(
                    kv_layer=layer_kv,
                    page_ids=page_ids,
                    dst_rank=dst_rank,
                    stream=send_stream,
                )
            )
        if send_stream is not None:
            send_stream.synchronize()
        for work in works:
            work.wait()

    # ----- GPU-bound implementations (run on GPU thread only) -----

    @torch.inference_mode()
    def _do_run_batched_prefill(
        self, batch_items: list[PrefillBatchItem]
    ) -> tuple[list[PrefillResult], float, dict[str, float] | None]:
        """Run batched prefill and retain paged KV locally for deferred/streaming transfer."""
        block_lengths = [item.block_end - item.block_start for item in batch_items]
        distinct_block_lengths = sorted(set(block_lengths))
        if len(distinct_block_lengths) != 1:
            req_ids = [item.req_id for item in batch_items]
            raise RuntimeError(
                "Mixed prefill block lengths are unsupported: "
                f"block_lengths={distinct_block_lengths}, request_ids={req_ids}"
            )

        seq_lens = [len(item.sequence) for item in batch_items]

        # Allocate KV pages for each request
        kv_states: list[RequestKVState] = []
        all_page_ids: list = []
        try:
            for seq_len in seq_lens:
                page_ids, last_page_len = self._prefill_kv_pool.allocate(seq_len)
                kv_states.append(
                    RequestKVState(
                        page_ids=page_ids,
                        seq_len=seq_len,
                        last_page_len=last_page_len,
                    )
                )
                all_page_ids.append(page_ids)
        except Exception:
            for page_ids in all_page_ids:
                self._prefill_kv_pool.free(page_ids)
            raise

        streaming_contexts: list[
            tuple[StreamingKVTransferContext, sangam_pb2.ReceiveKVCacheRequest]
        ] = []
        for item, seq_len, kv_state in zip(
            batch_items, seq_lens, kv_states, strict=True
        ):
            if not item.is_streaming:
                continue
            with self._retained_lock:
                self._retained_requests[item.req_id] = RetainedPrefillState(
                    request_id=item.req_id,
                    block_index=item.block_index,
                    request_seed=item.request_seed,
                    sequence=list(item.sequence),
                    kv_state=kv_state,
                    cuda_event=None,
                )
            ctx = StreamingKVTransferContext(
                request_id=item.req_id,
                block_index=item.block_index,
                dst_rank=item.streaming_decode_dst_rank,
                decode_worker_id=item.streaming_decode_worker_id,
                decode_worker_address=item.streaming_decode_worker_address,
                page_ids=list(kv_state.page_ids),
                num_layers=self.model.num_layers,
                layer_events=[None] * self.model.num_layers,
                layer_ready=queue.Queue(),
                done_event=threading.Event(),
            )
            receive_request = sangam_pb2.ReceiveKVCacheRequest(
                request_id=item.req_id,
                src_rank=self.dist_rank,
                num_layers=self.model.num_layers,
                num_kv_heads=self.model.num_kv_heads,
                head_dim=self.model.head_dim,
                seq_length=seq_len,
                block_start=item.block_start,
                block_end=item.block_end,
                block_index=item.block_index,
                request_seed=item.request_seed,
                streaming=True,
                auto_enqueue_decode=False,
            )
            streaming_contexts.append((ctx, receive_request))

        packed_batch = pack_mixed_batch(
            items=[
                MixedBatchItem(
                    request_id=item.req_id,
                    token_ids=item.sequence,
                    query_start=0,
                    query_end=seq_len,
                    active_kv_len=seq_len,
                    kv_state=kv_state,
                    phase="prefill",
                )
                for item, seq_len, kv_state in zip(
                    batch_items, seq_lens, kv_states, strict=True
                )
            ],
            pool=self._prefill_kv_pool,
            num_q_heads=self.model.num_q_heads,
            device=self.device,
            workspace=self._flashinfer_workspace,
            kv_page_callback=(
                self._make_streaming_callback([ctx for ctx, _ in streaming_contexts])
                if streaming_contexts
                else None
            ),
        )
        for ctx, receive_request in streaming_contexts:
            threading.Thread(
                target=self._coordinate_streaming_kv_transfer,
                args=(ctx, receive_request),
                daemon=True,
                name=f"stream-kv-{ctx.request_id}",
            ).start()
        try:
            forward_result = self._run_mixed_forward(packed_batch)
        except Exception:
            streaming_request_ids = {ctx.request_id for ctx, _ in streaming_contexts}
            for ctx, _ in streaming_contexts:
                ctx.layer_ready.put(None)
            for item, page_ids in zip(batch_items, all_page_ids, strict=True):
                if item.req_id not in streaming_request_ids:
                    self._prefill_kv_pool.free(page_ids)
            raise

        logits = forward_result.packed_logits

        results: list[PrefillResult] = []
        with DurationTimer(
            "worker_sampling_prefill",
            use_cuda=logits.is_cuda,
            device=logits.device,
        ) as timer:
            token_ids_batch = [
                torch.tensor([item.sequence], dtype=torch.long, device=self.device)
                for item in batch_items
            ]
            sampling_reqs = [
                SamplingRequest(
                    request_id=item.req_id,
                    request_seed=item.request_seed,
                    token_ids=token_ids_i,
                    block_start_idx=item.block_start,
                    block_end_index=item.block_end,
                    step_index=0,
                    sampling_parameters=item.sampling_parameters,
                    mask_token_id=item.mask_id,
                )
                for item, token_ids_i in zip(batch_items, token_ids_batch)
            ]
            block_tokens_batch = torch.cat(
                [
                    token_ids_i[:, item.block_start : item.block_end]
                    for item, token_ids_i in zip(batch_items, token_ids_batch)
                ],
                dim=0,
            )
            logits_batch = torch.cat(
                [
                    item_logits[:, item.block_start : item.block_end, :]
                    for item, item_logits in zip(
                        batch_items, forward_result.item_logits, strict=True
                    )
                ],
                dim=0,
            )
            updated_block_tokens, num_unmasked_tokens = self._sampler.sample_batch(
                reqs=sampling_reqs,
                block_tokens=block_tokens_batch,
                logits=logits_batch,
            )
            for i, token_ids_i in enumerate(token_ids_batch):
                item = batch_items[i]
                token_ids_i[:, item.block_start : item.block_end] = (
                    updated_block_tokens[i : i + 1]
                )
                results.append(
                    PrefillResult(
                        sampled_sequence=token_ids_i[0].tolist(),
                        num_unmasked_tokens=int(num_unmasked_tokens[i].item()),
                        kv_state=kv_states[i],
                        cuda_event=None,  # filled in below
                    )
                )
        sampling_duration = timer.elapsed_s
        # Record an event on the current (default) stream after all sampling is
        # done. The deferred send queue waits on this event before reading KV
        # pages, ensuring the forward-pass writes are visible.
        cuda_event: torch.cuda.Event | None = None
        if self.device.type == "cuda":
            cuda_event = torch.cuda.Event()
            cuda_event.record()
        for item, result in zip(batch_items, results, strict=True):
            result.cuda_event = None if item.is_streaming else cuda_event
        return (
            results,
            sampling_duration,
            getattr(forward_result, "operation_metrics_seconds", None),
        )


class PrefillWorker(BaseWorker):
    """Concrete worker for prefill operations."""

    def __init__(self, config: PrefillWorkerConfig):
        super().__init__(config)

    @property
    def config(self) -> PrefillWorkerConfig:
        return self._config  # type: ignore[return-value]

    def _get_worker_type(self) -> WorkerType:
        return WorkerType.PREFILL

    def _registration_extra_fields(self) -> dict[str, int]:
        return {
            "max_pages": self.config.kv_max_pages,
            "page_size": self.config.kv_page_size,
        }

    def _create_servicer(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> PrefillWorkerServicer:
        gpu_resources = create_worker_shared_gpu_resources(
            model=model,
            device=device,
            kv_page_size=self.config.kv_page_size,
            kv_max_pages=self.config.kv_max_pages,
            kv_dtype=self.config.kv_dtype,
            zero_init=False,
        )
        return PrefillWorkerServicer(
            self.config,
            model=model,
            device=device,
            gpu_resources=gpu_resources,
        )

    def _add_servicer_to_server(self, servicer: object, server: grpc.Server) -> None:
        sangam_pb2_grpc.add_PrefillWorkerServiceServicer_to_server(servicer, server)


def serve_prefill_worker(config: PrefillWorkerConfig) -> None:
    """Main entry point for a prefill worker process."""
    worker = PrefillWorker(config)
    worker.serve()
