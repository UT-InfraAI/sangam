"""Colocated worker: does both prefill and decode on the same GPU.

Each colocated worker loads the model, runs prefill forward passes to produce
KV cache, then decodes from those locally (no NCCL transfer).

The worker uses a token budget to interleave prefill and decode:
  - Decode is prioritised (continuous batching).
  - Each iteration, decode consumes tokens from the budget first.
  - If remaining budget >= next request's prefill cost AND enough KV pages
    are free, the worker runs a prefill and admits the request to decode.
  - A priority queue ordered by original arrival time ensures fairness.

In hybrid mode (enable_kv_receive=True), the worker additionally serves
DecodeWorkerService RPCs (ReceiveKVCache, EnqueueDecode, FreeDecodeState)
so it can receive KV caches from prefill workers via NCCL and decode
externally-prefilled requests alongside locally-prefilled ones.
"""

import heapq
import math
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import grpc
import torch

from sangam.logger import init_logger
from sangam.metrics.constants import WorkerStateTimeline
from sangam.kv_cache.kv_cache_transfer import (
    recv_paged_kv_layer,
    recv_paged_kv_layer_drain,
)
from sangam.model.model_runner import (
    MixedBatchItem,
    pack_mixed_batch,
    paged_attention_context,
    run_mixed_paged_forward,
)
from sangam.kv_cache.paged_kv_cache import (
    PagedKVPool,
    RequestKVState,
)
from sangam.model.cuda_graph_runner import DecodeCudaGraphRunner, DecodeGraphItem
from sangam.metrics.utils.cuda_timer import DurationTimer
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.sampler import Sampler, SamplingRequest
from sangam.sampling_parameters import SamplingParameters
from sangam.types import PrefillQueuePolicy, WorkerType
from sangam.batch import Batch, BatchRequestUpdate
from sangam.worker.base_worker import BaseWorker, create_scheduler_callback_stub
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
from sangam.worker.worker_config import ColocatedWorkerConfig

logger = init_logger(__name__)


@dataclass
class ActiveDecodeRequest:
    """A request actively being decoded in the continuous batching loop."""

    request_id: str
    sequence_ids: (
        torch.Tensor | list[int]
    )  # (1, seq_len) on device, or list when in ready queue
    block_start: int
    block_end: int
    block_index: int
    request_seed: int
    sampling_parameters: SamplingParameters
    mask_id: int
    step_index: int = 0
    decode_start_time: float = field(default_factory=time.time)
    prefill_duration: float = 0.0
    prefill_queue_wait_duration: float = 0.0
    prefill_num_unmasked_tokens: int = 0
    decode_queue_wait_duration: float = 0.0
    num_forward_evals: int = 0
    ready_time: float | None = None


@dataclass(order=True)
class WaitingRequest:
    """A request waiting to be prefilled, ordered by queue priority."""

    priority: tuple[int, float, str] | tuple[float, str]
    enqueue_req: object = field(compare=False)  # EnqueueColocatedRequest


class ColocatedWorkerServicer(
    sangam_pb2_grpc.ColocatedWorkerServiceServicer,
    sangam_pb2_grpc.DecodeWorkerServiceServicer,
):
    """gRPC servicer for colocated prefill + decode operations.

    In hybrid mode (gpu_resources is not None), also serves DecodeWorkerService
    RPCs to receive KV caches from prefill workers via NCCL.
    """

    def __init__(
        self,
        config: ColocatedWorkerConfig,
        *,
        model: torch.nn.Module,
        device: torch.device,
        gpu_resources: WorkerSharedGpuResources | None = None,
    ):
        self.model = model
        self.device = device
        self.worker_id = config.worker_id
        self.dist_rank = config.dist_rank
        self._max_batch_size = config.max_batch_size
        self._max_tokens_per_iteration = config.max_tokens_per_iteration
        self._deficit_tokens: int = 0
        self._prefill_queue_policy = PrefillQueuePolicy(config.prefill_queue_policy)
        self._op_metrics_enabled = (
            config.enable_metrics and config.enable_operation_metrics
        )
        self._op_metrics_layer_id = resolve_profile_layer_id(
            enabled=self._op_metrics_enabled,
            requested_layer_id=config.op_metrics_layer_id,
            num_layers=self.model.num_layers,
        )

        # CUDA graph support for decode-only batches. The runner is built once
        # at startup on the GPU thread (see proactive capture below); it needs
        # the KV pool, which is lazy in pure colocated mode. _enable_cuda_graphs
        # gates construction.
        self._enable_cuda_graphs = config.enable_cuda_graphs
        self._cuda_graph_batch_sizes = config.cuda_graph_batch_sizes
        self._block_length = config.block_length
        self._graph_runner: DecodeCudaGraphRunner | None = None

        # Per-request decode state (page table)
        self._state: dict[str, RequestKVState] = {}

        # KV pages reserved at admission time by _collect_prefill_batch and
        # consumed by _do_run_mixed_iteration_local. Reservation under
        # _state_lock makes admission atomic with respect to ReceiveKVCache.
        self._reserved_prefill_kv: dict[str, RequestKVState] = {}

        # Hybrid mode: eager KV pool from shared GPU resources
        # Pure colocated mode: lazy KV pool initialized on first prefill
        if gpu_resources is not None:
            self._kv_pool: PagedKVPool | None = gpu_resources.kv_pool
            self._flashinfer_workspace = gpu_resources.flashinfer_workspace
            # Per-source-rank recv queues and CUDA streams. Mirrors
            # decode_worker.py so that an NCCL recv blocked on a slow
            # send from one prefill rank cannot stall recvs from another.
            self._recv_queues: dict[int, GpuWorkQueue] | None = {}
            self._recv_stream_pool: dict[int, torch.cuda.Stream | None] | None = {}
            self._recv_init_lock: threading.Lock | None = threading.Lock()
            self._state_lock: threading.Lock | None = threading.Lock()
        else:
            self._kv_pool = None
            self._flashinfer_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self._recv_queues = None
            self._recv_stream_pool = None
            self._recv_init_lock = None
            self._state_lock = None

        self._kv_page_size = config.kv_page_size
        self._kv_max_pages = config.kv_max_pages
        self._kv_dtype = config.kv_dtype

        # GPU work queue — serializes all GPU operations
        self._gpu_queue = GpuWorkQueue(device=self.device)
        self._gpu_queue.start()

        # Incoming RPC queue (drained into waiting heap)
        self._incoming_queue: queue.Queue = queue.Queue()

        # External decode queue: requests arriving via ReceiveKVCache/EnqueueDecode
        # (thread-safe, drained by main loop into _decode_ready_queue)
        self._external_decode_queue: queue.Queue = queue.Queue()

        # Priority heap of requests waiting for prefill (ordered by arrival_time)
        self._waiting_heap: list[WaitingRequest] = []

        # Requests whose prefill finished but have not yet been admitted to decode.
        self._decode_ready_queue: list[ActiveDecodeRequest] = []

        # Active decode batch
        self._active_batch: list[ActiveDecodeRequest] = []
        self._sampler = Sampler()

        # Scheduler callback stub
        _, self._scheduler_stub = create_scheduler_callback_stub(
            config.scheduler_address,
            config.max_grpc_message_length,
        )

        # Proactively capture decode CUDA graphs at startup (on the GPU thread)
        # so the first live decode batch does not pay the capture cost. The
        # blocking submit warms the worker before the main loop serves any
        # batch. No-op when CUDA graphs are disabled; failures fall back to
        # eager decode inside _ensure_graph_runner.
        if self._enable_cuda_graphs:
            self._gpu_queue.submit(self._ensure_graph_runner)

        # Start main loop
        self._main_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="colocated-main"
        )
        self._main_thread.start()

    @staticmethod
    def _block_is_complete(block_tokens: torch.Tensor, mask_id: int) -> bool:
        return (block_tokens == mask_id).sum().item() == 0

    @contextmanager
    def _state_guard(self):
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            yield
            return
        with lock:
            yield

    @contextmanager
    def _paged_attention_context(self, paged_state):
        with paged_attention_context(self.model, paged_state):
            yield

    def _ensure_kv_pool(self) -> PagedKVPool:
        """Lazily construct the KV pool in pure colocated mode.

        In hybrid mode the pool is provided eagerly via shared GPU resources.
        """
        if self._kv_pool is None:
            self._kv_pool = PagedKVPool(
                num_layers=self.model.num_layers,
                max_pages=self._kv_max_pages,
                page_size=self._kv_page_size,
                num_kv_heads=self.model.num_kv_heads,
                head_dim=self.model.head_dim,
                device=self.device,
                dtype=self._kv_dtype,
            )
        return self._kv_pool

    def _ensure_graph_runner(self) -> DecodeCudaGraphRunner | None:
        """Lazily build and capture the decode CUDA-graph runner.

        Runs on the GPU-queue thread (called from within the GPU-bound
        iteration), so the captured graph and replays share one stream. On any
        failure the feature is disabled (runner left None) and decode falls back
        to the eager path. Capture reserves a dummy page from the pool.
        """
        if not self._enable_cuda_graphs:
            return None
        if self._graph_runner is not None:
            return self._graph_runner
        if self.device.type != "cuda":
            self._enable_cuda_graphs = False
            return None
        try:
            pool = self._ensure_kv_pool()
            runner = DecodeCudaGraphRunner(
                model=self.model,
                pool=pool,
                flashinfer_workspace=self._flashinfer_workspace,
                device=self.device,
                block_length=self._block_length,
                num_q_heads=self.model.num_q_heads,
                buckets=self._cuda_graph_batch_sizes,
                rope_max_pos=pool.max_pages * pool.page_size,
            )
            # Capture (incl. the dummy-page reservation that mutates the pool
            # allocator) under the state guard so it is serialized against
            # concurrent ReceiveKVCache allocations in hybrid mode. No-op lock
            # in pure colocated mode. Capture is pure compute (no NCCL).
            with self._state_guard():
                runner.maybe_capture()
            self._graph_runner = runner
        except Exception:
            logger.exception("CUDA graph capture failed; falling back to eager decode")
            self._enable_cuda_graphs = False
            self._graph_runner = None
        return self._graph_runner

    def _make_batch_report(
        self,
        *,
        batch_phase: int,
        batch_size: int,
        prompt_len: int,
        gen_len: int,
        batch_start_time: float,
        batch_end_time: float,
        num_unmasked_tokens: int,
        sampling_duration: float,
        batch_op_attn_time: float | None,
        batch_op_mlp_time: float | None,
        batch_op_qkv_time: float | None,
        request_updates: list[BatchRequestUpdate],
        worker_state_after: sangam_pb2.WorkerStateSnapshot,
    ) -> None:
        kv_total_pages, kv_used_pages, kv_free_pages = self._current_kv_page_stats()
        self._report_batch(
            Batch(
                worker_id=self.worker_id,
                worker_type=WorkerType.COLOCATED.value,
                batch_size=batch_size,
                prompt_len=prompt_len,
                gen_len=gen_len,
                batch_start_time=batch_start_time,
                batch_end_time=batch_end_time,
                kv_total_pages=kv_total_pages,
                kv_used_pages=kv_used_pages,
                kv_free_pages=kv_free_pages,
                num_unmasked_tokens=num_unmasked_tokens,
                batch_phase=batch_phase,
                sampling_duration=sampling_duration,
                batch_op_attn_time=batch_op_attn_time,
                batch_op_mlp_time=batch_op_mlp_time,
                batch_op_qkv_time=batch_op_qkv_time,
                request_updates=request_updates,
                worker_state_after=worker_state_after,
            )
        )

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
        if self._kv_pool is None:
            return 0, 0, 0
        with self._state_guard():
            total = self._kv_pool.max_pages
            free = self._kv_pool.allocator.num_free
            used = self._kv_pool.allocator.num_used
        return total, used, free

    def _build_state_snapshot(
        self, state: WorkerStateTimeline
    ) -> sangam_pb2.WorkerStateSnapshot:
        kv_total_pages, kv_used_pages, kv_free_pages = self._current_kv_page_stats()
        return sangam_pb2.WorkerStateSnapshot(
            state=getattr(sangam_pb2, f"WORKER_STATE_{state.value.upper()}"),
            timestamp=time.time(),
            waiting_queue_depth=(
                len(self._waiting_heap) + self._incoming_queue.qsize()
            ),
            active_batch_size=len(self._active_batch)
            if state is WorkerStateTimeline.BUSY
            else 0,
            kv_total_pages=kv_total_pages,
            kv_used_pages=kv_used_pages,
            kv_free_pages=kv_free_pages,
            deficit_tokens=self._deficit_tokens,
        )

    def _current_worker_state(self) -> WorkerStateTimeline:
        if self._active_batch:
            return WorkerStateTimeline.BUSY
        if (
            self._waiting_heap
            or self._incoming_queue.qsize() > 0
            or self._decode_ready_queue
        ):
            return WorkerStateTimeline.QUEUED
        return WorkerStateTimeline.IDLE

    def shutdown(self) -> None:
        self._gpu_queue.shutdown()
        if self._recv_queues is not None:
            for recv_queue in self._recv_queues.values():
                recv_queue.shutdown()

    def _get_recv_queue(self, src_rank: int) -> GpuWorkQueue:
        recv_queue = self._recv_queues.get(src_rank)
        if recv_queue is None:
            with self._recv_init_lock:
                recv_queue = self._recv_queues.get(src_rank)
                if recv_queue is None:
                    recv_queue = GpuWorkQueue(device=self.device)
                    recv_queue.start()
                    self._recv_queues[src_rank] = recv_queue
        return recv_queue

    def _get_recv_stream(self, src_rank: int) -> torch.cuda.Stream | None:
        if src_rank not in self._recv_stream_pool:
            with self._recv_init_lock:
                if src_rank not in self._recv_stream_pool:
                    self._recv_stream_pool[src_rank] = (
                        torch.cuda.Stream(device=self.device)
                        if self.device.type == "cuda"
                        else None
                    )
        return self._recv_stream_pool[src_rank]

    # ----- gRPC handlers -----

    def EnqueueRequest(
        self,
        request: sangam_pb2.EnqueueColocatedRequest,
        context,
    ) -> sangam_pb2.EnqueueColocatedResponse:
        """Enqueue a request, return immediately."""
        self._incoming_queue.put(request)
        snapshot = self._build_state_snapshot(self._current_worker_state())
        return sangam_pb2.EnqueueColocatedResponse(
            success=True, accepted_state=snapshot
        )

    # ----- DecodeWorkerService gRPC handlers (hybrid mode) -----

    def ReceiveKVCache(
        self, request: sangam_pb2.ReceiveKVCacheRequest, context
    ) -> sangam_pb2.ReceiveKVCacheResponse:
        """Receive KV cache from prefill worker via NCCL (hybrid mode)."""
        if self._recv_queues is None:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("KV receive not enabled on this worker")
            return sangam_pb2.ReceiveKVCacheResponse(success=False)
        try:
            src_rank = getattr(request, "src_rank", 0)
            return self._get_recv_queue(src_rank).submit(
                self._do_receive_kv_cache,
                request,
                self._get_recv_stream(src_rank),
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sangam_pb2.ReceiveKVCacheResponse(success=False)

    def EnqueueDecode(
        self, request: sangam_pb2.EnqueueDecodeRequest, context
    ) -> sangam_pb2.EnqueueDecodeResponse:
        """Enqueue a decode request from the scheduler (hybrid mode)."""
        self._external_decode_queue.put(request)
        snapshot = self._build_state_snapshot(self._current_worker_state())
        return sangam_pb2.EnqueueDecodeResponse(success=True, accepted_state=snapshot)

    def FreeDecodeState(
        self, request: sangam_pb2.FreeDecodeStateRequest, context
    ) -> sangam_pb2.FreeDecodeStateResponse:
        """Free all stored state for a request."""
        try:
            self._gpu_queue.submit(self._do_free_decode_state, request)
        except RuntimeError:
            return sangam_pb2.FreeDecodeStateResponse(success=False)
        return sangam_pb2.FreeDecodeStateResponse(success=True)

    def _do_receive_kv_cache(
        self,
        request: sangam_pb2.ReceiveKVCacheRequest,
        recv_stream: torch.cuda.Stream | None = None,
    ) -> sangam_pb2.ReceiveKVCacheResponse:
        """NCCL receive of KV cache pages from a prefill worker."""
        req_id = request.request_id
        src_rank = request.src_rank
        num_layers = request.num_layers
        seq_length = request.seq_length

        logger.debug(f"[{req_id}] ReceiveKVCache from rank {src_rank}")

        try:
            with self._state_guard():
                page_ids, last_page_len = self._kv_pool.allocate(seq_length)
        except RuntimeError as alloc_exc:
            # The prefill side has already queued the matching NCCL send. If
            # we return success=False without posting recvs, the send blocks
            # forever on send_stream.synchronize(), permanently jamming the
            # per-destination GPU send queue. Drain the recvs into a
            # throwaway buffer so the send completes, then report failure.
            num_pages = math.ceil(seq_length / self._kv_pool.page_size)
            logger.warning(
                f"[{req_id}] KV pool exhausted; draining "
                f"{num_layers}x{num_pages} pages of recvs from rank "
                f"{src_rank} to free the sender ({alloc_exc})"
            )
            try:
                stream_ctx = (
                    torch.cuda.stream(recv_stream)
                    if recv_stream is not None
                    else nullcontext()
                )
                with stream_ctx:
                    template = self._kv_pool.kv_data[0]
                    for _ in range(num_layers):
                        recv_paged_kv_layer_drain(
                            template=template,
                            num_pages=num_pages,
                            src_rank=src_rank,
                        )
                if recv_stream is not None:
                    recv_stream.synchronize()
            except Exception:
                logger.exception(
                    f"[{req_id}] Failed to drain orphaned NCCL recv from "
                    f"rank {src_rank}"
                )
            return sangam_pb2.ReceiveKVCacheResponse(success=False)

        try:
            stream_ctx = (
                torch.cuda.stream(recv_stream)
                if recv_stream is not None
                else nullcontext()
            )
            with stream_ctx:
                for layer_idx in range(num_layers):
                    recv_paged_kv_layer(
                        kv_layer=self._kv_pool.kv_data[layer_idx],
                        page_ids=page_ids,
                        src_rank=src_rank,
                    )
            if recv_stream is not None:
                recv_stream.synchronize()
        except Exception:
            with self._state_guard():
                self._kv_pool.free(page_ids)
            raise

        kv_state = RequestKVState(
            page_ids=page_ids,
            seq_len=seq_length,
            last_page_len=last_page_len,
        )
        with self._state_guard():
            self._state[req_id] = kv_state
            free_pages = self._kv_pool.allocator.num_free

        # Auto-enqueue for decode if sequence data is included
        if getattr(request, "sequence_ids", None):
            decode_enqueue_time = request.decode_enqueue_time
            if decode_enqueue_time <= 0:
                decode_enqueue_time = time.time()
            enqueue_req = sangam_pb2.EnqueueDecodeRequest(
                request_id=req_id,
                sequence_ids=request.sequence_ids,
                block_start=request.block_start,
                block_end=request.block_end,
                block_index=request.block_index,
                request_seed=request.request_seed,
                mask_id=request.mask_id,
                arrival_time=request.arrival_time,
                decode_enqueue_time=decode_enqueue_time,
            )
            enqueue_req.sampling_parameters.CopyFrom(request.sampling_parameters)
            self._external_decode_queue.put(enqueue_req)

        logger.debug(
            f"[{req_id}] KV cache received and paged: {num_layers} layers, "
            f"seq_len={seq_length}, {len(page_ids)} pages "
            f"({free_pages} free)"
        )
        return sangam_pb2.ReceiveKVCacheResponse(success=True)

    def _do_free_decode_state(self, request: sangam_pb2.FreeDecodeStateRequest) -> None:
        req_id = request.request_id
        self._do_free_state(req_id)
        logger.debug(f"[{req_id}] Freed decode state (via FreeDecodeState)")

    # ----- Main loop -----

    def _drain_incoming(self) -> None:
        """Non-blocking drain from RPC queue into the priority waiting heap."""
        self._drain_incoming_inner()

    def _drain_incoming_inner(self) -> None:
        while True:
            try:
                req = self._incoming_queue.get_nowait()
            except queue.Empty:
                break
            heapq.heappush(
                self._waiting_heap,
                WaitingRequest(
                    priority=self._prefill_queue_priority(req),
                    enqueue_req=req,
                ),
            )

    def _prefill_queue_priority(
        self, request: sangam_pb2.EnqueueColocatedRequest
    ) -> tuple[int, float, str] | tuple[float, str]:
        return compute_prefill_queue_priority(request, self._prefill_queue_policy)

    def _drain_external_decodes(self) -> None:
        """Drain externally-prefilled requests into the decode ready queue.

        These arrive via ReceiveKVCache (auto-enqueue) or EnqueueDecode RPCs.
        Their KV is already in the pool — they skip local prefill and go
        straight to decode admission.
        """
        while True:
            try:
                enqueue_req = self._external_decode_queue.get_nowait()
            except queue.Empty:
                break
            # Verify KV state exists for this request
            req_id = enqueue_req.request_id
            lock = self._state_lock
            if lock is not None:
                with lock:
                    state_ready = req_id in self._state
            else:
                state_ready = req_id in self._state
            if not state_ready:
                logger.error(
                    f"[{req_id}] External decode request but no KV state, skipping"
                )
                continue
            self._decode_ready_queue.append(
                ActiveDecodeRequest(
                    request_id=req_id,
                    sequence_ids=list(enqueue_req.sequence_ids),
                    block_start=enqueue_req.block_start,
                    block_end=enqueue_req.block_end,
                    block_index=enqueue_req.block_index,
                    request_seed=enqueue_req.request_seed,
                    step_index=0,
                    sampling_parameters=SamplingParameters.from_proto(
                        enqueue_req.sampling_parameters
                        if enqueue_req.HasField("sampling_parameters")
                        else None
                    ),
                    mask_id=enqueue_req.mask_id,
                    ready_time=time.time(),
                )
            )
            logger.debug(
                f"[{req_id}] External decode request queued for batch admission"
            )

    def _main_loop(self) -> None:
        """Main loop: run one mixed iteration over decode and admitted prefills."""
        while True:
            self._drain_incoming()
            self._drain_external_decodes()
            self._admit_ready_decode_requests()

            # If nothing to do, block until a request arrives
            if (
                not self._active_batch
                and not self._waiting_heap
                and not self._decode_ready_queue
            ):
                try:
                    req = self._incoming_queue.get(timeout=0.01)
                    self._incoming_queue.put(req)
                except queue.Empty:
                    pass
                continue

            prefill_batch = self._collect_prefill_batch()

            decode_reqs = [req for req in self._active_batch if req.step_index > 0]
            if not decode_reqs and not prefill_batch:
                continue

            self._run_batch(prefill_batch, decode_reqs)
            self._evict_completed()
            self._drain_incoming()
            self._drain_external_decodes()
            self._admit_ready_decode_requests()

    def _evict_completed(self) -> None:
        """Remove completed requests from active batch.

        Pool pages and `_state` entries are freed eagerly inside `_run_batch`
        before the batch report is sent, so the scheduler observes a freed
        pool when it processes `block_completed=True`. This method only
        prunes the local active-batch list.
        """
        completed = []
        continuing = []
        for req in self._active_batch:
            block_tokens = req.sequence_ids[:, req.block_start : req.block_end]
            if self._block_is_complete(block_tokens, req.mask_id):
                completed.append(req)
            else:
                continuing.append(req)
        self._active_batch = continuing

        for req in completed:
            decode_duration = time.time() - req.decode_start_time
            logger.debug(
                f"[{req.request_id}] Block {req.block_index}: complete, "
                f"{req.num_forward_evals} fwd evals, {decode_duration:.3f}s"
            )

    def _collect_prefill_batch(self) -> list:
        """Select waiting prefills that fit this iteration's token and KV budget.

        KV pages for admitted requests are reserved atomically under
        `_state_lock` so admission is serialized against concurrent
        `ReceiveKVCache` allocations from remote prefill workers. Reserved
        states are stashed in `self._reserved_prefill_kv` keyed by request_id
        and consumed by `_do_run_mixed_iteration_local`.
        """
        decode_tokens = sum(r.block_end - r.block_start for r in self._active_batch)
        iteration_budget = self._max_tokens_per_iteration - decode_tokens
        prior_deficit = self._deficit_tokens
        remaining_budget = iteration_budget + prior_deficit

        if not self._waiting_heap:
            self._deficit_tokens = 0
            return []

        batch_to_prefill: list = []
        prefill_tokens_so_far = 0
        memory_blocked = False
        waiting_requests: list[WaitingRequest] = []

        while self._waiting_heap:
            waiting_requests.append(heapq.heappop(self._waiting_heap))

        # In hybrid mode (lock present, pool eagerly initialized) we reserve
        # KV pages atomically under the lock so admission is serialized with
        # ReceiveKVCache. In pure colocated mode the pool is lazily built on
        # the GPU thread and there is no concurrent receive path; the
        # admission check is purely budget-based and pages are allocated
        # later inside _do_run_mixed_iteration_local.
        reserve_under_lock = (
            getattr(self, "_state_lock", None) is not None and self._kv_pool is not None
        )
        if reserve_under_lock and not hasattr(self, "_reserved_prefill_kv"):
            self._reserved_prefill_kv = {}
        pages_needed_so_far = 0

        with self._state_guard():
            for idx, waiting in enumerate(waiting_requests):
                req = waiting.enqueue_req
                seq_len = len(req.sequence_ids)
                new_total = prefill_tokens_so_far + seq_len

                # Strict head-of-line admission: if this request cannot be
                # admitted (budget or KV/memory capacity), stop the scan and
                # requeue it together with everything behind it. A
                # higher-priority request must never be leapfrogged by a
                # smaller one queued later.
                allow_oversized_first = not self._active_batch and not batch_to_prefill
                has_budget = new_total <= remaining_budget or allow_oversized_first
                if not has_budget:
                    for blocked in waiting_requests[idx:]:
                        heapq.heappush(self._waiting_heap, blocked)
                    break

                if reserve_under_lock:
                    try:
                        page_ids, last_page_len = self._kv_pool.allocate(seq_len)
                    except RuntimeError:
                        # KV pool exhausted (likely drained by concurrent
                        # ReceiveKVCache). Requeue this and the rest, try again
                        # later.
                        memory_blocked = True
                        for blocked in waiting_requests[idx:]:
                            heapq.heappush(self._waiting_heap, blocked)
                        break
                    self._reserved_prefill_kv[req.request_id] = RequestKVState(
                        page_ids=page_ids,
                        seq_len=seq_len,
                        last_page_len=last_page_len,
                    )
                else:
                    # Pure mode: keep the original num_free heuristic.
                    new_pages = pages_needed_so_far + math.ceil(
                        seq_len / self._kv_page_size
                    )
                    has_memory = (
                        self._kv_pool is None
                        or self._kv_pool.allocator.num_free >= new_pages
                    )
                    if not has_memory:
                        memory_blocked = True
                        for blocked in waiting_requests[idx:]:
                            heapq.heappush(self._waiting_heap, blocked)
                        break
                    pages_needed_so_far = new_pages

                batch_to_prefill.append(req)
                prefill_tokens_so_far = new_total

        if memory_blocked:
            # Memory stalled admission, not lack of token budget: do not add
            # this iteration's fresh budget to the deficit. Carry over the
            # prior deficit, debited by any portion an admitted request already
            # spent from it.
            overspend = max(0, prefill_tokens_so_far - iteration_budget)
            self._deficit_tokens = max(0, prior_deficit - overspend)
        else:
            self._deficit_tokens = max(0, remaining_budget - prefill_tokens_so_far)
        return batch_to_prefill

    def _try_batch_prefill(self) -> None:
        """Run one prefill-only iteration over selected waiting requests."""
        batch_to_prefill = self._collect_prefill_batch()
        if not batch_to_prefill:
            return
        self._run_batch(batch_to_prefill, [])

    def _admit_ready_decode_requests(self) -> None:
        """Move prefills that are ready into the active decode batch."""
        while (
            self._decode_ready_queue and len(self._active_batch) < self._max_batch_size
        ):
            ready_req = self._decode_ready_queue.pop(0)
            decode_queue_wait_duration = max(0.0, time.time() - ready_req.ready_time)
            ready_req.sequence_ids = torch.tensor(
                [ready_req.sequence_ids],
                dtype=torch.long,
                device=self.device,
            )
            ready_req.step_index = 1
            ready_req.decode_start_time = time.time()
            ready_req.decode_queue_wait_duration = decode_queue_wait_duration
            ready_req.ready_time = None
            self._active_batch.append(ready_req)
            logger.debug(
                f"[{ready_req.request_id}] Admitted to decode batch "
                f"(batch_size={len(self._active_batch)})"
            )

    def _run_batch(
        self,
        prefill_batch: list,
        decode_reqs: list[ActiveDecodeRequest],
    ) -> None:
        """Run one packed colocated iteration over prefills and/or decodes."""
        if not prefill_batch and not decode_reqs:
            return
        if prefill_batch and decode_reqs:
            batch_phase = sangam_pb2.BATCH_PHASE_MIXED
        elif prefill_batch:
            batch_phase = sangam_pb2.BATCH_PHASE_PREFILL
        else:
            batch_phase = sangam_pb2.BATCH_PHASE_DECODE

        dequeue_time = time.time()

        queue_wait_durations = []
        for enqueue_req in prefill_batch:
            wait = 0.0
            if enqueue_req.prefill_enqueue_time > 0:
                wait = max(0.0, dequeue_time - enqueue_req.prefill_enqueue_time)
            queue_wait_durations.append(wait)

        total_prefill_tokens = sum(
            len(enqueue_req.sequence_ids) for enqueue_req in prefill_batch
        )
        total_decode_tokens = sum(
            req.block_end - req.block_start for req in decode_reqs
        )
        stored_req_ids: list[str] = []

        try:
            batch_start_time = time.time()
            (
                prefill_results,
                decode_updates,
                total_unmasked,
                sampling_duration,
                operation_metrics_seconds,
            ) = self._gpu_queue.submit(
                self._do_run_mixed_iteration_local, prefill_batch, decode_reqs
            )
            batch_end_time = time.time()
            (
                batch_op_attn_time,
                batch_op_mlp_time,
                batch_op_qkv_time,
            ) = batch_operation_metric_values(
                enabled=getattr(self, "_op_metrics_enabled", False),
                operation_metrics_seconds=operation_metrics_seconds,
            )
            prefill_duration = batch_end_time - batch_start_time
            request_updates: list[BatchRequestUpdate] = []
            for enqueue_req, result, wait in zip(
                prefill_batch, prefill_results, queue_wait_durations, strict=True
            ):
                req_id = enqueue_req.request_id
                sequence = result["sampled_sequence"]
                block_tokens = sequence[enqueue_req.block_start : enqueue_req.block_end]
                block_completed = all(
                    token_id != enqueue_req.mask_id for token_id in block_tokens
                )
                request_updates.append(
                    BatchRequestUpdate(
                        request_id=req_id,
                        block_index=enqueue_req.block_index,
                        success=True,
                        updated_sequence=sequence,
                        num_unmasked_tokens=result["num_unmasked_tokens"],
                        num_forward_evals_in_batch_phase=1,
                        request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                        prefill_duration=prefill_duration,
                        prefill_queue_wait_duration=wait,
                        block_completed=block_completed,
                    )
                )
                if block_completed:
                    with self._state_guard():
                        self._kv_pool.free(result["kv_state"].page_ids)
                else:
                    with self._state_guard():
                        self._state[req_id] = result["kv_state"]
                    stored_req_ids.append(req_id)
                    self._decode_ready_queue.append(
                        ActiveDecodeRequest(
                            request_id=req_id,
                            sequence_ids=sequence,
                            block_start=enqueue_req.block_start,
                            block_end=enqueue_req.block_end,
                            block_index=enqueue_req.block_index,
                            request_seed=enqueue_req.request_seed,
                            step_index=0,
                            sampling_parameters=SamplingParameters.from_proto(
                                enqueue_req.sampling_parameters
                                if enqueue_req.HasField("sampling_parameters")
                                else None
                            ),
                            mask_id=enqueue_req.mask_id,
                            prefill_duration=prefill_duration,
                            prefill_queue_wait_duration=wait,
                            prefill_num_unmasked_tokens=result["num_unmasked_tokens"],
                            ready_time=time.time(),
                        )
                    )
                logger.debug(f"[{req_id}] Prefill complete")

            updates_by_request = {
                update.request_id: update for update in decode_updates
            }
            processed_decode_reqs = [
                req for req in decode_reqs if req.request_id in updates_by_request
            ]
            for req in processed_decode_reqs:
                req.step_index += 1
                req.num_forward_evals += 1
            for req in decode_reqs:
                block_tokens = req.sequence_ids[:, req.block_start : req.block_end]
                is_complete = self._block_is_complete(block_tokens, req.mask_id)
                update = updates_by_request.get(req.request_id)
                if update is None:
                    if not is_complete:
                        continue
                    update = BatchRequestUpdate(
                        request_id=req.request_id,
                        block_index=req.block_index,
                        success=True,
                        updated_sequence=req.sequence_ids.squeeze(0).tolist(),
                        num_unmasked_tokens=0,
                        num_forward_evals_in_batch_phase=req.num_forward_evals,
                        request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                    )
                    decode_updates.append(update)
                update.num_forward_evals_in_batch_phase = req.num_forward_evals
                update.request_phase = sangam_pb2.BATCH_PHASE_DECODE
                update.decode_queue_wait_duration = req.decode_queue_wait_duration
                update.prefill_duration = req.prefill_duration
                update.prefill_queue_wait_duration = req.prefill_queue_wait_duration
                if is_complete:
                    update.block_completed = True
                    update.decode_duration = batch_end_time - req.decode_start_time
                    # Free pool pages before sending the batch report so the
                    # report's kv_free_pages snapshot and the scheduler's
                    # release_decode_reservation (triggered by
                    # block_completed=True) observe a consistent state. Without
                    # this, the scheduler could pick this worker for a new
                    # ReceiveKVCache before _evict_completed runs the actual
                    # free, leading to KV page OOM and a transfer failure.
                    self._do_free_state(req.request_id)

            request_updates.extend(decode_updates)
            self._make_batch_report(
                batch_phase=batch_phase,
                batch_size=len(prefill_batch) + len(processed_decode_reqs),
                prompt_len=total_prefill_tokens,
                gen_len=total_decode_tokens,
                batch_start_time=batch_start_time,
                batch_end_time=batch_end_time,
                num_unmasked_tokens=total_unmasked,
                sampling_duration=sampling_duration,
                batch_op_attn_time=batch_op_attn_time,
                batch_op_mlp_time=batch_op_mlp_time,
                batch_op_qkv_time=batch_op_qkv_time,
                request_updates=request_updates,
                worker_state_after=self._build_state_snapshot(
                    self._current_worker_state()
                ),
            )

            logger.debug(
                "Colocated batch complete: %s prefills, %s decodes, "
                "prompt_tokens=%s, gen_tokens=%s, total=%.3fs",
                len(prefill_batch),
                len(processed_decode_reqs),
                total_prefill_tokens,
                total_decode_tokens,
                batch_end_time - batch_start_time,
            )

        except Exception:
            logger.exception(
                "Colocated batch failed for %s prefills and %s decodes",
                len(prefill_batch),
                len(decode_reqs),
            )
            # Free any KV reservations that were not consumed by the GPU thread
            # (e.g. exception raised before or during _do_run_mixed_iteration_local).
            reserved_map = getattr(self, "_reserved_prefill_kv", None)
            if reserved_map:
                with self._state_guard():
                    for enqueue_req in prefill_batch:
                        reserved = reserved_map.pop(enqueue_req.request_id, None)
                        if reserved is not None and self._kv_pool is not None:
                            self._kv_pool.free(reserved.page_ids)
            for req_id in stored_req_ids:
                with self._state_guard():
                    state = self._state.pop(req_id, None)
                    if state is not None and self._kv_pool is not None:
                        self._kv_pool.free(state.page_ids)
            failure_time = time.time()
            self._report_batch(
                Batch(
                    worker_id=self.worker_id,
                    worker_type=WorkerType.COLOCATED.value,
                    batch_size=len(prefill_batch) + len(decode_reqs),
                    prompt_len=total_prefill_tokens,
                    gen_len=total_decode_tokens,
                    batch_start_time=failure_time,
                    batch_end_time=failure_time,
                    kv_total_pages=0,
                    kv_used_pages=0,
                    kv_free_pages=0,
                    num_unmasked_tokens=0,
                    batch_phase=batch_phase,
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
                        for enqueue_req in prefill_batch
                    ]
                    + [
                        BatchRequestUpdate(
                            request_id=req.request_id,
                            block_index=req.block_index,
                            success=False,
                            updated_sequence=req.sequence_ids.squeeze(0).tolist(),
                            num_unmasked_tokens=0,
                            num_forward_evals_in_batch_phase=req.num_forward_evals,
                            request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                        )
                        for req in decode_reqs
                    ],
                    worker_state_after=self._build_state_snapshot(
                        self._current_worker_state()
                    ),
                )
            )

    # ----- GPU-bound implementations -----

    @torch.inference_mode()
    def _do_run_batched_prefill_local(
        self, batch: list
    ) -> tuple[list[dict], float, dict[str, float] | None]:
        """Run one prefill-only mixed iteration for compatibility tests."""
        (
            prefill_results,
            _decode_updates,
            _total_unmasked,
            sampling_duration,
            operation_metrics_seconds,
        ) = self._do_run_mixed_iteration_local(batch, [])
        return prefill_results, sampling_duration, operation_metrics_seconds

    @torch.inference_mode()
    def _do_run_mixed_iteration_local(
        self,
        prefill_batch: list,
        decode_reqs: list[ActiveDecodeRequest],
    ) -> tuple[
        list[dict],
        list[BatchRequestUpdate],
        int,
        float,
        dict[str, float] | None,
    ]:
        """Run one packed forward over prefills and decodes."""
        if not prefill_batch and not decode_reqs:
            return [], [], 0, 0.0, None

        sequences = [list(req.sequence_ids) for req in prefill_batch]
        active_decode_reqs = [
            req
            for req in decode_reqs
            if not self._block_is_complete(
                req.sequence_ids[:, req.block_start : req.block_end], req.mask_id
            )
        ]
        block_lengths = [
            *(req.block_end - req.block_start for req in prefill_batch),
            *(req.block_end - req.block_start for req in active_decode_reqs),
        ]
        if not block_lengths:
            return [], [], 0, 0.0, None
        distinct_block_lengths = sorted(set(block_lengths))
        if len(distinct_block_lengths) != 1:
            req_ids = [req.request_id for req in prefill_batch] + [
                req.request_id for req in active_decode_reqs
            ]
            raise RuntimeError(
                "Mixed batch block lengths are unsupported: "
                f"block_lengths={distinct_block_lengths}, request_ids={req_ids}"
            )
        seq_lens = [len(s) for s in sequences]

        # In pure colocated mode the pool is lazily constructed on first
        # iteration. In hybrid mode it is built eagerly from shared GPU
        # resources, so this branch is a no-op.
        if self._kv_pool is None:
            self._ensure_kv_pool()

        # Consume KV reservations (hybrid mode) or allocate pages now (pure
        # mode fallback). In hybrid mode, _collect_prefill_batch reserved
        # pages atomically with respect to ReceiveKVCache under _state_lock,
        # so the lookup below is just bookkeeping.
        reserved_map: dict[str, RequestKVState] = getattr(
            self, "_reserved_prefill_kv", {}
        )
        kv_states: list[RequestKVState] = []
        all_page_ids: list = []
        try:
            with self._state_guard():
                for req, seq_len in zip(prefill_batch, seq_lens, strict=True):
                    reserved = reserved_map.pop(req.request_id, None)
                    if reserved is not None:
                        state = reserved
                    else:
                        page_ids, last_page_len = self._kv_pool.allocate(seq_len)
                        state = RequestKVState(
                            page_ids=page_ids,
                            seq_len=seq_len,
                            last_page_len=last_page_len,
                        )
                    kv_states.append(state)
                    all_page_ids.append(state.page_ids)
                decode_kv_states = [
                    self._state[req.request_id] for req in active_decode_reqs
                ]
        except Exception:
            with self._state_guard():
                for page_ids in all_page_ids:
                    self._kv_pool.free(page_ids)
            raise

        # Decode-only fast path: replay a captured CUDA graph instead of
        # launching the forward eagerly. Gated on no prefills, op-metrics off
        # (its per-layer timing needs host syncs incompatible with capture),
        # and the batch fitting a captured bucket. Falls back to eager
        # otherwise. The block_length match is enforced by can_run.
        use_graph = (
            not prefill_batch
            and active_decode_reqs
            and not getattr(self, "_op_metrics_enabled", False)
            and getattr(self, "_enable_cuda_graphs", False)
        )
        graph_runner = self._ensure_graph_runner() if use_graph else None
        if graph_runner is not None and graph_runner.can_run(
            len(active_decode_reqs), distinct_block_lengths[0]
        ):
            try:
                forward_result = graph_runner.run(
                    [
                        DecodeGraphItem(
                            token_ids=req.sequence_ids[
                                0, req.block_start : req.block_end
                            ].tolist(),
                            block_start=req.block_start,
                            active_kv_len=kv_state.seq_len,
                            page_ids=kv_state.page_ids,
                        )
                        for req, kv_state in zip(
                            active_decode_reqs, decode_kv_states, strict=True
                        )
                    ]
                )
            except Exception:
                with self._state_guard():
                    for page_ids in all_page_ids:
                        self._kv_pool.free(page_ids)
                raise
        else:
            packed_batch = pack_mixed_batch(
                items=[
                    MixedBatchItem(
                        request_id=req.request_id,
                        token_ids=sequence,
                        query_start=0,
                        query_end=seq_len,
                        active_kv_len=seq_len,
                        kv_state=kv_state,
                        phase="prefill",
                    )
                    for req, sequence, seq_len, kv_state in zip(
                        prefill_batch, sequences, seq_lens, kv_states, strict=True
                    )
                ]
                + [
                    MixedBatchItem(
                        request_id=req.request_id,
                        token_ids=req.sequence_ids[
                            0, req.block_start : req.block_end
                        ].tolist(),
                        query_start=req.block_start,
                        query_end=req.block_end,
                        active_kv_len=kv_state.seq_len,
                        kv_state=kv_state,
                        phase="decode",
                    )
                    for req, kv_state in zip(
                        active_decode_reqs, decode_kv_states, strict=True
                    )
                ],
                pool=self._kv_pool,
                num_q_heads=self.model.num_q_heads,
                device=self.device,
                workspace=self._flashinfer_workspace,
            )

            try:
                forward_result = self._run_mixed_forward(packed_batch)
            except Exception:
                with self._state_guard():
                    for page_ids in all_page_ids:
                        self._kv_pool.free(page_ids)
                raise

        prefill_results: list[dict] = []
        decode_updates: list[BatchRequestUpdate] = []
        total_unmasked_tokens = 0
        sampling_duration = 0.0
        try:
            timer_name = (
                "worker_sampling_mixed"
                if prefill_batch and decode_reqs
                else "worker_sampling_prefill"
                if prefill_batch
                else "worker_sampling_decode"
            )
            with DurationTimer(
                timer_name,
                use_cuda=forward_result.packed_logits.is_cuda,
                device=forward_result.packed_logits.device,
            ) as timer:
                prefill_token_ids_batch = [
                    torch.tensor([seq_i], dtype=torch.long, device=self.device)
                    for seq_i in sequences
                ]
                sampling_reqs = [
                    SamplingRequest(
                        request_id=req.request_id,
                        request_seed=req.request_seed,
                        token_ids=token_ids_i,
                        block_start_idx=req.block_start,
                        block_end_index=req.block_end,
                        step_index=0,
                        sampling_parameters=SamplingParameters.from_proto(
                            req.sampling_parameters
                            if req.HasField("sampling_parameters")
                            else None
                        ),
                        mask_token_id=req.mask_id,
                    )
                    for req, token_ids_i in zip(
                        prefill_batch, prefill_token_ids_batch, strict=True
                    )
                ] + [
                    SamplingRequest(
                        request_id=req.request_id,
                        request_seed=req.request_seed,
                        token_ids=req.sequence_ids,
                        block_start_idx=req.block_start,
                        block_end_index=req.block_end,
                        step_index=req.step_index,
                        sampling_parameters=req.sampling_parameters,
                        mask_token_id=req.mask_id,
                    )
                    for req in active_decode_reqs
                ]
                mixed_reqs = [*prefill_batch, *active_decode_reqs]
                mixed_token_ids = [
                    *prefill_token_ids_batch,
                    *[req.sequence_ids for req in active_decode_reqs],
                ]
                block_tokens_batch = torch.cat(
                    [
                        token_ids_i[:, req.block_start : req.block_end]
                        for req, token_ids_i in zip(
                            mixed_reqs,
                            mixed_token_ids,
                            strict=True,
                        )
                    ],
                    dim=0,
                )
                logits_batch = torch.cat(
                    [
                        item_logits[:, req.block_start : req.block_end, :]
                        if idx < len(prefill_batch)
                        else item_logits
                        for idx, (req, item_logits) in enumerate(
                            zip(mixed_reqs, forward_result.item_logits, strict=True)
                        )
                    ],
                    dim=0,
                )
                updated_block_tokens, num_unmasked_tokens = self._sampler.sample_batch(
                    reqs=sampling_reqs,
                    block_tokens=block_tokens_batch,
                    logits=logits_batch,
                )
                num_unmasked_tokens_list = num_unmasked_tokens.cpu().tolist()
                for i, token_ids_i in enumerate(prefill_token_ids_batch):
                    req = prefill_batch[i]
                    token_ids_i[:, req.block_start : req.block_end] = (
                        updated_block_tokens[i : i + 1]
                    )
                    total_unmasked_tokens += num_unmasked_tokens_list[i]
                    prefill_results.append(
                        {
                            "kv_state": kv_states[i],
                            "sampled_sequence": token_ids_i[0].tolist(),
                            "num_unmasked_tokens": num_unmasked_tokens_list[i],
                        }
                    )
                offset = len(prefill_batch)
                for idx, req in enumerate(active_decode_reqs, start=offset):
                    req.sequence_ids[:, req.block_start : req.block_end] = (
                        updated_block_tokens[idx : idx + 1]
                    )
                    num_unmasked = num_unmasked_tokens_list[idx]
                    total_unmasked_tokens += num_unmasked
                    decode_updates.append(
                        BatchRequestUpdate(
                            request_id=req.request_id,
                            block_index=req.block_index,
                            success=True,
                            updated_sequence=req.sequence_ids[0].tolist(),
                            num_unmasked_tokens=num_unmasked,
                            num_forward_evals_in_batch_phase=0,
                            request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                        )
                    )
            sampling_duration = timer.elapsed_s
        except Exception:
            # Free all allocated KV pages on sampling failure
            with self._state_guard():
                for page_ids in all_page_ids:
                    self._kv_pool.free(page_ids)
            raise

        # Do NOT free kv_states — they persist in pool for decode
        return (
            prefill_results,
            decode_updates,
            total_unmasked_tokens,
            sampling_duration,
            getattr(forward_result, "operation_metrics_seconds", None),
        )

    def _do_free_state(self, req_id: str) -> None:
        """Free paged KV state for a completed request."""
        with self._state_guard():
            state = self._state.pop(req_id, None)
            if state is not None and self._kv_pool is not None:
                self._kv_pool.free(state.page_ids)
        logger.debug(f"[{req_id}] Freed state")

    @torch.inference_mode()
    def _paged_forward_pass(
        self, reqs: list[ActiveDecodeRequest]
    ) -> tuple[int, list[BatchRequestUpdate], float, dict[str, float] | None]:
        """Decode-only wrapper around the shared mixed-batch implementation."""
        (
            _prefill_results,
            request_updates,
            total_unmasked,
            sampling_duration,
            operation_metrics_seconds,
        ) = self._do_run_mixed_iteration_local([], reqs)
        return (
            total_unmasked,
            request_updates,
            sampling_duration,
            operation_metrics_seconds,
        )


class ColocatedWorker(BaseWorker):
    """Concrete worker for colocated prefill + decode operations."""

    def __init__(self, config: ColocatedWorkerConfig):
        super().__init__(config)
        self._gpu_resources: WorkerSharedGpuResources | None = None

    @property
    def config(self) -> ColocatedWorkerConfig:
        return self._config  # type: ignore[return-value]

    def _get_worker_type(self) -> WorkerType:
        return WorkerType.COLOCATED

    def _registration_address(self) -> str:
        return f"localhost:{self.port}"

    def _registration_extra_fields(self) -> dict[str, int]:
        # CUDA-graph capture permanently reserves dummy pages out of the KV
        # pool. Advertise the reduced capacity so the scheduler's reservation
        # accounting matches what the worker can actually allocate.
        max_pages = self.config.kv_max_pages
        if self.config.enable_cuda_graphs:
            max_pages -= DecodeCudaGraphRunner.dummy_pages_per_req(
                self.config.block_length, self.config.kv_page_size
            )
        return {
            "max_pages": max_pages,
            "page_size": self.config.kv_page_size,
        }

    def _create_servicer(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> ColocatedWorkerServicer:
        gpu_resources = None
        if self.config.enable_kv_receive:
            gpu_resources = create_worker_shared_gpu_resources(
                model=model,
                device=device,
                kv_page_size=self.config.kv_page_size,
                kv_max_pages=self.config.kv_max_pages,
                kv_dtype=self.config.kv_dtype,
                zero_init=False,
            )
            self._gpu_resources = gpu_resources
        return ColocatedWorkerServicer(
            self.config,
            model=model,
            device=device,
            gpu_resources=gpu_resources,
        )

    def _add_servicer_to_server(self, servicer: object, server: grpc.Server) -> None:
        sangam_pb2_grpc.add_ColocatedWorkerServiceServicer_to_server(
            servicer,
            server,
        )
        if self.config.enable_kv_receive:
            sangam_pb2_grpc.add_DecodeWorkerServiceServicer_to_server(
                servicer,
                server,
            )


def serve_colocated_worker(config: ColocatedWorkerConfig) -> None:
    """Main entry point for a colocated worker process."""
    worker = ColocatedWorker(config)
    worker.serve()
