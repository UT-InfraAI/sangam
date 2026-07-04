"""Shared scheduler infrastructure for colocated and hybrid modes."""

from __future__ import annotations

import hashlib
import heapq
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from sangam.engine.grpc_server import (
    bind_insecure_port_or_raise,
    create_server,
    suppress_grpc_shutdown_thread_noise,
)
from sangam.engine.scheduler_config import BaseSchedulerConfig
from sangam.logger import init_logger
from sangam.metrics.constants import SchedulerQueueTimeSeries, WorkerStateTimeline
from sangam.metrics.metrics_store import MetricsStore
from sangam.process_lifecycle import install_termination_handler
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.request import Request
from sangam.sampling_parameters import SamplingParameters
from sangam.types import WorkerType

logger = init_logger(__name__)


def _fold_seed_bytes_to_int64(seed_bytes: bytes) -> int:
    return int.from_bytes(seed_bytes[:8], byteorder="big", signed=True)


def _stop_server_quietly(server) -> None:
    try:
        server.stop(grace=0)
    except KeyboardInterrupt:
        return
    except ValueError as exc:
        if "server must be started and not shutting down" not in str(exc):
            raise


def derive_request_seed_from_generate_request(
    request: sangam_pb2.GenerateRequest,
    block_length: int,
) -> int:
    seed_request = sangam_pb2.GenerateRequest(
        prompt_token_ids=request.prompt_token_ids,
        gen_length=request.gen_length,
    )
    if request.HasField("sampling_parameters"):
        seed_request.sampling_parameters.CopyFrom(request.sampling_parameters)
    digest = hashlib.sha256(
        seed_request.SerializeToString(deterministic=True)
        + block_length.to_bytes(4, byteorder="big", signed=True)
    ).digest()
    return _fold_seed_bytes_to_int64(digest)


class EventType(Enum):
    REQUEST_ARRIVED = auto()
    BATCH_METRICS = auto()
    KV_TRANSFER = auto()
    WORKER_REGISTER = auto()
    CONVERSION_RPC_FINISHED = auto()
    ENQUEUE_WORKER_DONE = auto()
    PREFILL_REDISTRIBUTION_DONE = auto()


@dataclass
class EnqueueWorkerResult:
    request_id: str
    block_index: int
    worker_id: str
    worker_type: str
    success: bool
    accepted_state: Any | None  # sangam_pb2.WorkerStateSnapshot or None
    error_message: str = ""


@dataclass
class PrefillRedistributionResult:
    source_worker_id: str
    returned_requests: list  # list[sangam_pb2.EnqueuePrefillRequest]


@dataclass
class SchedulerEvent:
    event_type: EventType
    payload: Any
    enqueue_time: float = field(default_factory=time.time)


@dataclass
class WorkerRegistrationRequest:
    worker_id: str
    worker_type: str
    address: str
    dist_rank: int
    gpu_id: int
    max_pages: int | None
    page_size: int | None
    is_conversion: bool
    reply_queue: queue.Queue[bool]


@dataclass
class WorkerInfo:
    worker_id: str
    worker_type: WorkerType
    address: str
    dist_rank: int
    gpu_id: int
    max_pages: int | None = None
    page_size: int | None = None
    free_pages: int | None = None
    outstanding_prefill_tokens: int = 0
    outstanding_requests: int = 0
    active_request_length_sum: int = 0
    draining: bool = False
    consecutive_slow_iterations: int = 0


class BaseScheduler(ABC):
    """Common scheduler infrastructure without lifecycle policy."""

    def __init__(
        self,
        config: BaseSchedulerConfig,
        *,
        event_thread_name: str,
    ) -> None:
        self._config = config
        self.block_length = config.block_length
        self.mask_id = config.mask_id
        self.max_gen_len = config.max_gen_len
        self._requests: dict[str, Request] = {}
        self._lock = threading.Lock()
        self._event_queue: queue.Queue[SchedulerEvent] = queue.Queue()
        self._pending_requests: dict[str, Request] = {}
        self._pending_request_heap: list[tuple[float, str]] = []
        self._outbound_executor = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="sched-outbound"
        )
        self._event_thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name=event_thread_name,
        )
        self._event_thread.start()
        self._metrics_store = MetricsStore.get_or_create_instance(
            output_dir=config.metrics_output_dir,
            enabled=config.enable_metrics,
            enable_individual_batch_metrics=config.enable_individual_batch_metrics,
        )
        self._export_partial_metrics = config.export_partial_metrics

    def submit(self, req: Request) -> str:
        with self._lock:
            self._requests[req.request_id] = req
        self._metrics_store.on_request_arrival(req.request_id, req.submit_time)
        self._enqueue_event(EventType.REQUEST_ARRIVED, req)
        return req.request_id

    def poll(self, request_id: str) -> Request | None:
        return self._requests.get(request_id)

    def plot_metrics(self) -> None:
        self._metrics_store.plot()

    def reset_metrics(self) -> None:
        self._metrics_store.reset()

    def _enqueue_event(self, event_type: EventType, payload: Any) -> None:
        self._event_queue.put(SchedulerEvent(event_type, payload))

    def _drain_pending_requests(self) -> None:
        if not self._is_ready_for_requests() or not self._pending_requests:
            return
        while self._pending_request_heap:
            _, request_id = heapq.heappop(self._pending_request_heap)
            req = self._pending_requests.pop(request_id, None)
            if req is None:
                continue
            self._enqueue_event(EventType.REQUEST_ARRIVED, req)
        self._record_pending_requests_depth(time.time())

    def _queue_pending_request(self, req: Request, *, now: float | None = None) -> None:
        if req.request_id in self._pending_requests:
            return
        self._pending_requests[req.request_id] = req
        heapq.heappush(
            self._pending_request_heap,
            (req.submit_time, req.request_id),
        )
        self._record_pending_requests_depth(time.time() if now is None else now)

    def _remove_pending_request(
        self, request_id: str, *, now: float | None = None
    ) -> Request | None:
        removed = self._pending_requests.pop(request_id, None)
        if removed is not None:
            self._record_pending_requests_depth(time.time() if now is None else now)
        return removed

    def _record_pending_requests_depth(self, now: float) -> None:
        self._metrics_store.on_scheduler_queue_depth(
            SchedulerQueueTimeSeries.PENDING_REQUESTS,
            len(self._pending_requests),
            now,
        )
        if self._metrics_store.enabled:
            pending_tokens = sum(
                req.request_accounting_tokens for req in self._pending_requests.values()
            )
            self._metrics_store.on_scheduler_pending_prefill_tokens(pending_tokens, now)

    def _mark_scheduler_wait(
        self,
        block: Any,
        now: float,
        phase: str,
    ) -> None:
        if block.scheduler_wait_start_time is None:
            block.scheduler_wait_start_time = now
            block.scheduler_wait_phase = phase

    def _clear_scheduler_wait(
        self,
        request_id: str,
        block: Any,
        now: float,
    ) -> None:
        if block.scheduler_wait_start_time is None:
            return
        wait_duration = max(0.0, now - block.scheduler_wait_start_time)
        block.scheduler_wait_duration += wait_duration
        if block.scheduler_wait_phase == "prefill":
            block.prefill_scheduler_wait_duration += wait_duration
        elif block.scheduler_wait_phase == "decode":
            block.decode_scheduler_wait_duration += wait_duration
        block.scheduler_wait_start_time = None
        block.scheduler_wait_phase = None

    def _process_worker_state_snapshot(
        self,
        snapshot: sangam_pb2.WorkerStateSnapshot,
        worker_id: str,
        worker_type: str,
    ) -> None:
        """Feed a piggybacked state snapshot to the metrics store."""
        state = WorkerStateTimeline(
            sangam_pb2.WorkerState.Name(snapshot.state)
            .removeprefix("WORKER_STATE_")
            .lower()
        )
        self._metrics_store.on_worker_state(
            worker_id=worker_id,
            worker_type=worker_type,
            state=state,
            timestamp=snapshot.timestamp,
            waiting_queue_depth=snapshot.waiting_queue_depth,
            active_batch_size=snapshot.active_batch_size,
            kv_total_pages=snapshot.kv_total_pages,
            kv_used_pages=snapshot.kv_used_pages,
            kv_free_pages=snapshot.kv_free_pages,
        )

    def _submit_enqueue_async(
        self, stub_fn, request, worker_id: str, worker_type: str
    ) -> None:
        """Submit an enqueue RPC to the outbound thread pool.

        The response's accepted_state is fed back to the event loop as an
        ENQUEUE_WORKER_DONE event so that all state updates remain on the
        single event-loop thread with monotonically ordered timestamps.
        """

        def _run() -> None:
            accepted_state = None
            success = False
            error_message = ""
            try:
                resp = stub_fn(request)
                success = bool(resp.success)
                if resp.HasField("accepted_state"):
                    accepted_state = resp.accepted_state
                if not success:
                    error_message = (
                        f"{worker_type} worker {worker_id} rejected enqueue for "
                        f"request {request.request_id} block {request.block_index}"
                    )
            except Exception:
                error_message = (
                    f"Async enqueue RPC failed for worker {worker_id} "
                    f"(request_id={request.request_id}, block_index={request.block_index})"
                )
                logger.warning(
                    "%s",
                    error_message,
                    exc_info=True,
                )
            self._enqueue_event(
                EventType.ENQUEUE_WORKER_DONE,
                EnqueueWorkerResult(
                    request_id=request.request_id,
                    block_index=request.block_index,
                    worker_id=worker_id,
                    worker_type=worker_type,
                    success=success,
                    accepted_state=accepted_state,
                    error_message=error_message,
                ),
            )

        self._outbound_executor.submit(_run)

    def _timed_grpc_call(self, stub_fn, request, rpc_name: str):
        """Compatibility wrapper for outbound scheduler RPC calls."""
        del rpc_name
        return stub_fn(request)

    def _drain_outbound(self) -> None:
        """Block until all pending async outbound calls have completed.

        Submits a no-op sentinel task and waits for it; because the executor
        processes tasks in submission order this guarantees all previously
        submitted tasks have finished. Intended for use in tests only.
        """
        self._outbound_executor.submit(lambda: None).result(timeout=5.0)

    def _on_enqueue_worker_done(self, result: EnqueueWorkerResult) -> None:
        if result.accepted_state is not None:
            self._process_worker_state_snapshot(
                result.accepted_state, result.worker_id, result.worker_type
            )
        self._handle_enqueue_worker_done(result)

    def _handle_enqueue_worker_done(self, result: EnqueueWorkerResult) -> None:
        return None

    def _event_loop(self) -> None:
        while True:
            event = self._event_queue.get()

            if event.event_type == EventType.WORKER_REGISTER:
                registration: WorkerRegistrationRequest = event.payload
                try:
                    success = self._register_worker(
                        registration.worker_id,
                        registration.worker_type,
                        registration.address,
                        registration.dist_rank,
                        registration.gpu_id,
                        registration.max_pages,
                        registration.page_size,
                        registration.is_conversion,
                    )
                except Exception:
                    logger.exception("Error handling event %s", event.event_type)
                    success = False
                registration.reply_queue.put(success)
                continue

            try:
                if event.event_type == EventType.REQUEST_ARRIVED:
                    self._on_request_arrived(event.payload)
                elif event.event_type == EventType.BATCH_METRICS:
                    self._on_batch_metrics(event.payload)
                elif event.event_type == EventType.KV_TRANSFER:
                    self._on_kv_transfer(event.payload)
                elif event.event_type == EventType.CONVERSION_RPC_FINISHED:
                    self._on_conversion_rpc_finished(event.payload)
                elif event.event_type == EventType.ENQUEUE_WORKER_DONE:
                    self._on_enqueue_worker_done(event.payload)
                elif event.event_type == EventType.PREFILL_REDISTRIBUTION_DONE:
                    self._on_prefill_redistribution_done(event.payload)
            except Exception:
                logger.exception("Error handling event %s", event.event_type)

    def register_worker(
        self,
        worker_id: str,
        worker_type: str,
        address: str,
        dist_rank: int,
        gpu_id: int,
        max_pages: int | None = None,
        page_size: int | None = None,
        is_conversion: bool = False,
    ) -> bool:
        if threading.current_thread() is self._event_thread:
            return self._register_worker(
                worker_id,
                worker_type,
                address,
                dist_rank,
                gpu_id,
                max_pages,
                page_size,
                is_conversion,
            )

        reply_queue: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._enqueue_event(
            EventType.WORKER_REGISTER,
            WorkerRegistrationRequest(
                worker_id=worker_id,
                worker_type=worker_type,
                address=address,
                dist_rank=dist_rank,
                gpu_id=gpu_id,
                max_pages=max_pages,
                page_size=page_size,
                is_conversion=is_conversion,
                reply_queue=reply_queue,
            ),
        )
        return reply_queue.get()

    @abstractmethod
    def _register_worker(
        self,
        worker_id: str,
        worker_type: str,
        address: str,
        dist_rank: int,
        gpu_id: int,
        max_pages: int | None = None,
        page_size: int | None = None,
        is_conversion: bool = False,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _is_ready_for_requests(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _on_request_arrived(self, req: Request) -> None:
        raise NotImplementedError

    @abstractmethod
    def _on_batch_metrics(self, report: sangam_pb2.BatchMetricsReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def _on_kv_transfer(self, report: sangam_pb2.KVTransferReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def _scheduler_status_counts(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def _on_conversion_rpc_finished(self, payload: Any) -> None:
        raise NotImplementedError

    def _on_prefill_redistribution_done(self, payload: Any) -> None:
        raise NotImplementedError


class BaseSchedulerServicer(
    sangam_pb2_grpc.SchedulerServiceServicer,
    sangam_pb2_grpc.SchedulerWorkerServiceServicer,
):
    """Implements both scheduler-facing gRPC services.

    Bound to two gRPC servers in :func:`serve_scheduler_instance`: a client
    server (Generate / Submit / Poll / admin) and a worker server (registration
    and batch / KV-transfer reports). Splitting onto separate ports prevents
    the client pool from starving worker callbacks when many in-flight Generate
    calls park threads on `req.done_event.wait`.
    """

    def __init__(self, scheduler: BaseScheduler):
        self._scheduler = scheduler

    def _build_and_submit(self, request: sangam_pb2.GenerateRequest) -> Request:
        request_seed = (
            request.request_seed
            if request.HasField("request_seed")
            else derive_request_seed_from_generate_request(
                request, self._scheduler.block_length
            )
        )
        block_length = self._scheduler.block_length
        max_gen_len = self._scheduler.max_gen_len
        requested_gen_length = request.gen_length
        requested_blocks = (requested_gen_length + block_length - 1) // block_length
        if max_gen_len is None:
            effective_gen_length = requested_gen_length
            target_blocks = requested_blocks
        else:
            effective_gen_length = max_gen_len
            max_blocks = max_gen_len // block_length
            if requested_blocks > max_blocks:
                logger.warning(
                    "Submit: requested gen_length=%s requires %s blocks but "
                    "--max-gen-len=%s only allows %s blocks; clamping target "
                    "for this request",
                    requested_gen_length,
                    requested_blocks,
                    max_gen_len,
                    max_blocks,
                )
                target_blocks = max_blocks
            else:
                target_blocks = max(requested_blocks, 1)
        req = Request(
            prompt_token_ids=list(request.prompt_token_ids),
            gen_length=effective_gen_length,
            block_length=block_length,
            sampling_parameters=SamplingParameters.from_proto(
                request.sampling_parameters
                if request.HasField("sampling_parameters")
                else None
            ),
            mask_id=self._scheduler.mask_id,
            request_seed=request_seed,
            target_blocks=target_blocks,
        )
        self._scheduler.submit(req)
        logger.debug(
            "Submitted request %s: gen_length=%s, block_length=%s, target_blocks=%s",
            req.request_id,
            req.gen_length,
            req.block_length,
            req.target_blocks,
        )
        return req

    def _terminal_response_kwargs(self, req: Request) -> dict[str, float]:
        kwargs: dict[str, float] = {"server_arrival_time": req.submit_time}
        first_block = req.block_states[0] if req.block_states else None
        if first_block is not None and first_block.prefill_start_time is not None:
            kwargs["server_start_time"] = first_block.prefill_start_time
        if req.complete_time is not None:
            kwargs["server_complete_time"] = req.complete_time
        return kwargs

    def Submit(
        self, request: sangam_pb2.GenerateRequest, context
    ) -> sangam_pb2.SubmitResponse:
        req = self._build_and_submit(request)
        return sangam_pb2.SubmitResponse(request_id=req.request_id)

    def Generate(
        self, request: sangam_pb2.GenerateRequest, context
    ) -> sangam_pb2.GenerateResponse:
        """Blocking unary RPC: submit and park on the request's done_event
        until it reaches a terminal status, then return the terminal payload.

        Replaces the Submit+Poll round-trip storm for clients (e.g. the
        benchmark backend) that would otherwise hammer Poll on a fixed
        interval and saturate the gRPC thread pool / GIL.
        """
        req = self._build_and_submit(request)
        # Loop on short waits so client cancellation / deadline expiry is
        # observed promptly without polling scheduler state.
        while context.is_active():
            remaining = context.time_remaining()
            wait_s = 1.0 if remaining == float("inf") else min(1.0, remaining)
            if wait_s <= 0:
                break
            if req.done_event.wait(timeout=wait_s):
                break
        if not req.done_event.is_set():
            # Client gave up or deadline expired. The scheduler will still
            # finish the request and set done_event; we just can't reply.
            return sangam_pb2.GenerateResponse()
        return sangam_pb2.GenerateResponse(
            request_id=req.request_id,
            status=req.status.value,
            output_token_ids=req.sequence_ids,
            num_forward_evals=req.num_forward_evals,
            error_message=req.error_message or "",
            **self._terminal_response_kwargs(req),
        )

    def Poll(self, request: sangam_pb2.PollRequest, context) -> sangam_pb2.PollResponse:
        req = self._scheduler.poll(request.request_id)
        if req is None:
            return sangam_pb2.PollResponse(status="NOT_FOUND")
        return sangam_pb2.PollResponse(
            status=req.status.value,
            output_token_ids=req.sequence_ids,
            num_forward_evals=req.num_forward_evals,
            error_message=req.error_message or "",
            **self._terminal_response_kwargs(req),
        )

    def RegisterWorker(
        self, request: sangam_pb2.RegisterWorkerRequest, context
    ) -> sangam_pb2.RegisterWorkerResponse:
        success = self._scheduler.register_worker(
            request.worker_id,
            request.worker_type,
            request.address,
            request.dist_rank,
            request.gpu_id,
            request.max_pages if request.HasField("max_pages") else None,
            request.page_size if request.HasField("page_size") else None,
            request.is_conversion,
        )
        return sangam_pb2.RegisterWorkerResponse(success=success)

    def ReportBatchMetrics(
        self, request: sangam_pb2.BatchMetricsReport, context
    ) -> sangam_pb2.BatchMetricsAck:
        self._scheduler._enqueue_event(EventType.BATCH_METRICS, request)
        return sangam_pb2.BatchMetricsAck(success=True)

    def ReportKVTransfer(
        self, request: sangam_pb2.KVTransferReport, context
    ) -> sangam_pb2.KVTransferAck:
        self._scheduler._enqueue_event(EventType.KV_TRANSFER, request)
        return sangam_pb2.KVTransferAck(success=True)

    def ExportMetrics(
        self, request: sangam_pb2.ExportMetricsRequest, context
    ) -> sangam_pb2.ExportMetricsResponse:
        self._scheduler.plot_metrics()
        return sangam_pb2.ExportMetricsResponse(
            success=True,
            output_dir=self._scheduler._metrics_store.output_dir,
        )

    def ResetMetrics(
        self, request: sangam_pb2.ResetMetricsRequest, context
    ) -> sangam_pb2.ResetMetricsResponse:
        self._scheduler.reset_metrics()
        return sangam_pb2.ResetMetricsResponse(success=True)

    def GetSchedulerStatus(
        self, request: sangam_pb2.GetSchedulerStatusRequest, context
    ) -> sangam_pb2.GetSchedulerStatusResponse:
        num_prefill, num_decode, num_colocated = (
            self._scheduler._scheduler_status_counts()
        )
        return sangam_pb2.GetSchedulerStatusResponse(
            num_prefill_workers=num_prefill,
            num_decode_workers=num_decode,
            num_colocated_workers=num_colocated,
            ready_for_requests=self._scheduler._is_ready_for_requests(),
        )


def worker_callback_port(scheduler_port: int) -> int:
    """Worker → scheduler RPCs land on this port (scheduler_port + 1)."""
    return scheduler_port + 1


def serve_scheduler_instance(
    *,
    scheduler: BaseScheduler,
    servicer_cls: type[BaseSchedulerServicer],
    port: int,
    service_name: str,
    startup_log_name: str,
    logger_instance: Any,
) -> None:
    install_termination_handler()
    suppress_grpc_shutdown_thread_noise()
    servicer = servicer_cls(scheduler)
    # Two servers on separate ports so worker callbacks (RegisterWorker /
    # ReportBatchMetrics / ReportKVTransfer) cannot be starved by long-lived
    # Generate calls that park one pool thread each on `req.done_event.wait`.
    # The client pool is generously sized for high benchmark concurrency; the
    # worker pool stays small because traffic is bounded by `num_workers ×
    # few-concurrent-RPCs`.
    max_message_length = scheduler._config.max_grpc_message_length
    client_server = create_server(
        max_workers=4096, max_message_length=max_message_length
    )
    sangam_pb2_grpc.add_SchedulerServiceServicer_to_server(servicer, client_server)
    bind_insecure_port_or_raise(
        server=client_server,
        port=port,
        service_name=f"{service_name} (client)",
    )

    worker_server = create_server(max_workers=64, max_message_length=max_message_length)
    sangam_pb2_grpc.add_SchedulerWorkerServiceServicer_to_server(
        servicer, worker_server
    )
    worker_port = worker_callback_port(port)
    bind_insecure_port_or_raise(
        server=worker_server,
        port=worker_port,
        service_name=f"{service_name} (worker)",
    )

    client_server.start()
    worker_server.start()
    logger_instance.info(
        "%s serving on ports %s (client) and %s (worker)",
        startup_log_name,
        port,
        worker_port,
    )
    try:
        client_server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_server_quietly(client_server)
        _stop_server_quietly(worker_server)
        if scheduler._export_partial_metrics:
            scheduler.plot_metrics()
        else:
            logger_instance.debug(
                "Skipping metrics export on termination "
                "(pass --export-partial-metrics to enable)."
            )
