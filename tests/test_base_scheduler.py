"""Tests for shared scheduler infrastructure in base_scheduler.py."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from sangam.engine.base_scheduler import (
    BaseScheduler,
    BaseSchedulerServicer,
    EventType,
    WorkerInfo,
    derive_request_seed_from_generate_request,
    serve_scheduler_instance,
)
from sangam.engine.scheduler_config import BaseSchedulerConfig
from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.metrics.metrics_store import MetricsStore
from sangam.proto import sangam_pb2
from sangam.request import Request, RequestStatus
from sangam.types import WorkerType


def _active_context() -> MagicMock:
    """A gRPC ServicerContext stand-in that reports an active call with no
    deadline. Generate uses ``is_active`` and ``time_remaining`` to bound its
    wait."""
    ctx = MagicMock()
    ctx.is_active.return_value = True
    ctx.time_remaining.return_value = float("inf")
    return ctx


def _wait_for_submitted_request(
    scheduler: BaseScheduler, timeout: float = 2.0
) -> Request:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with scheduler._lock:
            if scheduler._requests:
                return next(iter(scheduler._requests.values()))
        time.sleep(0.01)
    raise AssertionError("request never registered with scheduler")


@pytest.fixture(autouse=True)
def reset_metrics_store_singleton():
    MetricsStore._instance = None
    yield
    MetricsStore._instance = None


class FakeScheduler(BaseScheduler):
    def __init__(self, *, block_length: int = 4, max_gen_len: int | None = None):
        self._workers: list[WorkerInfo] = []
        self.events: list[tuple[str, object]] = []
        super().__init__(
            BaseSchedulerConfig(
                metrics_output_dir="test_output",
                enable_metrics=False,
                enable_individual_batch_metrics=False,
                export_partial_metrics=False,
                block_length=block_length,
                mask_id=126336,
                max_gen_len=max_gen_len,
                max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
            ),
            event_thread_name="fake-scheduler-loop",
        )

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
        assert is_conversion is False
        self._workers.append(
            WorkerInfo(
                worker_id=worker_id,
                worker_type=WorkerType(worker_type),
                address=address,
                dist_rank=dist_rank,
                gpu_id=gpu_id,
                max_pages=max_pages,
                page_size=page_size,
                free_pages=max_pages,
            )
        )
        self._drain_pending_requests()
        return True

    def _is_ready_for_requests(self) -> bool:
        return bool(self._workers)

    def _on_request_arrived(self, req) -> None:
        self.events.append(("request", req))

    def _on_batch_metrics(self, report) -> None:
        self.events.append(("batch", report))

    def _on_kv_transfer(self, report) -> None:
        self.events.append(("kv_transfer", report))

    def _scheduler_status_counts(self) -> tuple[int, int, int]:
        return 1, 2, len(self._workers)

    def _on_conversion_rpc_finished(self, payload) -> None:
        self.events.append(("conversion_finished", payload))


def test_submit_and_poll_store_request():
    scheduler = FakeScheduler()
    req = sangam_pb2.GenerateRequest(prompt_token_ids=[1], gen_length=4)
    servicer = BaseSchedulerServicer(scheduler)

    response = servicer.Submit(req, context=None)

    stored = scheduler.poll(response.request_id)
    assert stored is not None
    assert stored.request_id == response.request_id
    assert stored.block_length == scheduler.block_length
    assert stored.request_seed == derive_request_seed_from_generate_request(
        req, scheduler.block_length
    )


def test_submit_preserves_explicit_request_seed():
    scheduler = FakeScheduler()
    req = sangam_pb2.GenerateRequest(
        prompt_token_ids=[1],
        gen_length=4,
        request_seed=-123,
    )
    servicer = BaseSchedulerServicer(scheduler)

    response = servicer.Submit(req, context=None)

    stored = scheduler.poll(response.request_id)
    assert stored is not None
    assert stored.request_seed == -123


def test_submit_fallback_seed_is_content_stable():
    scheduler = FakeScheduler()
    servicer = BaseSchedulerServicer(scheduler)
    req_a = sangam_pb2.GenerateRequest(
        prompt_token_ids=[1, 2, 3],
        gen_length=4,
    )
    req_b = sangam_pb2.GenerateRequest(
        prompt_token_ids=[1, 2, 3],
        gen_length=4,
    )

    resp_a = servicer.Submit(req_a, context=None)
    resp_b = servicer.Submit(req_b, context=None)

    stored_a = scheduler.poll(resp_a.request_id)
    stored_b = scheduler.poll(resp_b.request_id)
    assert stored_a is not None
    assert stored_b is not None
    assert stored_a.request_seed == stored_b.request_seed


def test_submit_without_max_gen_len_uses_request_gen_length():
    scheduler = FakeScheduler(block_length=32)
    servicer = BaseSchedulerServicer(scheduler)
    req = sangam_pb2.GenerateRequest(prompt_token_ids=[1], gen_length=64)

    response = servicer.Submit(req, context=None)

    stored = scheduler.poll(response.request_id)
    assert stored is not None
    assert stored.gen_length == 64
    assert stored.target_blocks == 2
    assert len(stored.sequence_ids) == 1 + 64


def test_submit_with_max_gen_len_resizes_buffer_and_sets_target_blocks():
    scheduler = FakeScheduler(block_length=32, max_gen_len=256)
    servicer = BaseSchedulerServicer(scheduler)
    req = sangam_pb2.GenerateRequest(prompt_token_ids=[1], gen_length=64)

    response = servicer.Submit(req, context=None)

    stored = scheduler.poll(response.request_id)
    assert stored is not None
    assert stored.gen_length == 256
    assert len(stored.sequence_ids) == 1 + 256
    assert stored.target_blocks == 2
    assert stored.target_gen_tokens == 64
    # Accounting reflects the full buffer (prompt + max_gen_len), not the
    # early-exit target_gen_tokens, so scheduler policies charge for the
    # buffer the worker actually holds.
    assert stored.request_accounting_tokens == 1 + 256


def test_submit_with_max_gen_len_clamps_oversized_request(monkeypatch):
    scheduler = FakeScheduler(block_length=32, max_gen_len=64)
    servicer = BaseSchedulerServicer(scheduler)
    req = sangam_pb2.GenerateRequest(prompt_token_ids=[1], gen_length=200)

    warnings: list[str] = []
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.logger.warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
    )

    response = servicer.Submit(req, context=None)

    stored = scheduler.poll(response.request_id)
    assert stored is not None
    assert stored.gen_length == 64
    assert stored.target_blocks == 2
    clamp_warnings = [w for w in warnings if "clamping target" in w]
    assert len(clamp_warnings) == 1


def test_base_scheduler_rejects_max_gen_len_not_divisible_by_block():
    with pytest.raises(ValueError, match="must be divisible by"):
        FakeScheduler(block_length=32, max_gen_len=70)


def test_generate_blocks_until_done_event_then_returns_terminal_payload():
    scheduler = FakeScheduler()
    servicer = BaseSchedulerServicer(scheduler)
    req_proto = sangam_pb2.GenerateRequest(prompt_token_ids=[1, 2], gen_length=4)

    response_holder: list[sangam_pb2.GenerateResponse] = []

    def _call_generate() -> None:
        response_holder.append(servicer.Generate(req_proto, context=_active_context()))

    caller = threading.Thread(target=_call_generate)
    caller.start()
    try:
        stored = _wait_for_submitted_request(scheduler)
        # Initially the Generate handler is parked on the done_event.
        assert not stored.done_event.is_set()
        stored.sequence_ids = [1, 2, 7, 8, 9, 10]
        stored.num_forward_evals = 3
        stored.status = RequestStatus.COMPLETED
        stored.complete_time = time.time()
        stored.done_event.set()
        caller.join(timeout=2.0)
        assert not caller.is_alive(), "Generate did not unblock after done_event.set()"
    finally:
        if caller.is_alive():
            # Last-ditch cleanup so a failing test doesn't leak a thread.
            stored.done_event.set()
            caller.join(timeout=1.0)

    assert len(response_holder) == 1
    resp = response_holder[0]
    assert resp.status == "COMPLETED"
    assert list(resp.output_token_ids) == [1, 2, 7, 8, 9, 10]
    assert resp.num_forward_evals == 3
    assert resp.HasField("server_arrival_time")
    assert resp.HasField("server_complete_time")


def test_generate_returns_error_status_when_request_fails():
    scheduler = FakeScheduler()
    servicer = BaseSchedulerServicer(scheduler)
    req_proto = sangam_pb2.GenerateRequest(prompt_token_ids=[1, 2], gen_length=4)

    response_holder: list[sangam_pb2.GenerateResponse] = []

    def _call_generate() -> None:
        response_holder.append(servicer.Generate(req_proto, context=_active_context()))

    caller = threading.Thread(target=_call_generate)
    caller.start()
    try:
        stored = _wait_for_submitted_request(scheduler)
        stored.status = RequestStatus.ERROR
        stored.error_message = "boom"
        stored.done_event.set()
        caller.join(timeout=2.0)
        assert not caller.is_alive()
    finally:
        if caller.is_alive():
            stored.done_event.set()
            caller.join(timeout=1.0)

    resp = response_holder[0]
    assert resp.status == "ERROR"
    assert resp.error_message == "boom"


def test_generate_returns_empty_response_when_client_cancels():
    scheduler = FakeScheduler()
    servicer = BaseSchedulerServicer(scheduler)
    req_proto = sangam_pb2.GenerateRequest(prompt_token_ids=[1, 2], gen_length=4)

    # Context starts active, then flips to inactive on the next is_active() call.
    # is_active() is invoked once per wait-loop iteration; flipping after the
    # first iteration mimics a cancelled client.
    ctx = MagicMock()
    ctx.is_active.side_effect = [True, False]
    ctx.time_remaining.return_value = 0.05

    resp = servicer.Generate(req_proto, context=ctx)

    assert resp.status == ""
    assert list(resp.output_token_ids) == []


def test_drain_pending_requests_requeues_when_ready():
    scheduler = FakeScheduler()
    scheduler._event_queue.put = MagicMock()
    req = MagicMock()
    req.request_id = "req-1"
    req.submit_time = 1.0
    scheduler._queue_pending_request(req)

    scheduler._register_worker("w-0", "prefill", "addr:0", dist_rank=0, gpu_id=0)

    scheduler._event_queue.put.assert_called_once()
    assert req.request_id not in scheduler._pending_requests


def test_report_rpcs_enqueue_scheduler_events():
    scheduler = FakeScheduler()
    scheduler._event_queue.put = MagicMock()
    servicer = BaseSchedulerServicer(scheduler)

    servicer.ReportBatchMetrics(sangam_pb2.BatchMetricsReport(), context=None)
    servicer.ReportKVTransfer(sangam_pb2.KVTransferReport(), context=None)

    event_types = [
        call.args[0].event_type for call in scheduler._event_queue.put.call_args_list
    ]
    assert event_types == [
        EventType.BATCH_METRICS,
        EventType.KV_TRANSFER,
    ]


def test_get_scheduler_status_uses_scheduler_counts():
    scheduler = FakeScheduler()
    servicer = BaseSchedulerServicer(scheduler)

    scheduler.register_worker("w-0", "prefill", "addr:0", dist_rank=0, gpu_id=0)
    response = servicer.GetSchedulerStatus(
        sangam_pb2.GetSchedulerStatusRequest(),
        context=None,
    )

    assert response.num_prefill_workers == 1
    assert response.num_decode_workers == 2
    assert response.num_colocated_workers == 1
    assert response.ready_for_requests is True


def test_serve_scheduler_instance_exports_metrics_on_keyboard_interrupt(
    monkeypatch,
):
    scheduler = FakeScheduler()
    scheduler._export_partial_metrics = True
    scheduler.plot_metrics = MagicMock()
    events: list[tuple[str, object]] = []

    class _FakeServer:
        def start(self) -> None:
            events.append(("server.start", None))

        def wait_for_termination(self) -> None:
            raise KeyboardInterrupt

        def stop(self, grace: int) -> None:
            events.append(("server.stop", grace))

    monkeypatch.setattr(
        "sangam.engine.base_scheduler.create_server",
        lambda max_workers, max_message_length: _FakeServer(),
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.sangam_pb2_grpc."
        "add_SchedulerServiceServicer_to_server",
        lambda servicer, server: events.append(("add_client_servicer", servicer)),
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.sangam_pb2_grpc."
        "add_SchedulerWorkerServiceServicer_to_server",
        lambda servicer, server: events.append(("add_worker_servicer", servicer)),
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.bind_insecure_port_or_raise",
        lambda server, port, service_name: events.append(
            ("bind", (port, service_name))
        ),
    )

    serve_scheduler_instance(
        scheduler=scheduler,
        servicer_cls=BaseSchedulerServicer,
        port=50051,
        service_name="scheduler",
        startup_log_name="scheduler",
        logger_instance=MagicMock(),
    )

    # Two servers (client on port, worker on port+1) wired up before start;
    # KeyboardInterrupt during the client-server wait stops both servers.
    assert [event_name for event_name, _ in events] == [
        "add_client_servicer",
        "bind",
        "add_worker_servicer",
        "bind",
        "server.start",
        "server.start",
        "server.stop",
        "server.stop",
    ]
    bind_targets = [payload for name, payload in events if name == "bind"]
    assert bind_targets == [
        (50051, "scheduler (client)"),
        (50052, "scheduler (worker)"),
    ]
    scheduler.plot_metrics.assert_called_once_with()


def test_serve_scheduler_instance_ignores_server_stop_during_grpc_shutdown(
    monkeypatch,
) -> None:
    scheduler = FakeScheduler()
    scheduler._export_partial_metrics = True
    scheduler.plot_metrics = MagicMock()

    class _FakeServer:
        def start(self) -> None:
            pass

        def wait_for_termination(self) -> None:
            raise KeyboardInterrupt

        def stop(self, grace: int) -> None:
            raise ValueError("server must be started and not shutting down")

    monkeypatch.setattr(
        "sangam.engine.base_scheduler.create_server",
        lambda max_workers, max_message_length: _FakeServer(),
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.sangam_pb2_grpc."
        "add_SchedulerServiceServicer_to_server",
        lambda servicer, server: None,
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.sangam_pb2_grpc."
        "add_SchedulerWorkerServiceServicer_to_server",
        lambda servicer, server: None,
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.bind_insecure_port_or_raise",
        lambda server, port, service_name: None,
    )

    serve_scheduler_instance(
        scheduler=scheduler,
        servicer_cls=BaseSchedulerServicer,
        port=50051,
        service_name="scheduler",
        startup_log_name="scheduler",
        logger_instance=MagicMock(),
    )

    scheduler.plot_metrics.assert_called_once_with()


def test_serve_scheduler_instance_ignores_keyboard_interrupt_during_server_stop(
    monkeypatch,
) -> None:
    scheduler = FakeScheduler()
    scheduler._export_partial_metrics = True
    scheduler.plot_metrics = MagicMock()

    class _FakeServer:
        def start(self) -> None:
            pass

        def wait_for_termination(self) -> None:
            raise KeyboardInterrupt

        def stop(self, grace: int) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "sangam.engine.base_scheduler.create_server",
        lambda max_workers, max_message_length: _FakeServer(),
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.sangam_pb2_grpc."
        "add_SchedulerServiceServicer_to_server",
        lambda servicer, server: None,
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.sangam_pb2_grpc."
        "add_SchedulerWorkerServiceServicer_to_server",
        lambda servicer, server: None,
    )
    monkeypatch.setattr(
        "sangam.engine.base_scheduler.bind_insecure_port_or_raise",
        lambda server, port, service_name: None,
    )

    serve_scheduler_instance(
        scheduler=scheduler,
        servicer_cls=BaseSchedulerServicer,
        port=50051,
        service_name="scheduler",
        startup_log_name="scheduler",
        logger_instance=MagicMock(),
    )

    scheduler.plot_metrics.assert_called_once_with()
