import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.metrics.constants import WorkerStateTimeline
from sangam.kv_cache.paged_kv_cache import RequestKVState
from sangam.proto import sangam_pb2
from sangam.sampling_parameters import SamplingParameters
from sangam.types import PrefillQueuePolicy
from sangam.worker.prefill_queue_policy import compute_prefill_queue_priority
from sangam.worker.prefill_worker import (
    PrefillBatchItem,
    PrefillResult,
    PrefillWorker,
    PrefillWorkerServicer,
    RetainedPrefillState,
    StreamingKVTransferContext,
)
from sangam.worker.worker_config import PrefillWorkerConfig


def _make_prefill_worker_config(**overrides) -> PrefillWorkerConfig:
    defaults = dict(
        worker_id="pw-0",
        gpu_id=0,
        dist_rank=0,
        world_size=4,
        port=50100,
        model_name="dummy",
        scheduler_address="localhost:1",
        master_addr="localhost",
        master_port=29500,
        enable_metrics=True,
        enable_operation_metrics=False,
        op_metrics_layer_id=None,
        kv_page_size=32,
        kv_max_pages=1234,
        kv_dtype=torch.bfloat16,
        max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
        poll_interval=0.1,
        max_prefill_tokens_per_batch=4096,
        prefill_queue_policy="arrival_order",
        kv_transfer_timeout_s=30.0,
        streaming_layer_ready_timeout_s=30.0,
        streaming_recv_join_timeout_s=30.0,
    )
    defaults.update(overrides)
    return PrefillWorkerConfig(**defaults)


def _noop_overhead_tracker() -> SimpleNamespace:
    return SimpleNamespace(
        time_block=lambda *args, **kwargs: nullcontext(),
        drain=lambda: [],
    )


def _make_immediate_send_queue(
    on_submit=None,
) -> SimpleNamespace:
    """Build a mock send queue whose ``submit_async`` returns an already-done
    GpuWorkItem stand-in, matching the new prefill worker contract."""

    def _submit_async(fn, *args, **kwargs):
        if on_submit is not None:
            on_submit(fn, *args, **kwargs)
        event = threading.Event()
        event.set()
        return SimpleNamespace(done_event=event, result=None, exception=None)

    return SimpleNamespace(submit_async=_submit_async)


def test_enqueue_prefill_rejects_when_draining() -> None:
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._draining = __import__("threading").Event()
    servicer._draining.set()
    servicer._prefill_queue_policy = PrefillQueuePolicy.ARRIVAL_ORDER
    servicer._request_queue = SimpleNamespace(put=lambda item: None, qsize=lambda: 0)
    servicer._active_batch_size = 0

    response = servicer.EnqueuePrefill(
        SimpleNamespace(arrival_time=1.0, request_id="r1"),
        context=None,
    )

    assert response.success is False


def test_enqueue_prefill_returns_accepted_state_snapshot() -> None:
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._draining = __import__("threading").Event()
    servicer._prefill_queue_policy = PrefillQueuePolicy.ARRIVAL_ORDER
    servicer._request_queue = SimpleNamespace(
        put=lambda item: None,
        qsize=lambda: 1,
    )
    servicer._active_batch_size = 0
    servicer._current_kv_page_stats = lambda: (64, 16, 48)

    import time as _time

    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_QUEUED,
        timestamp=_time.time(),
        waiting_queue_depth=1,
        active_batch_size=0,
        kv_total_pages=64,
        kv_used_pages=16,
        kv_free_pages=48,
    )

    response = servicer.EnqueuePrefill(
        SimpleNamespace(arrival_time=1.0, request_id="r1"),
        context=None,
    )

    assert response.success is True
    assert response.HasField("accepted_state")
    assert response.accepted_state.state == sangam_pb2.WORKER_STATE_QUEUED


def test_prefill_queue_priority_arrival_order_uses_fcfs() -> None:
    req = SimpleNamespace(
        request_id="req-2",
        arrival_time=5.0,
        block_index=2,
        total_generation_blocks=4,
    )

    priority = compute_prefill_queue_priority(req, PrefillQueuePolicy.ARRIVAL_ORDER)

    assert priority == (5.0, "req-2")


def test_prefill_queue_priority_fewest_remaining_blocks_prioritizes_last_block() -> (
    None
):
    earlier = SimpleNamespace(
        request_id="req-early",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=4,
    )
    later = SimpleNamespace(
        request_id="req-late",
        arrival_time=2.0,
        block_index=3,
        total_generation_blocks=4,
    )

    earlier_priority = compute_prefill_queue_priority(
        earlier, PrefillQueuePolicy.FEWEST_REMAINING_BLOCKS
    )
    later_priority = compute_prefill_queue_priority(
        later, PrefillQueuePolicy.FEWEST_REMAINING_BLOCKS
    )

    assert later_priority < earlier_priority


def test_trigger_kv_transfer_rejects_unknown_request() -> None:
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._retained_lock = __import__("threading").Lock()
    servicer._retained_requests = {}

    response = servicer.TriggerKVTransfer(
        SimpleNamespace(request_id="req-1", block_index=0),
        context=None,
    )

    assert response.success is False


def test_trigger_kv_transfer_spawns_background_thread() -> None:
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._retained_lock = threading.Lock()
    servicer._retained_requests = {
        "req-1": RetainedPrefillState(
            request_id="req-1",
            block_index=0,
            request_seed=123,
            sequence=[1, 2, 3],
            kv_state=RequestKVState(page_ids=[1], seq_len=3, last_page_len=3),
            cuda_event=None,
        )
    }
    # Patch _coordinate_kv_transfer to a no-op so the thread exits immediately
    servicer._coordinate_kv_transfer = lambda req: None

    request = SimpleNamespace(
        request_id="req-1",
        block_index=0,
        decode_worker_address="localhost:20101",
        decode_dst_rank=2,
    )
    response = servicer.TriggerKVTransfer(request, context=None)

    assert response.success is True


def test_prefill_shutdown_tolerates_missing_executor() -> None:
    calls: list[str] = []
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._send_queues = {
        2: SimpleNamespace(shutdown=lambda: calls.append("send_queue_2"))
    }
    servicer._gpu_queue = SimpleNamespace(shutdown=lambda: calls.append("gpu"))

    servicer.shutdown()

    assert calls == ["send_queue_2", "gpu"]


def test_shutdown_stops_processing_without_executor() -> None:
    events: list[str] = []

    class _FakeThread:
        def join(self) -> None:
            events.append("process_thread.join")

    class _FakeQueue:
        def __init__(self, name: str) -> None:
            self._name = name

        def shutdown(self) -> None:
            events.append(f"{self._name}.shutdown")

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._draining = threading.Event()
    servicer._process_thread = _FakeThread()
    servicer._send_queues = {
        2: _FakeQueue("send_queue_2"),
        5: _FakeQueue("send_queue_5"),
    }
    servicer._gpu_queue = _FakeQueue("gpu_queue")

    servicer.shutdown()

    assert servicer._draining.is_set()
    assert events == [
        "process_thread.join",
        "send_queue_2.shutdown",
        "send_queue_5.shutdown",
        "gpu_queue.shutdown",
    ]


def test_batched_prefill_passes_logits_device_to_duration_timer(monkeypatch) -> None:
    timer_calls = []

    class _FakeDurationTimer:
        def __init__(self, name: str, use_cuda: bool, device=None) -> None:
            timer_calls.append((name, use_cuda, device))
            self.elapsed_s = 0.123

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyModel:
        num_layers = 1
        num_kv_heads = 1
        head_dim = 1
        num_q_heads = 1

        def __init__(self) -> None:
            self.model = SimpleNamespace(
                transformer=SimpleNamespace(
                    blocks=[SimpleNamespace(_paged_attn_state=None)]
                )
            )

        def __call__(self, input_ids):
            total_tokens = input_ids.shape[1]
            return SimpleNamespace(
                logits=torch.zeros((1, total_tokens, 8), dtype=torch.float32)
            )

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.pack_mixed_batch",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "sangam.worker.prefill_worker.run_mixed_paged_forward",
        lambda model, packed_batch: SimpleNamespace(
            packed_logits=torch.zeros((1, 4, 8), dtype=torch.float32),
            item_logits=[torch.zeros((1, 4, 8), dtype=torch.float32)],
        ),
    )
    monkeypatch.setattr(
        "sangam.worker.prefill_worker.DurationTimer",
        _FakeDurationTimer,
    )

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.device = torch.device("cpu")
    servicer.model = _DummyModel()
    servicer._flashinfer_workspace = object()
    servicer._prefill_kv_pool = SimpleNamespace(
        allocate=lambda seq_len: ([1], seq_len),
        free=lambda page_ids: None,
        kv_data=[],
    )
    servicer._sampler = SimpleNamespace(
        sample_batch=lambda reqs, block_tokens, logits: (
            block_tokens.clone(),
            torch.ones(len(reqs), dtype=torch.long, device=block_tokens.device),
        )
    )

    batch_items = [
        PrefillBatchItem(
            req_id="req-1",
            sequence=[1, 2, 3, 4],
            block_start=1,
            block_end=3,
            request_seed=123,
            sampling_parameters=SamplingParameters(),
            mask_id=7,
        )
    ]

    _, sampling_duration, _ = servicer._do_run_batched_prefill(batch_items)

    assert sampling_duration == pytest.approx(0.123)
    assert timer_calls == [("worker_sampling_prefill", False, torch.device("cpu"))]


def test_batched_prefill_raises_on_mixed_block_lengths() -> None:
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="Mixed prefill block lengths"):
        servicer._do_run_batched_prefill(
            [
                PrefillBatchItem(
                    req_id="req-1",
                    sequence=[1, 2, 3, 4],
                    block_start=0,
                    block_end=2,
                    request_seed=1,
                    sampling_parameters=SamplingParameters(),
                    mask_id=7,
                ),
                PrefillBatchItem(
                    req_id="req-2",
                    sequence=[1, 2, 3, 4],
                    block_start=0,
                    block_end=3,
                    request_seed=2,
                    sampling_parameters=SamplingParameters(),
                    mask_id=7,
                ),
            ]
        )


def test_processing_loop_reports_prefill_completion_and_retains_kv(monkeypatch) -> None:
    batch_reports: list[object] = []

    class _SingleItemQueue:
        def __init__(self, item) -> None:
            self._item = item
            self._calls = 0

        def get(self, timeout=None):
            if self._calls == 0:
                self._calls += 1
                return self._item
            raise KeyboardInterrupt

        def get_nowait(self):
            import queue

            raise queue.Empty

        def qsize(self) -> int:
            return 0

    enqueue_req = SimpleNamespace(
        arrival_time=1.0,
        request_id="req-1",
        block_index=0,
        sequence_ids=[1, 2, 3, 4],
        block_start=0,
        block_end=4,
        request_seed=123,
        temperature=0.0,
        unmasking_strategy="random",
        mask_id=7,
        prefill_enqueue_time=10.0,
        HasField=lambda name: False,
    )

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.worker_id = "pw-0"
    servicer._draining = __import__("threading").Event()
    servicer._request_queue = _SingleItemQueue((1.0, "req-1", enqueue_req))
    servicer._active_batch_size = 0
    servicer._max_prefill_tokens_per_batch = 4096
    servicer._kv_page_size = 16
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: (
            [
                PrefillResult(
                    sampled_sequence=[11, 7, 13, 14],
                    num_unmasked_tokens=1,
                    kv_state=RequestKVState(page_ids=[1], seq_len=4, last_page_len=4),
                    cuda_event=None,
                )
            ],
            0.25,
            None,
        )
    )
    servicer._retained_lock = __import__("threading").Lock()
    servicer._retained_requests = {}
    servicer._current_kv_page_stats = lambda: (64, 16, 48)
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: batch_reports.append(report)
    )
    servicer._overhead_tracker = _noop_overhead_tracker()
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE,
        timestamp=0.0,
        kv_total_pages=64,
        kv_used_pages=16,
        kv_free_pages=48,
    )

    timeline = iter([11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time", lambda: next(timeline)
    )

    with pytest.raises(KeyboardInterrupt):
        servicer._processing_loop()

    assert len(batch_reports) == 1
    report = batch_reports[0]
    assert report.batch_size == 1
    assert report.kv_total_pages == 64
    assert report.kv_used_pages == 16
    assert report.kv_free_pages == 48
    update = report.request_updates[0]
    assert update.success is True
    assert update.request_id == "req-1"
    assert list(update.updated_sequence) == [11, 7, 13, 14]
    assert update.prefill_duration == pytest.approx(1.0)
    assert update.prefill_queue_wait_duration == pytest.approx(1.0)
    assert report.sampling_duration == pytest.approx(0.25)
    assert "req-1" in servicer._retained_requests
    assert report.HasField("worker_state_after")


def test_processing_loop_marks_prefill_completed_blocks_and_frees_kv(
    monkeypatch,
) -> None:
    batch_reports: list[object] = []
    freed_page_ids: list[list[int]] = []

    class _SingleItemQueue:
        def __init__(self, item) -> None:
            self._item = item
            self._calls = 0

        def get(self, timeout=None):
            if self._calls == 0:
                self._calls += 1
                return self._item
            raise KeyboardInterrupt

        def get_nowait(self):
            import queue

            raise queue.Empty

        def qsize(self) -> int:
            return 0

    mask_id = 7
    enqueue_req = SimpleNamespace(
        arrival_time=1.0,
        request_id="req-1",
        block_index=0,
        sequence_ids=[1, mask_id, mask_id, 4],
        block_start=1,
        block_end=3,
        request_seed=123,
        mask_id=mask_id,
        prefill_enqueue_time=10.0,
        HasField=lambda name: False,
    )

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.worker_id = "pw-0"
    servicer._draining = __import__("threading").Event()
    servicer._request_queue = _SingleItemQueue((1.0, "req-1", enqueue_req))
    servicer._active_batch_size = 0
    servicer._max_prefill_tokens_per_batch = 4096
    servicer._kv_page_size = 16
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: (
            [
                PrefillResult(
                    sampled_sequence=[1, 9, 8, 4],
                    num_unmasked_tokens=2,
                    kv_state=RequestKVState(
                        page_ids=[1, 2], seq_len=4, last_page_len=4
                    ),
                    cuda_event=None,
                )
            ],
            0.25,
            None,
        )
    )
    servicer._prefill_kv_pool = SimpleNamespace(
        free=lambda page_ids: freed_page_ids.append(page_ids)
    )
    servicer._retained_lock = __import__("threading").Lock()
    servicer._retained_requests = {}
    servicer._current_kv_page_stats = lambda: (64, 16, 48)
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: batch_reports.append(report)
    )
    servicer._overhead_tracker = _noop_overhead_tracker()
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE,
        timestamp=0.0,
        kv_total_pages=64,
        kv_used_pages=16,
        kv_free_pages=48,
    )

    timeline = iter([11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time", lambda: next(timeline)
    )

    with pytest.raises(KeyboardInterrupt):
        servicer._processing_loop()

    assert len(batch_reports) == 1
    update = batch_reports[0].request_updates[0]
    assert update.block_completed is True
    assert servicer._retained_requests == {}
    assert freed_page_ids == [[1, 2]]


def test_coordinate_kv_transfer_reports_success_and_frees_retained_state(
    monkeypatch,
) -> None:
    transfer_reports: list[object] = []
    freed: list[str] = []

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 1
    servicer.model = SimpleNamespace(num_layers=2, num_kv_heads=4, head_dim=8)
    servicer._send_queues = {2: _make_immediate_send_queue()}
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._get_decode_stub = lambda address: SimpleNamespace(
        ReceiveKVCache=lambda request, timeout=None: SimpleNamespace(success=True)
    )
    servicer._free_retained_request = lambda request_id: freed.append(request_id)
    servicer._scheduler_stub = SimpleNamespace(
        ReportKVTransfer=lambda report: transfer_reports.append(report)
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: WorkerStateTimeline.IDLE
    servicer._retained_lock = threading.Lock()
    servicer._retained_requests = {
        "req-1": RetainedPrefillState(
            request_id="req-1",
            block_index=0,
            request_seed=123,
            sequence=[1, 2, 3, 4],
            kv_state=RequestKVState(page_ids=[1], seq_len=4, last_page_len=4),
            cuda_event=None,
        )
    }

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time",
        iter([20.0, 21.0]).__next__,
    )

    servicer._coordinate_kv_transfer(
        SimpleNamespace(
            request_id="req-1",
            block_index=0,
            decode_worker_address="localhost:20101",
            decode_dst_rank=2,
        )
    )

    assert freed == ["req-1"]
    assert len(transfer_reports) == 1
    report = transfer_reports[0]
    assert report.success is True
    assert report.request_id == "req-1"
    assert report.transfer_start_time == pytest.approx(20.0)
    assert report.transfer_end_time == pytest.approx(21.0)


def test_coordinate_kv_transfer_reports_failure(monkeypatch) -> None:
    transfer_reports: list[object] = []
    freed: list[str] = []

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 1
    servicer.model = SimpleNamespace(num_layers=2, num_kv_heads=4, head_dim=8)
    servicer._send_queues = {2: _make_immediate_send_queue()}
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._get_decode_stub = lambda address: SimpleNamespace(
        ReceiveKVCache=lambda request, timeout=None: SimpleNamespace(success=False)
    )
    servicer._free_retained_request = lambda request_id: freed.append(request_id)
    servicer._scheduler_stub = SimpleNamespace(
        ReportKVTransfer=lambda report: transfer_reports.append(report)
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: WorkerStateTimeline.IDLE
    servicer._retained_lock = threading.Lock()
    servicer._retained_requests = {
        "req-1": RetainedPrefillState(
            request_id="req-1",
            block_index=0,
            request_seed=123,
            sequence=[1, 2, 3, 4],
            kv_state=RequestKVState(page_ids=[1], seq_len=4, last_page_len=4),
            cuda_event=None,
        )
    }

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time",
        iter([20.0, 21.0]).__next__,
    )

    servicer._coordinate_kv_transfer(
        SimpleNamespace(
            request_id="req-1",
            block_index=0,
            decode_worker_address="localhost:20101",
            decode_dst_rank=2,
        )
    )

    assert freed == ["req-1"]
    assert len(transfer_reports) == 1
    report = transfer_reports[0]
    assert report.success is False
    assert report.request_id == "req-1"
    assert "rejected" in report.error_message


def test_coordinate_streaming_kv_transfer_reports_success(monkeypatch) -> None:
    transfer_reports: list[object] = []
    freed: list[str] = []
    received: list[object] = []

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 1
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._get_decode_stub = lambda address: SimpleNamespace(
        ReceiveKVCache=lambda request: (
            received.append(request),
            SimpleNamespace(success=True),
        )[1]
    )
    servicer._get_send_queue = lambda dst_rank: SimpleNamespace(
        submit=lambda fn, ctx, first_layer_idx: 20.0
    )
    servicer._scheduler_stub = SimpleNamespace(
        ReportKVTransfer=lambda report: transfer_reports.append(report)
    )
    servicer._free_retained_request = lambda request_id: freed.append(request_id)

    monkeypatch.setattr("sangam.worker.prefill_worker.time.time", lambda: 21.0)

    ctx = StreamingKVTransferContext(
        request_id="req-1",
        block_index=0,
        dst_rank=2,
        decode_worker_id="dw-0",
        decode_worker_address="localhost:20101",
        page_ids=[1, 3],
        num_layers=2,
        layer_events=[None, None],
        layer_ready=__import__("queue").Queue(),
        done_event=threading.Event(),
    )
    ctx.layer_ready.put(0)

    receive_request = sangam_pb2.ReceiveKVCacheRequest(
        request_id="req-1",
        src_rank=1,
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        seq_length=20,
        streaming=True,
        auto_enqueue_decode=False,
    )

    servicer._coordinate_streaming_kv_transfer(ctx, receive_request)

    assert ctx.done_event.is_set()
    assert freed == ["req-1"]
    assert received[0].streaming is True
    assert received[0].auto_enqueue_decode is False
    assert transfer_reports[0].success is True
    assert transfer_reports[0].transfer_start_time == pytest.approx(20.0)
    assert transfer_reports[0].transfer_end_time == pytest.approx(21.0)


def test_coordinate_streaming_kv_transfer_times_out_waiting_for_first_layer(
    monkeypatch,
) -> None:
    transfer_reports: list[object] = []
    freed: list[str] = []

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 1
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._scheduler_stub = SimpleNamespace(
        ReportKVTransfer=lambda report: transfer_reports.append(report)
    )
    servicer._free_retained_request = lambda request_id: freed.append(request_id)
    servicer._config = SimpleNamespace(
        streaming_layer_ready_timeout_s=0.001,
        streaming_recv_join_timeout_s=0.001,
        kv_transfer_timeout_s=0.5,
    )

    ctx = StreamingKVTransferContext(
        request_id="req-1",
        block_index=0,
        dst_rank=2,
        decode_worker_id="dw-0",
        decode_worker_address="localhost:20101",
        page_ids=[1, 3],
        num_layers=2,
        layer_events=[None, None],
        layer_ready=__import__("queue").Queue(),
        done_event=threading.Event(),
    )

    receive_request = sangam_pb2.ReceiveKVCacheRequest(
        request_id="req-1",
        src_rank=1,
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        seq_length=20,
        streaming=True,
        auto_enqueue_decode=False,
    )

    servicer._coordinate_streaming_kv_transfer(ctx, receive_request)

    assert ctx.done_event.is_set()
    assert freed == ["req-1"]
    assert transfer_reports[0].success is False
    assert "first streaming KV layer" in transfer_reports[0].error_message


def test_do_streaming_transfer_times_out_waiting_for_next_layer(monkeypatch) -> None:
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer.device = torch.device("cpu")
    servicer._get_send_stream = lambda dst_rank: None
    servicer._prefill_kv_pool = SimpleNamespace(
        kv_data=[torch.zeros((4, 2, 16, 2, 8), dtype=torch.bfloat16) for _ in range(2)]
    )

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.send_paged_kv_layer_async",
        lambda kv_layer, page_ids, dst_rank, stream=None: [],
    )
    servicer._config = SimpleNamespace(
        streaming_layer_ready_timeout_s=0.001,
        streaming_recv_join_timeout_s=0.001,
        kv_transfer_timeout_s=0.5,
    )

    ctx = StreamingKVTransferContext(
        request_id="req-1",
        block_index=0,
        dst_rank=2,
        decode_worker_id="dw-0",
        decode_worker_address="localhost:20101",
        page_ids=[1, 3],
        num_layers=2,
        layer_events=[None, None],
        layer_ready=__import__("queue").Queue(),
        done_event=threading.Event(),
    )

    with pytest.raises(RuntimeError, match="next streaming KV layer"):
        servicer._do_streaming_transfer(ctx, first_layer_idx=0)


def test_processing_loop_requeues_request_that_would_exceed_prefill_batch_cap(
    monkeypatch,
) -> None:
    first_req = SimpleNamespace(
        arrival_time=1.0,
        request_id="req-1",
        block_index=0,
        sequence_ids=list(range(4000)),
        block_start=0,
        block_end=4000,
        request_seed=123,
        temperature=0.0,
        unmasking_strategy="random",
        mask_id=7,
        prefill_enqueue_time=10.0,
        HasField=lambda name: False,
    )
    second_req = SimpleNamespace(
        arrival_time=2.0,
        request_id="req-2",
        block_index=0,
        sequence_ids=list(range(512)),
        block_start=0,
        block_end=512,
        request_seed=456,
        temperature=0.0,
        unmasking_strategy="random",
        mask_id=7,
        prefill_enqueue_time=10.0,
        HasField=lambda name: False,
    )

    class _TwoItemQueue:
        def __init__(self) -> None:
            self._items = [
                (first_req.arrival_time, first_req.request_id, first_req),
                (second_req.arrival_time, second_req.request_id, second_req),
            ]
            self._get_calls = 0
            self.requeued = []

        def get(self, timeout=None):
            if self._get_calls == 0:
                self._get_calls += 1
                return self._items.pop(0)
            raise KeyboardInterrupt

        def get_nowait(self):
            import queue

            if self._items:
                return self._items.pop(0)
            raise queue.Empty

        def put(self, item):
            self.requeued.append(item)

        def qsize(self) -> int:
            return len(self._items)

    queue = _TwoItemQueue()
    processed = []
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._draining = __import__("threading").Event()
    servicer._request_queue = queue
    servicer._active_batch_size = 0
    servicer._max_prefill_tokens_per_batch = 4096
    servicer._kv_page_size = 16
    servicer._prefill_queue_policy = PrefillQueuePolicy.ARRIVAL_ORDER
    servicer._prefill_kv_pool = SimpleNamespace(
        allocator=SimpleNamespace(num_free=1024)
    )
    servicer._process_batch = lambda batch: processed.append(batch)
    servicer._overhead_tracker = _noop_overhead_tracker()

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time",
        iter([11.0]).__next__,
    )

    with pytest.raises(KeyboardInterrupt):
        servicer._processing_loop()

    assert processed == [[(first_req, 11.0)]]
    assert queue.requeued == [
        (
            (second_req.arrival_time, second_req.request_id),
            second_req.request_id,
            second_req,
        )
    ]


def test_processing_loop_allows_oversized_first_request(monkeypatch) -> None:
    enqueue_req = SimpleNamespace(
        arrival_time=1.0,
        request_id="req-oversized",
        block_index=0,
        sequence_ids=list(range(5000)),
        block_start=0,
        block_end=5000,
        request_seed=123,
        temperature=0.0,
        unmasking_strategy="random",
        mask_id=7,
        prefill_enqueue_time=10.0,
        HasField=lambda name: False,
    )

    class _SingleItemQueue:
        def __init__(self, item) -> None:
            self._item = item
            self._calls = 0

        def get(self, timeout=None):
            if self._calls == 0:
                self._calls += 1
                return self._item
            raise KeyboardInterrupt

        def get_nowait(self):
            import queue

            raise queue.Empty

        def put(self, item):
            pytest.fail("did not expect oversized first request to be requeued")

        def qsize(self) -> int:
            return 0

    processed = []
    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._draining = __import__("threading").Event()
    servicer._request_queue = _SingleItemQueue(
        (enqueue_req.arrival_time, enqueue_req.request_id, enqueue_req)
    )
    servicer._active_batch_size = 0
    servicer._max_prefill_tokens_per_batch = 4096
    servicer._kv_page_size = 16
    servicer._prefill_kv_pool = SimpleNamespace(
        allocator=SimpleNamespace(num_free=1024)
    )
    servicer._process_batch = lambda batch: processed.append(batch)
    servicer._overhead_tracker = _noop_overhead_tracker()

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time",
        iter([11.0]).__next__,
    )

    with pytest.raises(KeyboardInterrupt):
        servicer._processing_loop()

    assert processed == [[(enqueue_req, 11.0)]]


def test_prefill_worker_passes_kv_page_size_to_servicer(monkeypatch) -> None:
    captured: dict[str, object] = {}
    gpu_resources = object()

    def _fake_servicer(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.PrefillWorkerServicer",
        _fake_servicer,
    )
    monkeypatch.setattr(
        "sangam.worker.prefill_worker.create_worker_shared_gpu_resources",
        lambda **kwargs: captured.update(kwargs) or gpu_resources,
    )

    config = _make_prefill_worker_config(kv_page_size=32, kv_max_pages=1234)
    worker = PrefillWorker.__new__(PrefillWorker)
    worker._config = config

    worker._create_servicer(
        model=SimpleNamespace(num_layers=1, num_kv_heads=1, head_dim=1),
        device=torch.device("cpu"),
    )

    assert captured["gpu_resources"] is gpu_resources
    assert captured["kv_page_size"] == 32
    assert captured["kv_max_pages"] == 1234
    assert captured["zero_init"] is False
    assert captured["config"] is config


def test_prefill_worker_registration_includes_kv_capacity() -> None:
    worker = PrefillWorker.__new__(PrefillWorker)
    worker._config = _make_prefill_worker_config(kv_page_size=32, kv_max_pages=1234)

    assert worker._registration_extra_fields() == {
        "max_pages": 1234,
        "page_size": 32,
    }


def test_concurrent_kv_transfers_to_same_dest_preserve_order(monkeypatch) -> None:
    """Regression: two concurrent transfers to the same decode worker must
    issue gRPC ReceiveKVCache and send-queue submission in matching order.

    Without _dest_locks, the gRPC calls can arrive at the decode worker in a
    different order than the NCCL sends are submitted to the per-destination
    send queue, causing
    NCCL FIFO mismatch (watchdog timeout or silent data corruption).
    """
    grpc_order: list[str] = []
    send_order: list[str] = []

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 0
    servicer.model = SimpleNamespace(num_layers=2, num_kv_heads=4, head_dim=8)
    servicer._retained_lock = threading.Lock()
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._retained_requests = {
        "req-A": RetainedPrefillState(
            request_id="req-A",
            block_index=0,
            request_seed=1,
            sequence=[1, 2, 3],
            kv_state=RequestKVState(page_ids=[1], seq_len=3, last_page_len=3),
            cuda_event=None,
        ),
        "req-B": RetainedPrefillState(
            request_id="req-B",
            block_index=0,
            request_seed=2,
            sequence=[4, 5, 6, 7],
            kv_state=RequestKVState(page_ids=[2], seq_len=4, last_page_len=4),
            cuda_event=None,
        ),
    }

    def _fake_receive_kv(request, timeout=None):
        grpc_order.append(request.request_id)
        return SimpleNamespace(success=True)

    servicer._get_decode_stub = lambda addr: SimpleNamespace(
        ReceiveKVCache=_fake_receive_kv
    )

    def _fake_send_submit(fn, request_id, dst_rank):
        send_order.append(request_id)

    servicer._send_queues = {2: _make_immediate_send_queue(on_submit=_fake_send_submit)}
    servicer._free_retained_request = lambda rid: None
    servicer._scheduler_stub = SimpleNamespace(ReportKVTransfer=lambda r: None)
    servicer._prefill_kv_pool = SimpleNamespace(free=lambda pids: None)

    monkeypatch.setattr(
        "sangam.worker.prefill_worker.time.time",
        lambda: 1.0,
    )

    req_a = SimpleNamespace(
        request_id="req-A",
        block_index=0,
        decode_worker_address="localhost:20102",
        decode_dst_rank=2,
        decode_request=None,
    )
    req_b = SimpleNamespace(
        request_id="req-B",
        block_index=0,
        decode_worker_address="localhost:20102",
        decode_dst_rank=2,
        decode_request=None,
    )

    # Launch both threads targeting the same decode_dst_rank.
    # The _dest_locks serialise them so gRPC and send order always match.
    t_a = threading.Thread(target=servicer._coordinate_kv_transfer, args=(req_a,))
    t_b = threading.Thread(target=servicer._coordinate_kv_transfer, args=(req_b,))
    t_a.start()
    time.sleep(0.01)  # give A a head start to acquire the lock first
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    # Both requests completed
    assert len(grpc_order) == 2
    assert len(send_order) == 2

    # The critical invariant: gRPC recv order must match send order
    assert grpc_order == send_order, (
        f"Order mismatch — gRPC recv order {grpc_order} != send order {send_order}. "
        "This would cause NCCL FIFO mismatch."
    )


def test_coordinate_kv_transfer_bounds_send_wait_when_recv_returns_failure(
    monkeypatch,
) -> None:
    """If the decode side rejects the recv (e.g. KV pool OOM), the prefill
    coordinator must not block forever on the NCCL send queue. It must
    report KVTransferReport(success=False) within a bounded time."""

    transfer_reports: list[object] = []
    freed: list[str] = []

    # Send queue submit_async returns a work item that NEVER fires its
    # done_event, simulating an NCCL send parked with no peer.
    stuck_event = threading.Event()  # intentionally never set

    def _stuck_submit_async(fn, *args, **kwargs):
        return SimpleNamespace(done_event=stuck_event, result=None, exception=None)

    stuck_send_queue = SimpleNamespace(submit_async=_stuck_submit_async)

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 1
    servicer.model = SimpleNamespace(num_layers=2, num_kv_heads=4, head_dim=8)
    servicer._send_queues = {2: stuck_send_queue}
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._get_decode_stub = lambda address: SimpleNamespace(
        ReceiveKVCache=lambda request, timeout=None: SimpleNamespace(success=False)
    )
    servicer._free_retained_request = lambda request_id: freed.append(request_id)
    servicer._scheduler_stub = SimpleNamespace(
        ReportKVTransfer=lambda report: transfer_reports.append(report)
    )
    servicer._retained_lock = threading.Lock()
    servicer._retained_requests = {
        "req-1": RetainedPrefillState(
            request_id="req-1",
            block_index=0,
            request_seed=123,
            sequence=[1, 2, 3, 4],
            kv_state=RequestKVState(page_ids=[1], seq_len=4, last_page_len=4),
            cuda_event=None,
        )
    }

    start = time.time()
    servicer._coordinate_kv_transfer(
        SimpleNamespace(
            request_id="req-1",
            block_index=0,
            decode_worker_address="localhost:20101",
            decode_dst_rank=2,
        )
    )
    elapsed = time.time() - start

    # The recv returned success=False immediately; the coordinator must
    # surface that as a failed transfer without blocking on the stuck send.
    assert elapsed < 5.0, f"_coordinate_kv_transfer hung for {elapsed:.2f}s"
    assert freed == ["req-1"]
    assert len(transfer_reports) == 1
    report = transfer_reports[0]
    assert report.success is False
    assert "rejected" in report.error_message


def test_coordinate_kv_transfer_bounds_recv_join_when_grpc_hangs(
    monkeypatch,
) -> None:
    """If the decode-side gRPC hangs (e.g. RPC queue stuck), the coordinator
    must not wait forever for the recv thread; it must time out and report
    failure so the scheduler can retry."""

    transfer_reports: list[object] = []
    freed: list[str] = []

    def _hanging_receive(request, timeout=None):
        # Simulate a stuck decode worker: block past the configured timeout.
        time.sleep(5.0)
        return SimpleNamespace(success=True)

    servicer = PrefillWorkerServicer.__new__(PrefillWorkerServicer)
    servicer.worker_id = "pw-0"
    servicer.dist_rank = 1
    servicer.model = SimpleNamespace(num_layers=2, num_kv_heads=4, head_dim=8)
    servicer._send_queues = {2: _make_immediate_send_queue()}
    servicer._dest_locks = defaultdict(threading.Lock)
    servicer._config = SimpleNamespace(
        kv_transfer_timeout_s=0.5,
        streaming_layer_ready_timeout_s=0.5,
        streaming_recv_join_timeout_s=0.5,
    )
    servicer._get_decode_stub = lambda address: SimpleNamespace(
        ReceiveKVCache=_hanging_receive
    )
    servicer._free_retained_request = lambda request_id: freed.append(request_id)
    servicer._scheduler_stub = SimpleNamespace(
        ReportKVTransfer=lambda report: transfer_reports.append(report)
    )
    servicer._retained_lock = threading.Lock()
    servicer._retained_requests = {
        "req-1": RetainedPrefillState(
            request_id="req-1",
            block_index=0,
            request_seed=123,
            sequence=[1, 2, 3, 4],
            kv_state=RequestKVState(page_ids=[1], seq_len=4, last_page_len=4),
            cuda_event=None,
        )
    }

    start = time.time()
    servicer._coordinate_kv_transfer(
        SimpleNamespace(
            request_id="req-1",
            block_index=0,
            decode_worker_address="localhost:20101",
            decode_dst_rank=2,
        )
    )
    elapsed = time.time() - start

    assert elapsed < 3.0, f"_coordinate_kv_transfer hung for {elapsed:.2f}s"
    assert freed == ["req-1"]
    assert len(transfer_reports) == 1
    report = transfer_reports[0]
    assert report.success is False
    assert "Timed out" in report.error_message
