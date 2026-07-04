import heapq
import queue
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sangam.worker.colocated_worker import (
    ActiveDecodeRequest,
    ColocatedWorkerServicer,
    WaitingRequest,
)
from sangam.batch import BatchRequestUpdate
from sangam.kv_cache.paged_kv_cache import RequestKVState
from sangam.proto import sangam_pb2
from sangam.sampling_parameters import SamplingParameters
from sangam.types import PrefillQueuePolicy
from sangam.worker.prefill_queue_policy import compute_prefill_queue_priority

MASK_ID = 126336


def _noop_overhead_tracker() -> SimpleNamespace:
    return SimpleNamespace(
        time_block=lambda *args, **kwargs: nullcontext(),
        drain=lambda: [],
    )


class _RecordingLock:
    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_enqueue_request(**overrides) -> SimpleNamespace:
    values = {
        "request_id": "req-1",
        "sequence_ids": [1, MASK_ID, MASK_ID, 4],
        "block_start": 1,
        "block_end": 3,
        "block_index": 0,
        "arrival_time": 1.0,
        "total_generation_blocks": 2,
        "request_seed": 123,
        "sampling_parameters": SamplingParameters().to_proto(),
        "mask_id": MASK_ID,
        "prefill_enqueue_time": 10.0,
        "HasField": lambda name: name == "sampling_parameters",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_active(block_start: int, block_end: int) -> SimpleNamespace:
    return SimpleNamespace(block_start=block_start, block_end=block_end)


def _waiting(enqueue_req: SimpleNamespace) -> WaitingRequest:
    if not hasattr(enqueue_req, "request_id"):
        enqueue_req.request_id = f"req-{id(enqueue_req)}"
    if not hasattr(enqueue_req, "arrival_time"):
        enqueue_req.arrival_time = 1.0
    if not hasattr(enqueue_req, "block_index"):
        enqueue_req.block_index = 0
    if not hasattr(enqueue_req, "total_generation_blocks"):
        enqueue_req.total_generation_blocks = 1
    return WaitingRequest(
        priority=compute_prefill_queue_priority(
            enqueue_req, PrefillQueuePolicy.ARRIVAL_ORDER
        ),
        enqueue_req=enqueue_req,
    )


def test_try_prefill_ignores_decode_batch_size_limit() -> None:
    enqueue_req = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=64,
        sequence_ids=list(range(64)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(enqueue_req)]
    servicer._active_batch = [_make_active(0, 32), _make_active(32, 64)]
    servicer._max_batch_size = 2
    servicer._max_tokens_per_iteration = 1024
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == [[enqueue_req]]


def test_try_prefill_stops_at_memory_blocked_head_request() -> None:
    """Strict head-of-line: a memory-blocked head halts the scan.

    The head needs 8 KV pages but only 1 is free. Rather than skipping ahead
    to the smaller second request, admission stops and both requests are
    requeued so the higher-priority head is not leapfrogged.
    """
    first = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(128)),
    )
    second = SimpleNamespace(
        request_id="req-2",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(16)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [
        _waiting(first),
        _waiting(second),
    ]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 256
    servicer._kv_page_size = 16
    servicer._kv_pool = SimpleNamespace(allocator=SimpleNamespace(num_free=1))
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == []
    assert len(servicer._waiting_heap) == 2
    assert servicer._waiting_heap[0].enqueue_req is first


def test_try_prefill_stops_at_budget_blocked_head_request() -> None:
    """Strict head-of-line: a budget-blocked head halts the scan.

    An active decode batch consumes most of the iteration budget, leaving the
    head over budget while it is not oversized-first eligible (active batch is
    non-empty). The smaller second request would have fit, but admission stops
    at the head and both requests are requeued.
    """
    first = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(128)),
    )
    second = SimpleNamespace(
        request_id="req-2",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(16)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [
        _waiting(first),
        _waiting(second),
    ]
    servicer._active_batch = [_make_active(0, 200)]
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 256
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == []
    assert len(servicer._waiting_heap) == 2
    assert servicer._waiting_heap[0].enqueue_req is first


def test_collect_prefill_batch_reserves_kv_pages_under_lock_in_hybrid_mode() -> None:
    """Hybrid mode: admission must reserve KV pages atomically under the lock.

    Without atomic reservation, a concurrent ReceiveKVCache from a remote
    prefill node could drain pages between admission and allocation. This
    test exercises the post-fix invariant: pages are taken from the pool at
    admission time, and reservations land in `_reserved_prefill_kv`.
    """
    allocator = SimpleNamespace(num_free=8)

    class _FakePool:
        def __init__(self) -> None:
            self.allocator = allocator
            self.allocated: list[int] = []
            self.freed: list[list[int]] = []
            self._next_id = 100

        def allocate(self, seq_len: int) -> tuple[list[int], int]:
            need = (seq_len + 3) // 4  # page_size=4 below
            if allocator.num_free < need:
                raise RuntimeError(
                    f"KV page OOM: need {need} pages, only {allocator.num_free} free"
                )
            allocator.num_free -= need
            page_ids = list(range(self._next_id, self._next_id + need))
            self._next_id += need
            self.allocated.append(seq_len)
            return page_ids, seq_len % 4 or 4

        def free(self, page_ids: list[int]) -> None:
            self.freed.append(list(page_ids))
            allocator.num_free += len(page_ids)

    first = SimpleNamespace(
        request_id="req-A",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(8)),
    )
    second = SimpleNamespace(
        request_id="req-B",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(16)),
    )

    pool = _FakePool()
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(first), _waiting(second)]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 256
    servicer._kv_page_size = 4
    servicer._kv_pool = pool
    servicer._state_lock = _RecordingLock()
    servicer._reserved_prefill_kv = {}
    servicer._deficit_tokens = 0

    called: list[list] = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    # Both requests admitted; pages reserved under the lock.
    assert [req.request_id for req in called[0]] == ["req-A", "req-B"]
    assert set(servicer._reserved_prefill_kv) == {"req-A", "req-B"}
    assert servicer._reserved_prefill_kv["req-A"].seq_len == 8
    assert servicer._reserved_prefill_kv["req-B"].seq_len == 16
    # 2 pages for req-A + 4 pages for req-B = 6 pages taken from 8.
    assert allocator.num_free == 2
    assert servicer._state_lock.enter_count >= 1


def test_collect_prefill_batch_requeues_on_kv_oom_in_hybrid_mode() -> None:
    """If the pool is drained mid-admission (e.g. concurrent receive races
    won the lock first), the candidate must be requeued, not crash the loop.
    """
    allocator = SimpleNamespace(num_free=2)

    class _TightPool:
        def __init__(self) -> None:
            self.allocator = allocator

        def allocate(self, seq_len: int) -> tuple[list[int], int]:
            need = (seq_len + 3) // 4
            if allocator.num_free < need:
                raise RuntimeError("KV page OOM")
            allocator.num_free -= need
            return list(range(need)), seq_len % 4 or 4

        def free(self, page_ids: list[int]) -> None:
            allocator.num_free += len(page_ids)

    too_big = SimpleNamespace(
        request_id="req-big",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(64)),  # needs 16 pages, only 2 free
    )

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(too_big)]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 256
    servicer._kv_page_size = 4
    servicer._kv_pool = _TightPool()
    servicer._state_lock = _RecordingLock()
    servicer._reserved_prefill_kv = {}
    servicer._deficit_tokens = 0

    called: list[list] = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == []  # admission rejected
    assert servicer._reserved_prefill_kv == {}
    assert len(servicer._waiting_heap) == 1
    assert servicer._waiting_heap[0].enqueue_req is too_big
    assert allocator.num_free == 2  # untouched


def test_memory_block_carries_prior_deficit_without_adding_budget() -> None:
    """A memory stall must not manufacture fresh token budget.

    Nothing can be admitted because the head needs 8 KV pages but only 1 is
    free. The prior deficit carries over unchanged; this iteration's fresh
    budget is NOT folded into the deficit.
    """
    head = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(128)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(head)]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 256
    servicer._kv_page_size = 16
    servicer._kv_pool = SimpleNamespace(allocator=SimpleNamespace(num_free=1))
    servicer._deficit_tokens = 500

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == []
    assert servicer._waiting_heap[0].enqueue_req is head
    assert servicer._deficit_tokens == 500


def test_memory_block_debits_deficit_spent_by_admitted_request() -> None:
    """When an admitted request spent from the prior deficit before a later
    request memory-blocks, the carried-over deficit is debited by that spend.
    """
    head = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(150)),
    )
    second = SimpleNamespace(
        request_id="req-2",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        sequence_ids=list(range(16)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(head), _waiting(second)]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 100
    servicer._kv_page_size = 16
    # Head needs ceil(150/16)=10 pages; second would push to 11. Only 10 free,
    # so the head is admitted and the second memory-blocks.
    servicer._kv_pool = SimpleNamespace(allocator=SimpleNamespace(num_free=10))
    servicer._deficit_tokens = 200

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert [req.request_id for req in called[0]] == ["req-1"]
    assert servicer._waiting_heap[0].enqueue_req is second
    # iteration_budget=100, head spent 150 → overspend=50 from the 200 deficit.
    assert servicer._deficit_tokens == 150


def test_run_batch_prefill_reports_completion_and_admits_to_decode(monkeypatch) -> None:
    kv_state = RequestKVState(page_ids=[2], seq_len=4, last_page_len=4)
    reports = []

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer.device = torch.device("cpu")
    servicer._kv_pool = None
    servicer._state = {}
    servicer._decode_ready_queue = []
    servicer._active_batch = []
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: (
            [
                {
                    "kv_state": kv_state,
                    "sampled_sequence": [1, 9, MASK_ID, 4],
                    "num_unmasked_tokens": 1,
                }
            ],
            [],
            1,
            0.25,
            None,
        )
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: reports.append(report),
    )
    servicer._overhead_tracker = _noop_overhead_tracker()

    timeline = iter([11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch([_make_enqueue_request()], [])

    assert servicer._state["req-1"] == kv_state
    assert servicer._active_batch == []
    assert len(servicer._decode_ready_queue) == 1
    ready_req = servicer._decode_ready_queue[0]
    assert isinstance(ready_req, ActiveDecodeRequest)
    assert ready_req.request_id == "req-1"
    assert ready_req.prefill_duration == pytest.approx(1.0)
    assert ready_req.prefill_queue_wait_duration == pytest.approx(1.0)
    assert ready_req.prefill_num_unmasked_tokens == 1
    assert ready_req.sequence_ids == [1, 9, MASK_ID, 4]
    assert len(reports) == 1
    report = reports[0]
    assert report.batch_phase == sangam_pb2.BATCH_PHASE_PREFILL
    assert report.num_unmasked_tokens == 1
    assert report.sampling_duration == pytest.approx(0.25)
    update = report.request_updates[0]
    assert update.success is True
    assert update.prefill_duration == pytest.approx(1.0)
    assert update.prefill_queue_wait_duration == pytest.approx(1.0)
    assert list(update.updated_sequence) == [1, 9, MASK_ID, 4]


def test_run_batch_prefill_marks_completed_block_and_skips_decode_admission(
    monkeypatch,
) -> None:
    kv_state = RequestKVState(page_ids=[2], seq_len=4, last_page_len=4)
    reports = []
    freed_page_ids = []

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer.device = torch.device("cpu")
    servicer._kv_pool = SimpleNamespace(
        free=lambda page_ids: freed_page_ids.append(page_ids)
    )
    servicer._state = {}
    servicer._decode_ready_queue = []
    servicer._active_batch = []
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: (
            [
                {
                    "kv_state": kv_state,
                    "sampled_sequence": [1, 9, 8, 4],
                    "num_unmasked_tokens": 2,
                }
            ],
            [],
            2,
            0.25,
            None,
        )
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: reports.append(report),
    )
    servicer._overhead_tracker = _noop_overhead_tracker()

    timeline = iter([11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch([_make_enqueue_request()], [])

    assert servicer._state == {}
    assert servicer._decode_ready_queue == []
    assert freed_page_ids == [[2]]
    assert reports[0].request_updates[0].block_completed is True


def test_run_batch_prefill_uses_state_lock_for_hybrid_state_updates(
    monkeypatch,
) -> None:
    kv_state = RequestKVState(page_ids=[2], seq_len=4, last_page_len=4)
    lock = _RecordingLock()

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer.device = torch.device("cpu")
    servicer._kv_pool = None
    servicer._state_lock = lock
    servicer._state = {}
    servicer._decode_ready_queue = []
    servicer._active_batch = []
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: (
            [
                {
                    "kv_state": kv_state,
                    "sampled_sequence": [1, 9, MASK_ID, 4],
                    "num_unmasked_tokens": 1,
                }
            ],
            [],
            1,
            0.25,
            None,
        )
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._scheduler_stub = SimpleNamespace(ReportBatchMetrics=lambda report: None)
    servicer._overhead_tracker = _noop_overhead_tracker()

    timeline = iter([11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch([_make_enqueue_request()], [])

    assert servicer._state["req-1"] == kv_state
    assert lock.enter_count >= 1


def test_run_batched_prefill_local_raises_on_mixed_block_lengths() -> None:
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="Mixed batch block lengths"):
        servicer._do_run_batched_prefill_local(
            [
                _make_enqueue_request(request_id="req-1", block_start=0, block_end=2),
                _make_enqueue_request(request_id="req-2", block_start=0, block_end=3),
            ]
        )


def test_run_decode_iteration_reports_batch_and_advances_steps(monkeypatch) -> None:
    batch_reports = []

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer._kv_pool = None
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, prefill_batch, decode_reqs: (
            [],
            [
                BatchRequestUpdate(
                    request_id="req-1",
                    block_index=0,
                    success=True,
                    updated_sequence=[1, 7, 4],
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=0,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                )
            ],
            1,
            0.15,
            None,
        )
    )
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: batch_reports.append(report)
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._overhead_tracker = _noop_overhead_tracker()
    servicer._active_batch = [
        ActiveDecodeRequest(
            request_id="req-1",
            sequence_ids=torch.tensor([[1, MASK_ID, 4]], dtype=torch.long),
            block_start=1,
            block_end=3,
            block_index=0,
            request_seed=123,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            step_index=1,
        )
    ]

    timeline = iter([20.0, 21.0, 22.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch(
        [],
        [req for req in servicer._active_batch if req.step_index > 0],
    )

    assert servicer._active_batch[0].step_index == 2
    assert servicer._active_batch[0].num_forward_evals == 1
    assert len(batch_reports) == 1
    report = batch_reports[0]
    assert report.batch_phase == sangam_pb2.BATCH_PHASE_DECODE
    assert report.sampling_duration == pytest.approx(0.15)
    update = report.request_updates[0]
    assert update.num_unmasked_tokens == 1
    assert update.num_forward_evals_in_batch_phase == 1


def test_run_decode_iteration_does_not_count_completed_rows_as_forward_passes(
    monkeypatch,
) -> None:
    batch_reports = []

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer._kv_pool = None
    servicer._state = {}
    servicer._state_lock = None
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, prefill_batch, decode_reqs: ([], [], 0, 0.0, None)
    )
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: batch_reports.append(report)
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._overhead_tracker = _noop_overhead_tracker()
    servicer._active_batch = [
        ActiveDecodeRequest(
            request_id="req-1",
            sequence_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            block_start=1,
            block_end=3,
            block_index=0,
            request_seed=123,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            step_index=1,
            decode_start_time=10.0,
            decode_queue_wait_duration=0.5,
            num_forward_evals=0,
        )
    ]

    timeline = iter([20.0, 21.0, 22.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch(
        [],
        [req for req in servicer._active_batch if req.step_index > 0],
    )

    assert len(batch_reports) == 1
    report = batch_reports[0]
    assert report.batch_size == 0
    assert report.request_updates[0].num_forward_evals_in_batch_phase == 0
    assert report.request_updates[0].block_completed is True


def test_evict_completed_prunes_active_batch_without_re_freeing(monkeypatch) -> None:
    """Eviction only removes completed requests from `_active_batch`.

    Pool pages are freed eagerly inside `_run_batch` (before the batch
    report is sent) so the scheduler sees a consistent state when it
    processes `block_completed=True`. `_evict_completed` must therefore
    not submit redundant free work to the GPU queue.
    """
    gpu_submits: list = []

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: gpu_submits.append((fn, args))
    )
    servicer._active_batch = [
        ActiveDecodeRequest(
            request_id="req-1",
            sequence_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            block_start=1,
            block_end=3,
            block_index=0,
            request_seed=123,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            step_index=2,
            decode_start_time=10.0,
            prefill_duration=0.5,
            prefill_queue_wait_duration=0.25,
            prefill_num_unmasked_tokens=1,
            decode_queue_wait_duration=0.75,
            num_forward_evals=3,
        )
    ]

    monkeypatch.setattr("sangam.worker.colocated_worker.time.time", lambda: 13.0)

    servicer._evict_completed()

    assert servicer._active_batch == []
    assert gpu_submits == []


def test_try_prefill_admits_when_batch_has_capacity() -> None:
    enqueue_req = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=64,
        sequence_ids=list(range(64)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(enqueue_req)]
    servicer._active_batch = [_make_active(0, 32)]
    servicer._max_batch_size = 2
    servicer._max_tokens_per_iteration = 1024
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == [[enqueue_req]]
    assert len(servicer._waiting_heap) == 0


def test_run_mixed_iteration_emits_batch_phase_mixed_with_per_request_phases(
    monkeypatch,
) -> None:
    """_run_batch with both prefill and decode produces batch_phase=MIXED
    and sets PREFILL/DECODE request_phase on each per-request update."""
    from sangam.kv_cache.paged_kv_cache import RequestKVState

    kv_state = RequestKVState(page_ids=[2], seq_len=4, last_page_len=4)
    reports = []

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer.device = torch.device("cpu")
    servicer._kv_pool = None
    servicer._state = {}
    servicer._decode_ready_queue = []
    servicer._active_batch = []
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, prefill_batch, decode_reqs: (
            [
                {
                    "kv_state": kv_state,
                    "sampled_sequence": [1, 9, MASK_ID, 4],
                    "num_unmasked_tokens": 1,
                }
            ],
            [
                BatchRequestUpdate(
                    request_id="decode-req",
                    block_index=0,
                    success=True,
                    updated_sequence=[1, 7, MASK_ID],
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=0,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                )
            ],
            2,
            0.3,
            None,
        )
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: reports.append(report)
    )
    servicer._overhead_tracker = _noop_overhead_tracker()

    active_decode_req = ActiveDecodeRequest(
        request_id="decode-req",
        sequence_ids=torch.tensor([[1, 7, MASK_ID]], dtype=torch.long),
        block_start=1,
        block_end=3,
        block_index=0,
        request_seed=123,
        sampling_parameters=SamplingParameters(),
        mask_id=MASK_ID,
        step_index=1,
    )

    timeline = iter([10.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch(
        [_make_enqueue_request()],
        [active_decode_req],
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.batch_phase == sangam_pb2.BATCH_PHASE_MIXED
    phases_by_id = {u.request_id: u.request_phase for u in report.request_updates}
    assert phases_by_id["req-1"] == sangam_pb2.BATCH_PHASE_PREFILL
    assert phases_by_id["decode-req"] == sangam_pb2.BATCH_PHASE_DECODE


def test_admit_ready_decode_requests_respects_max_batch_size(monkeypatch) -> None:
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.device = torch.device("cpu")
    servicer._max_batch_size = 2
    servicer._active_batch = [
        ActiveDecodeRequest(
            request_id="active-1",
            sequence_ids=torch.tensor([[1, MASK_ID]], dtype=torch.long),
            block_start=0,
            block_end=2,
            block_index=0,
            request_seed=100,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            step_index=1,
        )
    ]
    servicer._decode_ready_queue = [
        ActiveDecodeRequest(
            request_id="ready-1",
            sequence_ids=[1, 2, 3],
            block_start=0,
            block_end=3,
            block_index=0,
            request_seed=101,
            step_index=0,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            prefill_duration=0.5,
            prefill_queue_wait_duration=0.25,
            prefill_num_unmasked_tokens=1,
            ready_time=10.0,
        ),
        ActiveDecodeRequest(
            request_id="ready-2",
            sequence_ids=[4, 5, 6],
            block_start=0,
            block_end=3,
            block_index=0,
            request_seed=102,
            step_index=0,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            prefill_duration=0.5,
            prefill_queue_wait_duration=0.25,
            prefill_num_unmasked_tokens=1,
            ready_time=11.0,
        ),
    ]

    monkeypatch.setattr("sangam.worker.colocated_worker.time.time", lambda: 12.0)

    servicer._admit_ready_decode_requests()

    assert [req.request_id for req in servicer._active_batch] == ["active-1", "ready-1"]
    assert [req.request_id for req in servicer._decode_ready_queue] == ["ready-2"]
    assert servicer._active_batch[-1].decode_queue_wait_duration == pytest.approx(2.0)


def test_drain_external_decodes_queues_only_requests_with_kv_state(
    monkeypatch,
) -> None:
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._state_lock = _RecordingLock()
    servicer._state = {
        "req-ready": RequestKVState(page_ids=[5], seq_len=3, last_page_len=3)
    }
    servicer._external_decode_queue = queue.Queue()
    servicer._decode_ready_queue = []
    servicer._external_decode_queue.put(
        sangam_pb2.EnqueueDecodeRequest(
            request_id="req-ready",
            sequence_ids=[1, MASK_ID, 3],
            block_start=0,
            block_end=3,
            block_index=0,
            request_seed=123,
            mask_id=MASK_ID,
        )
    )
    servicer._external_decode_queue.put(
        sangam_pb2.EnqueueDecodeRequest(
            request_id="req-missing",
            sequence_ids=[4, MASK_ID, 6],
            block_start=0,
            block_end=3,
            block_index=1,
            request_seed=456,
            mask_id=MASK_ID,
        )
    )

    monkeypatch.setattr("sangam.worker.colocated_worker.time.time", lambda: 12.0)

    servicer._drain_external_decodes()

    assert servicer._external_decode_queue.empty()
    assert [req.request_id for req in servicer._decode_ready_queue] == ["req-ready"]
    ready_req = servicer._decode_ready_queue[0]
    assert ready_req.sequence_ids == [1, MASK_ID, 3]
    assert ready_req.step_index == 0
    assert ready_req.ready_time == 12.0
    assert servicer._state_lock.enter_count == 2


def test_run_batch_prefill_queues_for_decode_when_batch_is_nearly_full(
    monkeypatch,
) -> None:
    reports = []
    kv_state_1 = RequestKVState(page_ids=[1], seq_len=4, last_page_len=4)
    kv_state_2 = RequestKVState(page_ids=[2], seq_len=4, last_page_len=4)

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.worker_id = "cw-0"
    servicer.device = torch.device("cpu")
    servicer._kv_pool = None
    servicer._state = {}
    servicer._decode_ready_queue = []
    servicer._active_batch = [
        ActiveDecodeRequest(
            request_id="active-1",
            sequence_ids=torch.tensor([[1, MASK_ID]], dtype=torch.long),
            block_start=0,
            block_end=2,
            block_index=0,
            request_seed=100,
            sampling_parameters=SamplingParameters(),
            mask_id=MASK_ID,
            step_index=1,
        )
    ]
    servicer._max_batch_size = 2
    servicer._gpu_queue = SimpleNamespace(
        submit=lambda fn, *args: (
            [
                {
                    "kv_state": kv_state_1,
                    "sampled_sequence": [1, 9, MASK_ID, 4],
                    "num_unmasked_tokens": 1,
                },
                {
                    "kv_state": kv_state_2,
                    "sampled_sequence": [2, 7, MASK_ID, 5],
                    "num_unmasked_tokens": 1,
                },
            ],
            [],
            2,
            0.3,
            None,
        )
    )
    servicer._build_state_snapshot = lambda state: sangam_pb2.WorkerStateSnapshot(
        state=sangam_pb2.WORKER_STATE_IDLE, timestamp=0.0
    )
    servicer._current_worker_state = lambda: (
        __import__(
            "sangam.metrics.constants", fromlist=["WorkerStateTimeline"]
        ).WorkerStateTimeline.IDLE
    )
    servicer._current_kv_page_stats = lambda: (0, 0, 0)
    servicer._scheduler_stub = SimpleNamespace(
        ReportBatchMetrics=lambda report: reports.append(report)
    )
    servicer._overhead_tracker = _noop_overhead_tracker()

    req1 = _make_enqueue_request(request_id="req-1")
    req2 = _make_enqueue_request(request_id="req-2")
    timeline = iter([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.time.time", lambda: next(timeline)
    )

    servicer._run_batch([req1, req2], [])
    servicer._admit_ready_decode_requests()

    assert set(servicer._state) == {"req-1", "req-2"}
    assert [req.request_id for req in servicer._active_batch] == ["active-1", "req-1"]
    assert [req.request_id for req in servicer._decode_ready_queue] == ["req-2"]
    assert len(reports) == 1
    report = reports[0]
    assert report.batch_size == 2
    assert report.num_unmasked_tokens == 2
    updates = {update.request_id: update for update in report.request_updates}
    assert updates["req-1"].num_unmasked_tokens == 1
    assert updates["req-2"].num_unmasked_tokens == 1


def test_try_prefill_admits_oversized_first_request() -> None:
    enqueue_req = SimpleNamespace(
        request_id="req-oversized",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=5000,
        sequence_ids=list(range(5000)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [_waiting(enqueue_req)]
    servicer._active_batch = []
    servicer._max_batch_size = 2
    servicer._max_tokens_per_iteration = 1024
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == [[enqueue_req]]
    assert len(servicer._waiting_heap) == 0


def test_try_prefill_stops_after_oversized_first_request() -> None:
    first = SimpleNamespace(
        request_id="req-oversized",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=5000,
        sequence_ids=list(range(5000)),
    )
    second = SimpleNamespace(
        request_id="req-next",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=64,
        sequence_ids=list(range(64)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [
        _waiting(first),
        _waiting(second),
    ]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 1024
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == [[first]]
    assert servicer._waiting_heap == [_waiting(second)]


def test_try_prefill_batches_requests_up_to_iteration_budget() -> None:
    first = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=3000,
        sequence_ids=list(range(3000)),
    )
    second = SimpleNamespace(
        request_id="req-2",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=1500,
        sequence_ids=list(range(1500)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [
        _waiting(first),
        _waiting(second),
    ]
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 8192
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == [[first, second]]
    assert servicer._waiting_heap == []


def test_try_prefill_stops_before_request_that_exceeds_iteration_budget() -> None:
    first = SimpleNamespace(
        request_id="req-1",
        arrival_time=1.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=3000,
        sequence_ids=list(range(3000)),
    )
    second = SimpleNamespace(
        request_id="req-2",
        arrival_time=2.0,
        block_index=0,
        total_generation_blocks=1,
        block_end=1500,
        sequence_ids=list(range(1500)),
    )
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = [
        _waiting(first),
        _waiting(second),
    ]
    servicer._active_batch = [_make_active(0, 1024)]
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 4096
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._try_batch_prefill()

    assert called == [[first]]
    assert servicer._waiting_heap == [_waiting(second)]


def test_deficit_tokens_reset_when_waiting_queue_is_empty() -> None:
    """Idle iterations reset the deficit so it does not grow unboundedly."""
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._active_batch = []
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 1024
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0

    called = []
    servicer._run_batch = lambda batch, _decode_reqs: called.append(batch)

    servicer._waiting_heap = []
    servicer._try_batch_prefill()
    assert servicer._deficit_tokens == 0

    servicer._waiting_heap = []
    servicer._try_batch_prefill()
    assert servicer._deficit_tokens == 0


def test_deficit_tokens_does_not_go_negative_when_decode_exceeds_budget() -> None:
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._waiting_heap = []
    servicer._active_batch = [_make_active(0, 1024), _make_active(0, 1024)]
    servicer._max_batch_size = 4
    servicer._max_tokens_per_iteration = 1024
    servicer._kv_page_size = 16
    servicer._kv_pool = None
    servicer._deficit_tokens = 0
    servicer._run_batch = lambda batch, _decode_reqs: None

    servicer._try_batch_prefill()

    assert servicer._deficit_tokens == 0


def test_drain_incoming_uses_fewest_remaining_blocks_priority() -> None:
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._prefill_queue_policy = PrefillQueuePolicy.FEWEST_REMAINING_BLOCKS
    servicer._incoming_queue = queue.Queue()
    servicer._waiting_heap = []

    servicer._incoming_queue.put(
        _make_enqueue_request(
            request_id="early-block",
            block_index=0,
            total_generation_blocks=4,
            arrival_time=1.0,
        )
    )
    servicer._incoming_queue.put(
        _make_enqueue_request(
            request_id="last-block",
            block_index=3,
            total_generation_blocks=4,
            arrival_time=2.0,
        )
    )

    servicer._drain_incoming_inner()

    assert [
        heapq.heappop(servicer._waiting_heap).enqueue_req.request_id for _ in range(2)
    ] == [
        "last-block",
        "early-block",
    ]


def test_paged_forward_pass_uses_batched_sampler_once(monkeypatch) -> None:
    timer_calls = []

    class _FakeDurationTimer:
        def __init__(self, name: str, use_cuda: bool, device=None) -> None:
            timer_calls.append((name, use_cuda, device))
            self.elapsed_s = 0.15

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyModel:
        num_q_heads = 2
        num_layers = 1
        num_kv_heads = 2
        head_dim = 4

        def __init__(self):
            self.model = SimpleNamespace(
                transformer=SimpleNamespace(
                    blocks=[SimpleNamespace(_paged_attn_state=None)]
                )
            )

        def __call__(self, block_tokens):
            return SimpleNamespace(logits=torch.zeros((2, 2, 8), dtype=torch.float32))

    sample_calls = []

    def _sample_batch(reqs, block_tokens, logits):
        sample_calls.append((reqs, block_tokens.clone(), logits.shape))
        return (
            torch.tensor([[31, 32], [41, MASK_ID]], dtype=torch.long),
            torch.tensor([2, 1], dtype=torch.long),
        )

    monkeypatch.setattr(
        "sangam.worker.colocated_worker.pack_mixed_batch",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.run_mixed_paged_forward",
        lambda model, packed_batch: SimpleNamespace(
            packed_logits=torch.zeros((1, 4, 8), dtype=torch.float32),
            item_logits=[
                torch.zeros((1, 2, 8), dtype=torch.float32),
                torch.zeros((1, 2, 8), dtype=torch.float32),
            ],
        ),
    )
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.DurationTimer",
        _FakeDurationTimer,
    )

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.device = torch.device("cpu")
    servicer.model = DummyModel()
    servicer._sampler = SimpleNamespace(sample_batch=_sample_batch)
    servicer._state_lock = _RecordingLock()
    servicer._state = {
        "req-1": RequestKVState(page_ids=[1], seq_len=4, last_page_len=4),
        "req-2": RequestKVState(page_ids=[2], seq_len=4, last_page_len=4),
    }
    servicer._kv_pool = SimpleNamespace(allocator=SimpleNamespace(num_used=0))
    servicer._flashinfer_workspace = object()
    servicer._paged_attention_context = lambda paged_state: nullcontext()

    reqs = [
        ActiveDecodeRequest(
            request_id="req-1",
            sequence_ids=torch.tensor([[1, MASK_ID, MASK_ID, 4]], dtype=torch.long),
            block_start=1,
            block_end=3,
            block_index=0,
            request_seed=123,
            sampling_parameters=SamplingParameters(
                unmasking_strategy="conf_quota",
                fixed_unmask_quota=2,
            ),
            mask_id=MASK_ID,
            step_index=1,
        ),
        ActiveDecodeRequest(
            request_id="req-2",
            sequence_ids=torch.tensor([[5, MASK_ID, MASK_ID, 8]], dtype=torch.long),
            block_start=1,
            block_end=3,
            block_index=1,
            request_seed=456,
            sampling_parameters=SamplingParameters(
                unmasking_strategy="conf_quota",
                fixed_unmask_quota=1,
            ),
            mask_id=MASK_ID,
            step_index=2,
        ),
    ]

    total_unmasked, updates, _, _ = servicer._paged_forward_pass(reqs)

    assert len(sample_calls) == 1
    assert total_unmasked == 3
    assert reqs[0].sequence_ids.tolist() == [[1, 31, 32, 4]]
    assert reqs[1].sequence_ids.tolist() == [[5, 41, MASK_ID, 8]]
    assert [update.num_unmasked_tokens for update in updates] == [2, 1]
    assert timer_calls == [("worker_sampling_decode", False, torch.device("cpu"))]
    assert servicer._state_lock.enter_count >= 1


def test_do_run_mixed_iteration_local_batches_prefill_with_external_decode(
    monkeypatch,
) -> None:
    timer_calls = []
    sample_calls = []
    pack_calls = []
    freed_page_ids = []

    class _FakeDurationTimer:
        def __init__(self, name: str, use_cuda: bool, device=None) -> None:
            timer_calls.append((name, use_cuda, device))
            self.elapsed_s = 0.2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyModel:
        num_q_heads = 2
        num_layers = 1
        num_kv_heads = 2
        head_dim = 4

        def __init__(self):
            self.model = SimpleNamespace(
                transformer=SimpleNamespace(
                    blocks=[SimpleNamespace(_paged_attn_state=None)]
                )
            )

    class _FakePool:
        def __init__(self) -> None:
            self.allocator = SimpleNamespace(num_used=0)

        def allocate(self, seq_len: int) -> tuple[list[int], int]:
            assert seq_len == 4
            return [9], 4

        def free(self, page_ids: list[int]) -> None:
            freed_page_ids.append(page_ids)

    def _sample_batch(reqs, block_tokens, logits):
        sample_calls.append((reqs, block_tokens.clone(), logits.shape))
        return (
            torch.tensor([[31, 32], [41, MASK_ID]], dtype=torch.long),
            torch.tensor([2, 1], dtype=torch.long),
        )

    def _pack_mixed_batch(**kwargs):
        pack_calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "sangam.worker.colocated_worker.pack_mixed_batch", _pack_mixed_batch
    )
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.run_mixed_paged_forward",
        lambda model, packed_batch: SimpleNamespace(
            packed_logits=torch.zeros((1, 6, 8), dtype=torch.float32),
            item_logits=[
                torch.zeros((1, 4, 8), dtype=torch.float32),
                torch.zeros((1, 2, 8), dtype=torch.float32),
            ],
        ),
    )
    monkeypatch.setattr(
        "sangam.worker.colocated_worker.DurationTimer", _FakeDurationTimer
    )

    existing_kv_state = RequestKVState(page_ids=[7], seq_len=4, last_page_len=4)
    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer.device = torch.device("cpu")
    servicer.model = DummyModel()
    servicer._sampler = SimpleNamespace(sample_batch=_sample_batch)
    servicer._state_lock = _RecordingLock()
    servicer._state = {"decode-req": existing_kv_state}
    servicer._kv_pool = _FakePool()
    servicer._flashinfer_workspace = object()

    decode_req = ActiveDecodeRequest(
        request_id="decode-req",
        sequence_ids=torch.tensor([[5, MASK_ID, MASK_ID, 8]], dtype=torch.long),
        block_start=1,
        block_end=3,
        block_index=0,
        request_seed=456,
        sampling_parameters=SamplingParameters(
            unmasking_strategy="conf_quota",
            fixed_unmask_quota=1,
        ),
        mask_id=MASK_ID,
        step_index=1,
    )

    (
        prefill_results,
        decode_updates,
        total_unmasked,
        sampling_duration,
        operation_metrics_seconds,
    ) = servicer._do_run_mixed_iteration_local(
        [_make_enqueue_request(request_id="prefill-req")],
        [decode_req],
    )

    assert freed_page_ids == []
    assert len(pack_calls) == 1
    packed_items = pack_calls[0]["items"]
    assert [item.request_id for item in packed_items] == ["prefill-req", "decode-req"]
    assert [item.phase for item in packed_items] == ["prefill", "decode"]
    assert packed_items[0].query_start == 0
    assert packed_items[0].query_end == 4
    assert packed_items[0].active_kv_len == 4
    assert packed_items[0].kv_state.page_ids == [9]
    assert packed_items[1].query_start == 1
    assert packed_items[1].query_end == 3
    assert packed_items[1].active_kv_len == 4
    assert packed_items[1].kv_state is existing_kv_state

    assert len(sample_calls) == 1
    assert sample_calls[0][1].tolist() == [[MASK_ID, MASK_ID], [MASK_ID, MASK_ID]]
    assert sample_calls[0][2] == torch.Size([2, 2, 8])
    assert total_unmasked == 3
    assert sampling_duration == pytest.approx(0.2)
    assert timer_calls == [("worker_sampling_mixed", False, torch.device("cpu"))]

    assert len(prefill_results) == 1
    assert prefill_results[0]["kv_state"].page_ids == [9]
    assert prefill_results[0]["sampled_sequence"] == [1, 31, 32, 4]
    assert prefill_results[0]["num_unmasked_tokens"] == 2

    assert decode_req.sequence_ids.tolist() == [[5, 41, MASK_ID, 8]]
    assert len(decode_updates) == 1
    assert decode_updates[0].request_id == "decode-req"
    assert decode_updates[0].num_unmasked_tokens == 1
    assert decode_updates[0].request_phase == sangam_pb2.BATCH_PHASE_DECODE
    assert operation_metrics_seconds is None
    assert servicer._state_lock.enter_count >= 1


def test_do_free_state_uses_state_lock() -> None:
    freed = []
    lock = _RecordingLock()

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._state_lock = lock
    servicer._state = {
        "req-1": RequestKVState(page_ids=[7], seq_len=4, last_page_len=4)
    }
    servicer._kv_pool = SimpleNamespace(free=lambda page_ids: freed.append(page_ids))

    servicer._do_free_state("req-1")

    assert servicer._state == {}
    assert freed == [[7]]
    assert lock.enter_count == 1


def test_do_receive_kv_cache_drains_recvs_when_pool_exhausted(monkeypatch) -> None:
    """If kv_pool.allocate raises (pool full), _do_receive_kv_cache must
    drain the matching NCCL recvs into a throwaway buffer for every layer
    so the prefill-side send doesn't block forever on its GPU send queue.
    """

    class _ExhaustedPool:
        page_size = 4
        kv_data = [torch.zeros(8, 2, 4, 1, 8, dtype=torch.bfloat16)]

        def allocate(self, seq_len: int):
            raise RuntimeError("KV page OOM")

    drain_calls: list[dict] = []

    def _fake_drain(*, template, num_pages, src_rank) -> None:
        drain_calls.append(
            {"template": template, "num_pages": num_pages, "src_rank": src_rank}
        )

    monkeypatch.setattr(
        "sangam.worker.colocated_worker.recv_paged_kv_layer_drain", _fake_drain
    )

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)
    servicer._kv_pool = _ExhaustedPool()
    servicer._state_lock = _RecordingLock()
    servicer._state = {}

    request = SimpleNamespace(
        request_id="req-drain",
        src_rank=3,
        num_layers=4,
        seq_length=12,  # ceil(12 / 4) = 3 pages
        sequence_ids=[1, 2, 3, 4],
    )

    response = servicer._do_receive_kv_cache(request)

    assert response.success is False
    assert len(drain_calls) == 4
    assert {c["src_rank"] for c in drain_calls} == {3}
    assert {c["num_pages"] for c in drain_calls} == {3}
    assert all(c["template"] is servicer._kv_pool.kv_data[0] for c in drain_calls)
    # No state should be stored for a rejected receive.
    assert servicer._state == {}
