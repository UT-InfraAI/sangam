"""Tests for the HybridScheduler overflow routing."""

from unittest.mock import MagicMock

import pytest

import sangam.engine.hybrid_scheduler as hybrid_scheduler_module
from sangam.engine.base_scheduler import WorkerInfo
from sangam.engine.hybrid_scheduler import HybridScheduler
from sangam.engine.scheduler_config import HybridSchedulerConfig
from sangam.engine.topology_policy import parse_kv_fast_pairs
from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.proto import sangam_pb2
from sangam.request import Request, RequestStatus
from sangam.sampling_parameters import SamplingParameters
from sangam.types import WorkerType


def _make_hybrid_config(**overrides) -> HybridSchedulerConfig:
    defaults = dict(
        metrics_output_dir="test_output",
        enable_metrics=False,
        enable_individual_batch_metrics=False,
        export_partial_metrics=False,
        block_length=32,
        mask_id=126336,
        max_gen_len=None,
        prefill_scheduler_policy="round_robin",
        decode_grouping_slack_ratio=0.10,
        decode_scheduler_policy="round_robin",
        kv_fast_pairs="",
        kv_topology_alpha=0.7,
        prefill_overload_threshold=4,
        enable_prefill_overflow=True,
        max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
    )
    defaults.update(overrides)
    return HybridSchedulerConfig(**defaults)


def _make_request(**kwargs) -> Request:
    defaults = dict(
        prompt_token_ids=[1, 2, 3],
        gen_length=32,
        block_length=32,
        sampling_parameters=SamplingParameters(),
        mask_id=126336,
        request_seed=123,
    )
    defaults.update(kwargs)
    return Request(**defaults)


@pytest.fixture
def scheduler() -> HybridScheduler:
    return HybridScheduler(_make_hybrid_config())


def _add_prefill_worker(
    scheduler: HybridScheduler,
    worker_id: str = "pw-0",
    *,
    outstanding_prefill_tokens: int = 0,
    free_pages: int | None = 128,
    page_size: int | None = 16,
) -> None:
    scheduler._prefill_workers.append(
        WorkerInfo(
            worker_id=worker_id,
            worker_type=WorkerType.PREFILL,
            address="addr:0",
            dist_rank=0,
            gpu_id=0,
            max_pages=free_pages,
            page_size=page_size,
            free_pages=free_pages,
            outstanding_prefill_tokens=outstanding_prefill_tokens,
        )
    )
    stub = MagicMock()
    stub.EnqueuePrefill.return_value = sangam_pb2.EnqueuePrefillResponse(success=True)
    stub.TriggerKVTransfer.return_value = sangam_pb2.TriggerKVTransferResponse(
        success=True
    )
    scheduler._prefill_stubs[worker_id] = stub


def _add_colocated_worker(
    scheduler: HybridScheduler,
    worker_id: str = "cw-0",
    *,
    outstanding_prefill_tokens: int = 0,
    free_pages: int | None = None,
    page_size: int | None = None,
) -> MagicMock:
    scheduler._colocated_workers.append(
        WorkerInfo(
            worker_id=worker_id,
            worker_type=WorkerType.COLOCATED,
            address="addr:1",
            dist_rank=1,
            gpu_id=1,
            outstanding_prefill_tokens=outstanding_prefill_tokens,
            max_pages=free_pages,
            page_size=page_size,
            free_pages=free_pages,
        )
    )
    stub = MagicMock()
    stub.EnqueueRequest.return_value = sangam_pb2.EnqueueColocatedResponse(success=True)
    scheduler._colocated_stubs[worker_id] = stub
    return stub


def test_parse_kv_fast_pairs_accepts_valid_pairs() -> None:
    assert parse_kv_fast_pairs("0-1,2-3") == {(0, 1), (2, 3)}


def test_parse_kv_fast_pairs_allows_whitespace_and_deduplicates() -> None:
    assert parse_kv_fast_pairs(" 0-1 , 1-0, 2-3 ") == {(0, 1), (2, 3)}


@pytest.mark.parametrize("value", ["0-", "-1", "0-1-2", "a-b", "2-2", ","])
def test_parse_kv_fast_pairs_rejects_invalid_entries(value: str) -> None:
    with pytest.raises(ValueError):
        parse_kv_fast_pairs(value)


def test_prefill_completed_block_from_prefill_worker_skips_decode_and_completes(
    scheduler,
) -> None:
    _add_prefill_worker(scheduler)
    colocated_stub = _add_colocated_worker(scheduler)
    scheduler._metrics_store.on_block_decode_end = MagicMock()

    req = _make_request()
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    report = sangam_pb2.BatchMetricsReport(
        worker_id="pw-0",
        worker_type="prefill",
        batch_size=1,
        prompt_len=len(req.sequence_ids),
        gen_len=0,
        batch_start_time=100.0,
        batch_end_time=100.1,
        kv_total_pages=64,
        kv_used_pages=16,
        kv_free_pages=48,
        num_unmasked_tokens=32,
        batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
        request_updates=[
            sangam_pb2.BatchRequestUpdate(
                request_id=req.request_id,
                block_index=0,
                success=True,
                updated_sequence=list(range(35)),
                num_unmasked_tokens=32,
                num_forward_evals_in_batch_phase=1,
                request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                prefill_duration=0.1,
                block_completed=True,
            )
        ],
    )

    scheduler._on_batch_metrics(report)

    assert req.status is RequestStatus.COMPLETED
    assert req.num_forward_evals == 1
    assert colocated_stub.EnqueueRequest.call_count == 0
    scheduler._metrics_store.on_block_decode_end.assert_called_once_with(
        req.request_id, 0, 0.0
    )


def test_decode_update_before_transfer_success_does_not_use_overflow_timing(
    scheduler,
) -> None:
    _add_prefill_worker(scheduler)
    _add_colocated_worker(scheduler, free_pages=64, page_size=16)

    req = _make_request()
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="pw-0",
            worker_type="prefill",
            batch_size=1,
            prompt_len=len(req.sequence_ids),
            gen_len=0,
            batch_start_time=100.0,
            batch_end_time=100.1,
            kv_total_pages=64,
            kv_used_pages=16,
            kv_free_pages=48,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    prefill_duration=0.1,
                    prefill_queue_wait_duration=0.0,
                )
            ],
        )
    )

    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=1,
            prompt_len=0,
            gen_len=32,
            batch_start_time=100.6,
            batch_end_time=101.0,
            kv_total_pages=64,
            kv_used_pages=16,
            kv_free_pages=48,
            num_unmasked_tokens=2,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=list(range(35)),
                    num_unmasked_tokens=2,
                    num_forward_evals_in_batch_phase=3,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                    decode_duration=0.4,
                    decode_queue_wait_duration=0.2,
                    block_completed=True,
                )
            ],
        )
    )

    assert req.status is RequestStatus.COMPLETED
    assert req.current_block is None
    assert req.block_states[0].decode_start_time == pytest.approx(100.6)


def test_late_transfer_success_does_not_regress_completed_request(scheduler) -> None:
    _add_prefill_worker(scheduler)
    _add_colocated_worker(scheduler, free_pages=64, page_size=16)

    req = _make_request()
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="pw-0",
            worker_type="prefill",
            batch_size=1,
            prompt_len=len(req.sequence_ids),
            gen_len=0,
            batch_start_time=100.0,
            batch_end_time=100.1,
            kv_total_pages=64,
            kv_used_pages=16,
            kv_free_pages=48,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    prefill_duration=0.1,
                    prefill_queue_wait_duration=0.0,
                )
            ],
        )
    )
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=1,
            prompt_len=0,
            gen_len=32,
            batch_start_time=100.6,
            batch_end_time=101.0,
            kv_total_pages=64,
            kv_used_pages=16,
            kv_free_pages=48,
            num_unmasked_tokens=2,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=list(range(35)),
                    num_unmasked_tokens=2,
                    num_forward_evals_in_batch_phase=3,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                    decode_duration=0.4,
                    decode_queue_wait_duration=0.2,
                    block_completed=True,
                )
            ],
        )
    )

    scheduler._on_kv_transfer(
        sangam_pb2.KVTransferReport(
            worker_id="pw-0",
            request_id=req.request_id,
            block_index=0,
            success=True,
            transfer_start_time=100.2,
            transfer_end_time=100.4,
        )
    )

    assert req.status is RequestStatus.COMPLETED
    assert req.complete_time is not None
    assert req.block_states[0].decode_enqueue_time is None


def test_late_transfer_failure_after_decode_progress_is_ignored(scheduler) -> None:
    _add_prefill_worker(scheduler)
    _add_colocated_worker(scheduler, free_pages=64, page_size=16)

    req = _make_request()
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="pw-0",
            worker_type="prefill",
            batch_size=1,
            prompt_len=len(req.sequence_ids),
            gen_len=0,
            batch_start_time=100.0,
            batch_end_time=100.1,
            kv_total_pages=64,
            kv_used_pages=16,
            kv_free_pages=48,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    prefill_duration=0.1,
                    prefill_queue_wait_duration=0.0,
                )
            ],
        )
    )
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=1,
            prompt_len=0,
            gen_len=32,
            batch_start_time=100.6,
            batch_end_time=101.0,
            kv_total_pages=64,
            kv_used_pages=16,
            kv_free_pages=48,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                    decode_duration=0.4,
                    decode_queue_wait_duration=0.2,
                )
            ],
        )
    )

    scheduler._on_kv_transfer(
        sangam_pb2.KVTransferReport(
            worker_id="pw-0",
            request_id=req.request_id,
            block_index=0,
            success=False,
            transfer_start_time=100.2,
            transfer_end_time=100.4,
            error_message="late failure",
        )
    )

    assert req.status is RequestStatus.DECODING
    assert req.error_message is None
    assert req.block_states[0].prefill_worker_id == "pw-0"


def test_requeue_prefill_request_ignores_stale_block_index(scheduler) -> None:
    _add_prefill_worker(scheduler)

    req = _make_request(gen_length=64, block_length=32)
    req.status = RequestStatus.PREFILLING
    req.current_block_index = 1
    req.block_states[0].prefill_worker_id = "pw-0"
    req.block_states[0].prefill_reserved_pages = 2
    scheduler._prefill_workers[
        0
    ].outstanding_prefill_tokens = req.request_accounting_tokens
    scheduler._event_queue.put = MagicMock()

    scheduler._requeue_prefill_request(req, block_index=0)

    assert req.status is RequestStatus.PREFILLING
    assert req.current_block_index == 1
    assert req.block_states[0].prefill_worker_id is None
    assert req.block_states[0].prefill_reserved_pages == 0
    assert scheduler._prefill_workers[0].outstanding_prefill_tokens == 0
    scheduler._event_queue.put.assert_not_called()


def test_overflow_respects_colocated_free_pages(monkeypatch, scheduler) -> None:
    _add_prefill_worker(scheduler, outstanding_prefill_tokens=4)
    _add_colocated_worker(scheduler, worker_id="cw-0", free_pages=1, page_size=16)
    _add_colocated_worker(scheduler, worker_id="cw-1", free_pages=8, page_size=16)
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    req = _make_request(prompt_token_ids=[1] * 32)

    scheduler._on_request_arrived(req)

    assert req.current_block.prefill_worker_id == "cw-1"
    assert req.current_block.decode_worker_id == "cw-1"
    assert req.current_block.reserved_pages == 4
    enqueue_req = scheduler._submit_enqueue_async.call_args.args[1]
    assert enqueue_req.total_generation_blocks == len(req.block_states)
    assert scheduler._colocated_workers[0].free_pages == 1
    assert scheduler._colocated_workers[1].free_pages == 4


def test_batch_metrics_does_not_overwrite_scheduler_free_pages(
    monkeypatch, scheduler
) -> None:
    """Worker batch reports must not clobber scheduler-tracked free_pages.

    The scheduler is the sole originator of page-consuming actions, so its
    running tally is authoritative. A batch metrics report represents the
    worker's pool state at batch end, which lags any reservation that has not
    yet been allocated by the worker. Overwriting the scheduler tally with a
    stale report previously caused over-commit and KV-transfer failures.
    """
    _add_prefill_worker(scheduler)
    _add_colocated_worker(scheduler, worker_id="cw-0", free_pages=8, page_size=16)
    _add_colocated_worker(
        scheduler,
        worker_id="cw-1",
        free_pages=8,
        page_size=16,
        outstanding_prefill_tokens=1,
    )
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    overflow_req = _make_request(request_id="overflow", prompt_token_ids=[1] * 32)
    scheduler._requests[overflow_req.request_id] = overflow_req
    scheduler._prefill_workers[0].outstanding_prefill_tokens = 4
    scheduler._on_request_arrived(overflow_req)

    # Scheduler reserved 4 pages for the overflow request on cw-0
    # (ceil(64 sequence_ids / 16 page_size) = 4); 8 - 4 = 4 free.
    assert scheduler._colocated_workers[0].free_pages == 4

    # The worker's batch report from BEFORE the reservation was allocated
    # claims kv_free_pages=8 (a stale snapshot). The scheduler must not
    # overwrite its tally with this.
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=0,
            prompt_len=0,
            gen_len=0,
            batch_start_time=99.0,
            batch_end_time=99.1,
            kv_total_pages=8,
            kv_used_pages=0,
            kv_free_pages=8,
            num_unmasked_tokens=0,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[],
        )
    )
    assert scheduler._colocated_workers[0].free_pages == 4, (
        "Stale kv_free_pages=8 must not overwrite scheduler tally of 4"
    )

    # And the inverse: a later report claiming kv_free_pages=0 (i.e. the worker
    # is fully utilized) also must not overwrite. The scheduler still tracks 4
    # free pages from its own bookkeeping.
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=1,
            prompt_len=len(overflow_req.sequence_ids),
            gen_len=0,
            batch_start_time=100.0,
            batch_end_time=100.2,
            kv_total_pages=8,
            kv_used_pages=8,
            kv_free_pages=0,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=overflow_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=overflow_req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    prefill_duration=0.2,
                    prefill_queue_wait_duration=0.0,
                )
            ],
        )
    )
    assert scheduler._colocated_workers[0].free_pages == 4, (
        "Conservative kv_free_pages=0 must not overwrite scheduler tally of 4"
    )


def test_no_overcommit_under_lagging_batch_metrics(monkeypatch, scheduler) -> None:
    """A stale 'pool is empty' report cannot make the scheduler over-commit.

    Even when a worker reports kv_free_pages high (reflecting state from
    before a recent reservation was allocated on the worker), the scheduler
    must not re-pick that worker beyond its own running tally.
    """
    _add_prefill_worker(scheduler, free_pages=0)
    # cw-0 starts with 4 free; no other colocated workers means there's no
    # fallback if cw-0 has no tracked capacity.
    _add_colocated_worker(scheduler, worker_id="cw-0", free_pages=4, page_size=16)
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    # First overflow request consumes all 4 tracked free pages on cw-0
    # (ceil(64/16) = 4 pages).
    req_a = _make_request(request_id="A", prompt_token_ids=[1] * 32)
    scheduler._requests[req_a.request_id] = req_a
    scheduler._prefill_workers[0].outstanding_prefill_tokens = 4
    scheduler._on_request_arrived(req_a)
    assert scheduler._colocated_workers[0].free_pages == 0

    # Stale batch report claims the worker still has all 4 pages free.
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=0,
            prompt_len=0,
            gen_len=0,
            batch_start_time=99.0,
            batch_end_time=99.1,
            kv_total_pages=4,
            kv_used_pages=0,
            kv_free_pages=4,
            num_unmasked_tokens=0,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[],
        )
    )
    assert scheduler._colocated_workers[0].free_pages == 0

    # A second overflow request that needs 4 pages must not be dispatched to
    # cw-0; with no other capacity it should queue as pending.
    req_b = _make_request(request_id="B", prompt_token_ids=[1] * 32)
    scheduler._requests[req_b.request_id] = req_b
    scheduler._prefill_workers[0].outstanding_prefill_tokens = 4
    scheduler._on_request_arrived(req_b)

    assert req_b.current_block.prefill_worker_id is None
    assert req_b.current_block.decode_worker_id is None
    assert req_b.request_id in scheduler._pending_requests
    # cw-0's tally remained at 0; it was not re-picked despite the stale
    # report's claim of capacity. Only the first request was dispatched.
    assert scheduler._colocated_workers[0].free_pages == 0
    assert scheduler._submit_enqueue_async.call_count == 1


def test_divergence_warning_logged_when_worker_reports_fewer_free_pages(
    monkeypatch, scheduler, caplog
) -> None:
    """Warn when the worker reports fewer free pages than the scheduler
    tally — the OOM direction where the next dispatch may exceed the
    worker's actual capacity."""
    import logging

    _add_colocated_worker(scheduler, worker_id="cw-0", free_pages=100, page_size=16)

    hybrid_logger = logging.getLogger("sangam.engine.hybrid_scheduler")
    original_propagate = hybrid_logger.propagate
    hybrid_logger.propagate = True
    scheduler._last_divergence_warning_time.clear()
    try:
        with caplog.at_level(logging.WARNING, logger="sangam.engine.hybrid_scheduler"):
            # Tolerance is max(4, 5% * 100) = 5; scheduler=100 vs reported=50
            # is diff=+50 in the warn direction.
            scheduler._on_batch_metrics(
                sangam_pb2.BatchMetricsReport(
                    worker_id="cw-0",
                    worker_type="colocated",
                    batch_size=0,
                    prompt_len=0,
                    gen_len=0,
                    batch_start_time=99.0,
                    batch_end_time=99.1,
                    kv_total_pages=100,
                    kv_used_pages=50,
                    kv_free_pages=50,
                    num_unmasked_tokens=0,
                    batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    request_updates=[],
                )
            )
    finally:
        hybrid_logger.propagate = original_propagate

    assert any(
        "divergence" in record.getMessage() and "cw-0" in record.getMessage()
        for record in caplog.records
    ), f"Expected divergence warning, got: {[r.getMessage() for r in caplog.records]}"
    assert scheduler._colocated_workers[0].free_pages == 100


def test_divergence_warning_suppressed_when_worker_reports_more_free_pages(
    monkeypatch, scheduler, caplog
) -> None:
    """The reverse direction — reported_free above the scheduler tally —
    is benign release lag (scheduler hasn't yet credited a completion the
    worker has already freed) and must NOT warn."""
    import logging

    _add_colocated_worker(scheduler, worker_id="cw-0", free_pages=50, page_size=16)

    hybrid_logger = logging.getLogger("sangam.engine.hybrid_scheduler")
    original_propagate = hybrid_logger.propagate
    hybrid_logger.propagate = True
    scheduler._last_divergence_warning_time.clear()
    try:
        with caplog.at_level(logging.WARNING, logger="sangam.engine.hybrid_scheduler"):
            # scheduler=50 < reported=100: worker freed locally but scheduler
            # hasn't applied the corresponding release yet. Benign — no
            # warning expected.
            scheduler._on_batch_metrics(
                sangam_pb2.BatchMetricsReport(
                    worker_id="cw-0",
                    worker_type="colocated",
                    batch_size=0,
                    prompt_len=0,
                    gen_len=0,
                    batch_start_time=99.0,
                    batch_end_time=99.1,
                    kv_total_pages=100,
                    kv_used_pages=0,
                    kv_free_pages=100,
                    num_unmasked_tokens=0,
                    batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    request_updates=[],
                )
            )
    finally:
        hybrid_logger.propagate = original_propagate

    assert not any("divergence" in record.getMessage() for record in caplog.records), (
        "Did not expect divergence warning when reported_free > scheduler, "
        f"got: {[r.getMessage() for r in caplog.records]}"
    )
    assert scheduler._colocated_workers[0].free_pages == 50


def _route_overflow_and_complete_prefill(
    scheduler: HybridScheduler,
    req,
    *,
    monkeypatch,
) -> None:
    """Route req as colocated overflow, then send a non-completing prefill report."""
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())
    scheduler._on_request_arrived(req)
    assert req.status is RequestStatus.PREFILLING
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=1,
            prompt_len=len(req.sequence_ids),
            gen_len=0,
            batch_start_time=100.0,
            batch_end_time=100.2,
            kv_total_pages=0,
            kv_used_pages=0,
            kv_free_pages=0,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    prefill_duration=0.2,
                    prefill_queue_wait_duration=0.0,
                )
            ],
        )
    )
    assert req.status is RequestStatus.DECODING


def test_mixed_report_from_colocated_applies_prefill_and_decode_updates(
    monkeypatch, scheduler
) -> None:
    """BATCH_PHASE_MIXED report: prefill update advances prefill_req to DECODING;
    decode update accumulates forward evals on decode_req."""
    _add_prefill_worker(scheduler, outstanding_prefill_tokens=4)  # overloaded
    _add_colocated_worker(scheduler)

    # decode_req: overflow-prefilled and now in DECODING
    decode_req = _make_request(request_id="decode-req")
    scheduler._requests[decode_req.request_id] = decode_req
    _route_overflow_and_complete_prefill(scheduler, decode_req, monkeypatch=monkeypatch)

    # prefill_req: currently being prefilled on the colocated worker
    prefill_req = _make_request(request_id="prefill-req")
    scheduler._requests[prefill_req.request_id] = prefill_req
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())
    scheduler._on_request_arrived(prefill_req)
    assert prefill_req.status is RequestStatus.PREFILLING

    # Mixed report: prefill update for prefill_req, decode update for decode_req
    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=2,
            prompt_len=len(prefill_req.sequence_ids),
            gen_len=len(decode_req.sequence_ids),
            batch_start_time=100.5,
            batch_end_time=100.8,
            kv_total_pages=0,
            kv_used_pages=0,
            kv_free_pages=0,
            num_unmasked_tokens=2,
            batch_phase=sangam_pb2.BATCH_PHASE_MIXED,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=prefill_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=prefill_req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    prefill_duration=0.3,
                    prefill_queue_wait_duration=0.0,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                ),
                sangam_pb2.BatchRequestUpdate(
                    request_id=decode_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=decode_req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=2,
                    decode_duration=0.3,
                    decode_queue_wait_duration=0.0,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                ),
            ],
        )
    )

    # Prefill update applied: prefill_req advances to DECODING
    assert prefill_req.status is RequestStatus.DECODING
    assert prefill_req.current_block.prefill_end_time is not None

    # Decode update applied: decode_req accumulates forward evals
    assert decode_req.current_block.decode_forward_evals_applied == 2


def test_mixed_report_prefill_outstanding_tokens_decremented_once(
    monkeypatch, scheduler
) -> None:
    """Prefill update in mixed report decrements outstanding tokens exactly once;
    the decode update for a different request does not trigger a second decrement."""
    _add_prefill_worker(scheduler, outstanding_prefill_tokens=4)  # overloaded
    _add_colocated_worker(scheduler)

    decode_req = _make_request(request_id="decode-req")
    scheduler._requests[decode_req.request_id] = decode_req
    _route_overflow_and_complete_prefill(scheduler, decode_req, monkeypatch=monkeypatch)

    prefill_req = _make_request(request_id="prefill-req")
    scheduler._requests[prefill_req.request_id] = prefill_req
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())
    scheduler._on_request_arrived(prefill_req)

    tokens_before = scheduler._colocated_workers[0].outstanding_prefill_tokens

    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=2,
            prompt_len=len(prefill_req.sequence_ids),
            gen_len=len(decode_req.sequence_ids),
            batch_start_time=100.5,
            batch_end_time=100.8,
            kv_total_pages=0,
            kv_used_pages=0,
            kv_free_pages=0,
            num_unmasked_tokens=2,
            batch_phase=sangam_pb2.BATCH_PHASE_MIXED,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=prefill_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=prefill_req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    prefill_duration=0.3,
                    prefill_queue_wait_duration=0.0,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                ),
                sangam_pb2.BatchRequestUpdate(
                    request_id=decode_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=decode_req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    decode_duration=0.3,
                    decode_queue_wait_duration=0.0,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                ),
            ],
        )
    )

    # Exactly one decrement: for the prefill update
    expected = max(0, tokens_before - prefill_req.request_accounting_tokens)
    assert scheduler._colocated_workers[0].outstanding_prefill_tokens == expected


def test_mixed_report_decode_completion_marks_request_completed(
    monkeypatch, scheduler
) -> None:
    """Decode update with block_completed=True inside a mixed report completes the request."""
    _add_prefill_worker(scheduler, outstanding_prefill_tokens=4)  # overloaded
    _add_colocated_worker(scheduler)

    # decode_req: overflow-prefilled and now in DECODING
    decode_req = _make_request(request_id="decode-req")
    scheduler._requests[decode_req.request_id] = decode_req
    _route_overflow_and_complete_prefill(scheduler, decode_req, monkeypatch=monkeypatch)

    # prefill_req: being prefilled in the same mixed batch
    prefill_req = _make_request(request_id="prefill-req")
    scheduler._requests[prefill_req.request_id] = prefill_req
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())
    scheduler._on_request_arrived(prefill_req)

    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=2,
            prompt_len=len(prefill_req.sequence_ids),
            gen_len=len(decode_req.sequence_ids),
            batch_start_time=101.0,
            batch_end_time=101.5,
            kv_total_pages=0,
            kv_used_pages=0,
            kv_free_pages=0,
            num_unmasked_tokens=32,
            batch_phase=sangam_pb2.BATCH_PHASE_MIXED,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=prefill_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=prefill_req.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    prefill_duration=0.5,
                    prefill_queue_wait_duration=0.0,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                ),
                sangam_pb2.BatchRequestUpdate(
                    request_id=decode_req.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=list(range(35)),
                    num_unmasked_tokens=31,
                    num_forward_evals_in_batch_phase=5,
                    decode_duration=0.5,
                    decode_queue_wait_duration=0.0,
                    block_completed=True,
                    request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                ),
            ],
        )
    )

    assert decode_req.status is RequestStatus.COMPLETED
    assert decode_req.complete_time is not None


def test_overflow_held_back_when_colocated_at_threshold(monkeypatch, scheduler) -> None:
    """A colocated worker at/over the overload threshold is not eligible for
    new overflow prefills, even with free memory. This keeps colocated prefill
    load from overtaking the prefill workers it overflows from: overflow is
    active only when every prefill worker is at/over the threshold, so an
    eligible colocated worker sits strictly below it. With the only colocated
    worker gated, the request holds back as pending rather than piling on."""
    _add_prefill_worker(
        scheduler, outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    _add_colocated_worker(
        scheduler,
        outstanding_prefill_tokens=4,  # at threshold -> gated
        free_pages=128,
        page_size=16,
    )

    mock_enqueue = MagicMock()
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", mock_enqueue)

    req = _make_request(request_id="overflowed")
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    mock_enqueue.assert_not_called()
    assert req.request_id in scheduler._pending_requests
    assert req.current_block.decode_worker_id is None


def test_overflow_disabled_queues_request_when_prefill_overloaded(monkeypatch) -> None:
    """With overflow disabled, an overloaded prefill state must queue the request
    even when colocated workers have capacity."""
    scheduler = HybridScheduler(_make_hybrid_config(enable_prefill_overflow=False))
    _add_prefill_worker(scheduler, outstanding_prefill_tokens=4)  # at threshold
    _add_colocated_worker(scheduler, outstanding_prefill_tokens=0)  # has capacity

    mock_enqueue = MagicMock()
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", mock_enqueue)

    req = _make_request(request_id="no-overflow")
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    mock_enqueue.assert_not_called()
    assert req.request_id in scheduler._pending_requests
    assert req.current_block.decode_worker_id is None


def test_drains_pending_after_colocated_prefill_completes(
    monkeypatch, scheduler
) -> None:
    """_drain_pending_requests is triggered after a colocated overflow prefill completes."""
    # Both workers start with no free pages so the picker returns None and
    # the request lands in _pending_requests. After the colocated prefill
    # completes, drain should re-enqueue it.
    _add_prefill_worker(
        scheduler, outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    _add_colocated_worker(
        scheduler, outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )

    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    drain_calls: list[None] = []
    original_drain = scheduler._drain_pending_requests

    def tracking_drain() -> None:
        drain_calls.append(None)
        original_drain()

    monkeypatch.setattr(scheduler, "_drain_pending_requests", tracking_drain)

    # First request arrives: both workers overloaded → queued
    req = _make_request(request_id="waiting")
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)
    assert req.request_id in scheduler._pending_requests

    drain_calls.clear()  # reset counter after initial routing

    # Simulate colocated worker completing a prior prefill (tokens drop below threshold)
    in_flight = _make_request(request_id="in-flight", prompt_token_ids=[1] * 3)
    scheduler._requests[in_flight.request_id] = in_flight
    in_flight.current_block.prefill_worker_id = "cw-0"
    in_flight.current_block.decode_worker_id = "cw-0"
    scheduler._colocated_workers[0].outstanding_prefill_tokens = 4

    scheduler._on_batch_metrics(
        sangam_pb2.BatchMetricsReport(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=1,
            prompt_len=len(in_flight.sequence_ids),
            gen_len=0,
            batch_start_time=100.0,
            batch_end_time=100.2,
            kv_total_pages=128,
            kv_used_pages=4,
            kv_free_pages=124,
            num_unmasked_tokens=1,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            request_updates=[
                sangam_pb2.BatchRequestUpdate(
                    request_id=in_flight.request_id,
                    block_index=0,
                    success=True,
                    updated_sequence=in_flight.sequence_ids,
                    num_unmasked_tokens=1,
                    num_forward_evals_in_batch_phase=1,
                    request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                    prefill_duration=0.2,
                    prefill_queue_wait_duration=0.0,
                )
            ],
        )
    )

    # _drain_pending_requests must have been called after the token decrement
    assert len(drain_calls) >= 1
    # Under FIFO head-of-line admission the drain attempts placement in order
    # and only dequeues on success. Both workers still report free_pages=0, so
    # the head request cannot be admitted and correctly stays pending.
    assert req.request_id in scheduler._pending_requests


def test_topology_guarded_memory_prefers_fast_pair_within_alpha() -> None:
    scheduler = HybridScheduler(
        _make_hybrid_config(
            decode_scheduler_policy="topology_guarded_memory",
            kv_fast_pairs="0-2",
        )
    )
    _add_colocated_worker(scheduler, "cw-fast", free_pages=8, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 2
    _add_colocated_worker(scheduler, "cw-mem", free_pages=10, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 3

    worker, reserved_pages = scheduler._pick_colocated_worker_for_decode(
        required_seq_length=1,
        prefill_gpu_id=0,
    )

    assert worker is not None
    assert worker.worker_id == "cw-fast"
    assert reserved_pages == 1


def test_topology_guarded_memory_falls_back_to_mem_best_below_alpha() -> None:
    scheduler = HybridScheduler(
        _make_hybrid_config(
            decode_scheduler_policy="topology_guarded_memory",
            kv_fast_pairs="0-2",
            kv_topology_alpha=0.9,
        )
    )
    _add_colocated_worker(scheduler, "cw-fast", free_pages=7, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 2
    _add_colocated_worker(scheduler, "cw-mem", free_pages=10, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 3

    worker, _ = scheduler._pick_colocated_worker_for_decode(
        required_seq_length=1,
        prefill_gpu_id=0,
    )

    assert worker is not None
    assert worker.worker_id == "cw-mem"


def test_topology_guarded_memory_chooses_best_fast_candidate() -> None:
    scheduler = HybridScheduler(
        _make_hybrid_config(
            decode_scheduler_policy="topology_guarded_memory",
            kv_fast_pairs="0-2,0-4",
        )
    )
    _add_colocated_worker(scheduler, "cw-fast-a", free_pages=8, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 2
    _add_colocated_worker(scheduler, "cw-fast-b", free_pages=9, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 4
    _add_colocated_worker(scheduler, "cw-slow", free_pages=10, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 5

    worker, _ = scheduler._pick_colocated_worker_for_decode(
        required_seq_length=1,
        prefill_gpu_id=0,
    )

    assert worker is not None
    assert worker.worker_id == "cw-fast-b"


def test_topology_guarded_memory_falls_back_when_no_fast_candidate() -> None:
    scheduler = HybridScheduler(
        _make_hybrid_config(
            decode_scheduler_policy="topology_guarded_memory",
            kv_fast_pairs="0-2",
        )
    )
    _add_colocated_worker(scheduler, "cw-a", free_pages=8, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 3
    _add_colocated_worker(scheduler, "cw-b", free_pages=10, page_size=1)
    scheduler._colocated_workers[-1].gpu_id = 4

    worker, _ = scheduler._pick_colocated_worker_for_decode(
        required_seq_length=1,
        prefill_gpu_id=0,
    )

    assert worker is not None
    assert worker.worker_id == "cw-b"


def test_balanced_length_clustering_prefers_closer_mean_within_slack_window() -> None:
    scheduler = HybridScheduler(
        _make_hybrid_config(
            decode_grouping_slack_ratio=0.40,
            decode_scheduler_policy="balanced_length_clustering",
        )
    )
    _add_colocated_worker(scheduler, "cw-0", free_pages=40, page_size=1)
    scheduler._colocated_workers[-1].outstanding_requests = 1
    scheduler._colocated_workers[-1].active_request_length_sum = 20
    _add_colocated_worker(scheduler, "cw-1", free_pages=40, page_size=1)
    scheduler._colocated_workers[-1].outstanding_requests = 1
    scheduler._colocated_workers[-1].active_request_length_sum = 34

    worker, _ = scheduler._pick_colocated_worker_for_decode(required_seq_length=32)
    assert worker is not None
    assert worker.worker_id == "cw-1"


def test_balanced_length_clustering_uses_random_choice_when_no_active_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = HybridScheduler(
        _make_hybrid_config(decode_scheduler_policy="balanced_length_clustering")
    )
    _add_colocated_worker(scheduler, "cw-0", free_pages=32, page_size=1)
    _add_colocated_worker(scheduler, "cw-1", free_pages=32, page_size=1)
    monkeypatch.setattr(hybrid_scheduler_module.random, "choice", lambda seq: seq[1])

    worker, _ = scheduler._pick_colocated_worker_for_decode(required_seq_length=16)
    assert worker is not None
    assert worker.worker_id == "cw-1"


def test_unified_overflow_picks_colocated_when_prefill_workers_loaded(
    monkeypatch,
) -> None:
    """Prefill workers at threshold and a colocated worker has the lowest
    outstanding_prefill_tokens — overflow should land on the colocated worker."""
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_scheduler_policy="least_outstanding_prefill_tokens")
    )
    _add_prefill_worker(
        scheduler, "pw-0", outstanding_prefill_tokens=4, free_pages=128, page_size=16
    )
    _add_colocated_worker(
        scheduler, "cw-0", outstanding_prefill_tokens=0, free_pages=128, page_size=16
    )
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    req = _make_request(request_id="overflow-1", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    assert req.current_block.prefill_worker_id == "cw-0"
    assert req.current_block.decode_worker_id == "cw-0"


def test_unified_overflow_load_balances_across_colocated_workers(
    monkeypatch,
) -> None:
    """When prefill is overloaded, overflow lands on the colocated worker
    with the lowest outstanding_prefill_tokens, not whichever has the most
    free memory. This is the primary fix for clumping."""
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_scheduler_policy="least_outstanding_prefill_tokens")
    )
    _add_prefill_worker(
        scheduler, "pw-0", outstanding_prefill_tokens=4, free_pages=128, page_size=16
    )
    # cw-0 has more free memory but is busier; cw-1 has less free memory
    # but lighter load. The legacy picker would have chosen cw-0 (max free
    # memory); the unified picker must choose cw-1 (min outstanding).
    _add_colocated_worker(
        scheduler, "cw-0", outstanding_prefill_tokens=3, free_pages=128, page_size=16
    )
    _add_colocated_worker(
        scheduler, "cw-1", outstanding_prefill_tokens=0, free_pages=64, page_size=16
    )
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    req = _make_request(request_id="balance", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    assert req.current_block.prefill_worker_id == "cw-1"
    assert req.current_block.decode_worker_id == "cw-1"


def test_overflow_prefers_below_threshold_colocated_over_at_threshold(
    monkeypatch,
) -> None:
    """Overflow routes to an under-threshold colocated worker and skips one at
    the threshold, even when the gated worker has more free memory and lower
    outstanding load. This bounds colocated prefill load below the threshold,
    so it cannot overtake the prefill workers overflow comes from."""
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_scheduler_policy="least_outstanding_prefill_tokens")
    )
    _add_prefill_worker(
        scheduler, "pw-0", outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    # cw-0 is at the threshold (gated) despite abundant memory; cw-1 is below
    # it. Overflow must pick cw-1, keeping every colocated worker < threshold.
    _add_colocated_worker(
        scheduler, "cw-0", outstanding_prefill_tokens=4, free_pages=128, page_size=16
    )
    _add_colocated_worker(
        scheduler, "cw-1", outstanding_prefill_tokens=3, free_pages=128, page_size=16
    )
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    req = _make_request(request_id="bounded", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    # cw-1 (below threshold) is chosen; cw-0 (at threshold) is skipped despite
    # more free memory. The gate is an admission check on load before dispatch.
    assert req.current_block.prefill_worker_id == "cw-1"
    assert req.current_block.decode_worker_id == "cw-1"
    # cw-0 was never charged: gated out before reservation.
    assert scheduler._colocated_workers[0].outstanding_prefill_tokens == 4
    assert scheduler._colocated_workers[0].free_pages == 128


def test_unified_overflow_queues_when_no_worker_has_free_pages(monkeypatch) -> None:
    """When neither prefill nor colocated workers have free KV pages for the
    request, fall back to the central pending queue. Outstanding-token load is
    no longer a gate on its own."""
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_scheduler_policy="least_outstanding_prefill_tokens")
    )
    _add_prefill_worker(
        scheduler, "pw-0", outstanding_prefill_tokens=0, free_pages=0, page_size=16
    )
    _add_colocated_worker(
        scheduler, "cw-0", outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    mock_enqueue = MagicMock()
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", mock_enqueue)

    req = _make_request(request_id="gated", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    mock_enqueue.assert_not_called()
    assert req.request_id in scheduler._pending_requests
    assert req.current_block.prefill_worker_id is None
    assert req.current_block.decode_worker_id is None


def test_pending_admission_is_head_of_line(monkeypatch, scheduler) -> None:
    """Strict FIFO HOL: a newer, smaller request must not leapfrog an older
    request that cannot be admitted. With only enough memory for the small
    request, both stay pending because the older (head) request is blocked."""
    # No dedicated prefill worker -> always overflow to colocated. One colocated
    # worker with 4 free pages: enough for the small request (4 pages) but not
    # the large one (8 pages).
    _add_colocated_worker(scheduler, "cw-0", free_pages=4, page_size=16)
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    # Large head request: 96 prompt + 32 gen = 128 tokens -> ceil(128/16) = 8 pages.
    big = _make_request(request_id="big", prompt_token_ids=[1] * 96, submit_time=100.0)
    # Small later request: 32 prompt + 32 gen = 64 tokens -> ceil(64/16) = 4 pages.
    small = _make_request(
        request_id="small", prompt_token_ids=[1] * 32, submit_time=101.0
    )
    scheduler._requests[big.request_id] = big
    scheduler._requests[small.request_id] = small

    scheduler._on_request_arrived(big)
    scheduler._on_request_arrived(small)

    # Head (big) cannot be placed; small must not jump ahead. Both pending.
    assert big.request_id in scheduler._pending_requests
    assert small.request_id in scheduler._pending_requests
    assert big.current_block.prefill_worker_id is None
    assert small.current_block.prefill_worker_id is None


def test_hol_releases_in_order_when_capacity_frees(monkeypatch) -> None:
    """Once enough capacity frees up, the blocked head is admitted first, then
    the request behind it, in submit-time order."""
    # High overload threshold so admitting `big` does not gate `small` on the
    # overflow overload check — this test isolates head-of-line ordering.
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_overload_threshold=1_000_000)
    )
    _add_colocated_worker(scheduler, "cw-0", free_pages=4, page_size=16)
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    big = _make_request(request_id="big", prompt_token_ids=[1] * 96, submit_time=100.0)
    small = _make_request(
        request_id="small", prompt_token_ids=[1] * 32, submit_time=101.0
    )
    scheduler._requests[big.request_id] = big
    scheduler._requests[small.request_id] = small
    scheduler._on_request_arrived(big)
    scheduler._on_request_arrived(small)
    assert big.request_id in scheduler._pending_requests
    assert small.request_id in scheduler._pending_requests

    # Free enough memory for both (8 pages for big + 4 for small = 12) and drain.
    scheduler._colocated_workers[0].free_pages = 12
    scheduler._drain_pending_requests()

    # Both admitted, neither left pending.
    assert big.request_id not in scheduler._pending_requests
    assert small.request_id not in scheduler._pending_requests
    assert big.current_block.prefill_worker_id == "cw-0"
    assert small.current_block.prefill_worker_id == "cw-0"


def test_continuation_block_head_sorts_over_newer_pending(monkeypatch) -> None:
    """A continuation block of an in-flight (older submit_time) request is
    admitted ahead of a newer pending request when capacity is tight, because
    it routes through the same FIFO queue and head-sorts by submit_time."""
    # High overload threshold so the held-back `newer` is gated purely on
    # memory (the stated reason), not the overflow overload check.
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_overload_threshold=1_000_000)
    )
    # Start with no free memory so the newer request blocks on arrival. Sizing
    # is by len(sequence_ids) = prompt_len + gen_length (constant across blocks).
    _add_colocated_worker(scheduler, "cw-0", free_pages=0, page_size=16)
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    # Newer request: 32 prompt + 32 gen = 64 tokens -> 4 pages. Pending (no mem).
    newer = _make_request(
        request_id="newer", prompt_token_ids=[1] * 32, submit_time=101.0
    )
    scheduler._requests[newer.request_id] = newer
    scheduler._on_request_arrived(newer)
    assert newer.request_id in scheduler._pending_requests

    # Older multi-block request whose first block just completed; its
    # continuation needs 32 + 64 = 96 tokens -> 6 pages. submit_time is older,
    # so it head-sorts ahead of `newer`.
    older = _make_request(
        request_id="older",
        prompt_token_ids=[1] * 32,
        gen_length=64,
        block_length=32,
        submit_time=100.0,
    )
    scheduler._requests[older.request_id] = older
    older.current_block_index = 1
    older.status = RequestStatus.WAITING_NEXT_BLOCK
    scheduler._queue_pending_request(older, now=102.0)

    # Capacity frees up enough for the older continuation (6 pages) but, once it
    # claims that, nothing remains for the newer request (4 pages).
    scheduler._colocated_workers[0].free_pages = 6
    scheduler._drain_pending_requests()

    # The older continuation block claims the scarce capacity first.
    assert older.request_id not in scheduler._pending_requests
    assert older.current_block.prefill_worker_id == "cw-0"
    # The newer request remains pending behind it.
    assert newer.request_id in scheduler._pending_requests


def _enqueue_backlog_decode_ready(
    scheduler: HybridScheduler, *, request_id: str, prompt_len: int
) -> Request:
    """Add a completed-prefill request to the decode-ready backlog.

    These are requests whose KV is pinned on a prefill worker, waiting for
    colocated memory to land on for decode. They must out-rank new overflow
    prefills for that memory.
    """
    backlog = _make_request(request_id=request_id, prompt_token_ids=[1] * prompt_len)
    backlog.status = RequestStatus.WAITING_DECODE
    scheduler._requests[backlog.request_id] = backlog
    scheduler._enqueue_decode_ready_request(backlog, now=0.0)
    return backlog


def test_overflow_held_back_when_decode_ready_backlog_needs_memory(
    monkeypatch, scheduler
) -> None:
    """A new overflow prefill is held back when the decode-ready backlog's
    reserved headroom would leave too few colocated pages for it.

    Without the backlog the same request would be admitted (8 free >= 4
    needed), so the backlog reservation is what forces the hold-back."""
    _add_prefill_worker(
        scheduler, outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    _add_colocated_worker(scheduler, "cw-0", free_pages=8, page_size=16)
    # Backlog request: 48 prompt + 32 gen = 80 tokens -> ceil(80/16) = 5 pages.
    _enqueue_backlog_decode_ready(scheduler, request_id="backlog", prompt_len=48)
    mock_enqueue = MagicMock()
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", mock_enqueue)

    # Overflow request: 32 prompt + 32 gen = 64 tokens -> ceil(64/16) = 4 pages.
    # available(8) - reserved(5) = 3 < 4 -> held back.
    req = _make_request(request_id="overflow", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    mock_enqueue.assert_not_called()
    assert req.request_id in scheduler._pending_requests
    assert req.current_block.decode_worker_id is None
    # The backlog request must not have had its pages stolen.
    assert scheduler._colocated_workers[0].free_pages == 8


def test_overflow_admitted_when_headroom_beyond_backlog(monkeypatch, scheduler) -> None:
    """With the same backlog but enough spare colocated memory, the overflow
    prefill is admitted against the headroom that exceeds the reservation."""
    _add_prefill_worker(
        scheduler, outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    _add_colocated_worker(scheduler, "cw-0", free_pages=16, page_size=16)
    # Same 5-page backlog as the held-back test.
    _enqueue_backlog_decode_ready(scheduler, request_id="backlog", prompt_len=48)
    mock_enqueue = MagicMock()
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", mock_enqueue)

    # available(16) - reserved(5) = 11 >= 4 -> admitted.
    req = _make_request(request_id="overflow", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    mock_enqueue.assert_called_once()
    assert req.request_id not in scheduler._pending_requests
    assert req.current_block.prefill_worker_id == "cw-0"
    assert req.current_block.decode_worker_id == "cw-0"
    assert scheduler._colocated_workers[0].free_pages == 12


def test_overflow_never_dispatches_to_prefill_worker(monkeypatch) -> None:
    """Overflow is colocated-only: an overloaded prefill worker with abundant
    free memory is never chosen, even though a memory-greedy picker might
    prefer it. Overflow lands on the under-threshold colocated worker. The
    former unified picker would have considered the prefill worker here."""
    scheduler = HybridScheduler(
        _make_hybrid_config(prefill_scheduler_policy="least_outstanding_prefill_tokens")
    )
    # pw-0 is at threshold (overloaded) with the most free memory; the old union
    # picker would have considered it. cw-0 is below threshold (eligible).
    _add_prefill_worker(
        scheduler, "pw-0", outstanding_prefill_tokens=4, free_pages=128, page_size=16
    )
    _add_colocated_worker(
        scheduler, "cw-0", outstanding_prefill_tokens=0, free_pages=64, page_size=16
    )
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    req = _make_request(request_id="overflow", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)

    assert req.current_block.prefill_worker_id == "cw-0"
    assert req.current_block.decode_worker_id == "cw-0"


def test_decode_reservation_release_drains_pending(monkeypatch, scheduler) -> None:
    """Releasing a decode reservation (e.g. a decode block completes) retries
    held-back overflow prefills, not just the decode-ready queue."""
    _add_prefill_worker(
        scheduler, outstanding_prefill_tokens=4, free_pages=0, page_size=16
    )
    _add_colocated_worker(scheduler, "cw-0", free_pages=0, page_size=16)
    monkeypatch.setattr(scheduler, "_submit_enqueue_async", MagicMock())

    # No free colocated pages -> overflow request is held back as pending.
    req = _make_request(request_id="held-back", prompt_token_ids=[1] * 32)
    scheduler._requests[req.request_id] = req
    scheduler._on_request_arrived(req)
    assert req.request_id in scheduler._pending_requests

    drain_calls: list[None] = []
    original_drain = scheduler._drain_pending_requests

    def tracking_drain() -> None:
        drain_calls.append(None)
        original_drain()

    monkeypatch.setattr(scheduler, "_drain_pending_requests", tracking_drain)

    # An in-flight decode on cw-0 completes and releases its pages.
    in_flight = _make_request(request_id="in-flight", prompt_token_ids=[1] * 32)
    scheduler._requests[in_flight.request_id] = in_flight
    block = in_flight.current_block
    block.decode_worker_id = "cw-0"
    block.reserved_pages = 4
    scheduler._release_decode_reservation(block)

    # The decode release must retry the pending queue (not just decode-ready).
    # Drain re-enqueues the request as an event; asserting it left the pending
    # dict is race-free, whereas free_pages can be mutated by the live event
    # loop redispatching the drained request.
    assert len(drain_calls) >= 1
    assert req.request_id not in scheduler._pending_requests
