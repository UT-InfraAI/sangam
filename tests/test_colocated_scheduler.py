"""Tests for the ColocatedScheduler: worker registration, round-robin, event loop.

All tests mock gRPC stubs — no GPU or network required.
"""

import time
from unittest.mock import MagicMock

import pytest

import sangam.engine.colocated_scheduler as colocated_scheduler_module
from sangam.request import Request, RequestStatus
from sangam.engine.colocated_scheduler import (
    ColocatedScheduler,
    ColocatedSchedulerServicer,
    WorkerInfo,
)
from sangam.engine.scheduler_config import ColocatedSchedulerConfig
from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.proto import sangam_pb2
from sangam.sampling_parameters import SamplingParameters
from sangam.types import WorkerType


def _make_colocated_config(**overrides) -> ColocatedSchedulerConfig:
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
        colocated_sticky_worker=False,
        max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
    )
    defaults.update(overrides)
    return ColocatedSchedulerConfig(**defaults)


def _make_request(**kwargs):
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


def _mock_time_with_tail(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    timeline = iter(values)
    last = values[-1]

    def _next_time() -> float:
        nonlocal last
        try:
            last = next(timeline)
        except StopIteration:
            return last
        return last

    monkeypatch.setattr(time, "time", _next_time)


def _make_prefill_batch_report(
    req: Request,
    *,
    success: bool = True,
    block_index: int = 0,
    updated_sequence: list[int] | None = None,
    prefill_duration: float = 0.1,
    prefill_queue_wait_duration: float = 0.01,
    num_unmasked_tokens: int = 0,
    block_completed: bool = False,
):
    return sangam_pb2.BatchMetricsReport(
        worker_id="cw-0",
        worker_type="colocated",
        batch_size=1,
        prompt_len=len(req.sequence_ids),
        gen_len=0,
        batch_start_time=100.0,
        batch_end_time=100.0 + prefill_duration,
        kv_total_pages=128,
        kv_used_pages=16,
        kv_free_pages=112,
        num_unmasked_tokens=num_unmasked_tokens,
        batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
        request_updates=[
            sangam_pb2.BatchRequestUpdate(
                request_id=req.request_id,
                block_index=block_index,
                success=success,
                updated_sequence=updated_sequence
                if updated_sequence is not None
                else list(req.sequence_ids),
                num_unmasked_tokens=num_unmasked_tokens,
                num_forward_evals_in_batch_phase=1 if success else 0,
                request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
                prefill_duration=prefill_duration if success else 0.0,
                prefill_queue_wait_duration=prefill_queue_wait_duration
                if success
                else 0.0,
                block_completed=block_completed if success else False,
            )
        ],
    )


def _make_decode_batch_report(
    req: Request,
    *,
    success: bool = True,
    block_index: int = 0,
    updated_sequence: list[int] | None = None,
    num_forward_evals: int = 1,
    decode_duration: float = 0.5,
    decode_queue_wait_duration: float = 0.02,
    num_unmasked_tokens: int = 0,
    block_completed: bool = False,
    prefill_duration: float = 0.2,
    prefill_queue_wait_duration: float = 0.01,
):
    return sangam_pb2.BatchMetricsReport(
        worker_id="cw-0",
        worker_type="colocated",
        batch_size=1,
        prompt_len=0,
        gen_len=req.block_states[block_index].block_end
        - req.block_states[block_index].block_start,
        batch_start_time=200.0,
        batch_end_time=200.0 + decode_duration,
        kv_total_pages=128,
        kv_used_pages=16,
        kv_free_pages=112,
        num_unmasked_tokens=num_unmasked_tokens,
        batch_phase=sangam_pb2.BATCH_PHASE_DECODE,
        request_updates=[
            sangam_pb2.BatchRequestUpdate(
                request_id=req.request_id,
                block_index=block_index,
                success=success,
                updated_sequence=updated_sequence
                if updated_sequence is not None
                else list(req.sequence_ids),
                num_unmasked_tokens=num_unmasked_tokens,
                num_forward_evals_in_batch_phase=num_forward_evals if success else 0,
                request_phase=sangam_pb2.BATCH_PHASE_DECODE,
                decode_duration=decode_duration if success and block_completed else 0.0,
                decode_queue_wait_duration=decode_queue_wait_duration
                if success
                else 0.0,
                prefill_duration=prefill_duration,
                prefill_queue_wait_duration=prefill_queue_wait_duration,
                block_completed=block_completed,
            )
        ],
    )


def _make_mixed_batch_report(
    prefill_req: Request,
    decode_req: Request,
    *,
    prefill_sequence: list[int],
    decode_sequence: list[int],
    prefill_num_unmasked_tokens: int = 1,
    decode_num_unmasked_tokens: int = 1,
    prefill_duration: float = 0.2,
    prefill_queue_wait_duration: float = 0.01,
    decode_duration: float = 0.5,
    decode_queue_wait_duration: float = 0.02,
    decode_num_forward_evals: int = 2,
    decode_block_completed: bool = False,
):
    return sangam_pb2.BatchMetricsReport(
        worker_id="cw-0",
        worker_type="colocated",
        batch_size=2,
        prompt_len=len(prefill_req.sequence_ids),
        gen_len=decode_req.block_states[0].block_end
        - decode_req.block_states[0].block_start,
        batch_start_time=300.0,
        batch_end_time=300.0 + max(prefill_duration, decode_duration),
        kv_total_pages=128,
        kv_used_pages=16,
        kv_free_pages=112,
        num_unmasked_tokens=prefill_num_unmasked_tokens + decode_num_unmasked_tokens,
        batch_phase=sangam_pb2.BATCH_PHASE_MIXED,
        request_updates=[
            sangam_pb2.BatchRequestUpdate(
                request_id=prefill_req.request_id,
                block_index=0,
                success=True,
                updated_sequence=prefill_sequence,
                num_unmasked_tokens=prefill_num_unmasked_tokens,
                num_forward_evals_in_batch_phase=1,
                prefill_duration=prefill_duration,
                prefill_queue_wait_duration=prefill_queue_wait_duration,
                block_completed=False,
                request_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            ),
            sangam_pb2.BatchRequestUpdate(
                request_id=decode_req.request_id,
                block_index=0,
                success=True,
                updated_sequence=decode_sequence,
                num_unmasked_tokens=decode_num_unmasked_tokens,
                num_forward_evals_in_batch_phase=decode_num_forward_evals,
                prefill_duration=decode_req.block_states[0].prefill_duration,
                prefill_queue_wait_duration=decode_req.block_states[
                    0
                ].prefill_queue_wait_duration,
                decode_duration=decode_duration if decode_block_completed else 0.0,
                decode_queue_wait_duration=decode_queue_wait_duration,
                block_completed=decode_block_completed,
                request_phase=sangam_pb2.BATCH_PHASE_DECODE,
            ),
        ],
    )


@pytest.fixture
def scheduler():
    return ColocatedScheduler(_make_colocated_config())


@pytest.fixture
def balancing_scheduler():
    return ColocatedScheduler(
        _make_colocated_config(
            prefill_scheduler_policy="least_outstanding_prefill_tokens"
        )
    )


@pytest.fixture
def request_balancing_scheduler():
    return ColocatedScheduler(
        _make_colocated_config(prefill_scheduler_policy="least_outstanding_requests")
    )


@pytest.fixture
def sticky_request_balancing_scheduler():
    return ColocatedScheduler(
        _make_colocated_config(
            prefill_scheduler_policy="least_outstanding_requests",
            colocated_sticky_worker=True,
        )
    )


@pytest.fixture
def length_sum_scheduler():
    return ColocatedScheduler(
        _make_colocated_config(prefill_scheduler_policy="least_request_length_sum")
    )


@pytest.fixture
def sticky_length_sum_scheduler():
    return ColocatedScheduler(
        _make_colocated_config(
            prefill_scheduler_policy="least_request_length_sum",
            colocated_sticky_worker=True,
        )
    )


@pytest.fixture
def balanced_length_clustering_scheduler():
    return ColocatedScheduler(
        _make_colocated_config(prefill_scheduler_policy="balanced_length_clustering")
    )


def _add_worker(scheduler, worker_id="cw-0", dist_rank=0):
    """Add a mock colocated worker to the scheduler."""
    w = WorkerInfo(
        worker_id,
        WorkerType.COLOCATED,
        f"addr:{dist_rank}",
        dist_rank=dist_rank,
        gpu_id=dist_rank,
    )
    scheduler._workers.append(w)
    stub = MagicMock()
    stub.EnqueueRequest.return_value = sangam_pb2.EnqueueColocatedResponse(success=True)
    scheduler._worker_stubs[worker_id] = stub
    return stub


class TestWorkerRegistration:
    def test_register_colocated_worker(self, scheduler):
        ok = scheduler.register_worker(
            "cw-0", "colocated", "localhost:20100", dist_rank=0, gpu_id=0
        )
        assert ok
        assert len(scheduler._workers) == 1
        assert scheduler._workers[0].worker_id == "cw-0"
        assert "cw-0" in scheduler._worker_stubs

    def test_register_prefill_worker_rejected(self, scheduler):
        ok = scheduler.register_worker(
            "pw-0", "prefill", "localhost:20100", dist_rank=0, gpu_id=0
        )
        assert not ok
        assert len(scheduler._workers) == 0

    def test_register_unknown_type_fails(self, scheduler):
        ok = scheduler.register_worker(
            "x-0", "unknown", "localhost:20100", dist_rank=0, gpu_id=0
        )
        assert not ok


class TestRoundRobin:
    def test_round_robin(self, scheduler):
        for i in range(3):
            _add_worker(scheduler, f"cw-{i}", dist_rank=i)
        req = _make_request()
        picks = [scheduler._pick_worker(req).worker_id for _ in range(6)]
        assert picks == ["cw-0", "cw-1", "cw-2", "cw-0", "cw-1", "cw-2"]

    def test_balancing_picks_least_loaded_worker(self, balancing_scheduler):
        balancing_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_prefill_tokens=50,
                ),
                WorkerInfo(
                    "cw-1",
                    WorkerType.COLOCATED,
                    "addr:1",
                    dist_rank=1,
                    gpu_id=1,
                    outstanding_prefill_tokens=5,
                ),
            ]
        )
        assert balancing_scheduler._pick_worker(_make_request()).worker_id == "cw-1"

    def test_balancing_uses_random_tiebreak(self, balancing_scheduler, monkeypatch):
        chosen = WorkerInfo(
            "cw-1",
            WorkerType.COLOCATED,
            "addr:1",
            dist_rank=1,
            gpu_id=1,
            outstanding_prefill_tokens=5,
        )
        balancing_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_prefill_tokens=5,
                ),
                chosen,
            ]
        )
        monkeypatch.setattr(
            colocated_scheduler_module.random, "choice", lambda seq: seq[1]
        )
        assert balancing_scheduler._pick_worker(_make_request()) is chosen

    def test_request_balancing_picks_worker_with_fewer_requests(
        self, request_balancing_scheduler
    ):
        request_balancing_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_requests=2,
                ),
                WorkerInfo(
                    "cw-1",
                    WorkerType.COLOCATED,
                    "addr:1",
                    dist_rank=1,
                    gpu_id=1,
                    outstanding_requests=1,
                ),
            ]
        )

        assert (
            request_balancing_scheduler._pick_worker(_make_request()).worker_id
            == "cw-1"
        )

    def test_request_balancing_uses_random_tiebreak(
        self, request_balancing_scheduler, monkeypatch
    ):
        chosen = WorkerInfo(
            "cw-1",
            WorkerType.COLOCATED,
            "addr:1",
            dist_rank=1,
            gpu_id=1,
            outstanding_requests=1,
        )
        request_balancing_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_requests=1,
                ),
                chosen,
            ]
        )
        monkeypatch.setattr(
            colocated_scheduler_module.random, "choice", lambda seq: seq[1]
        )

        assert request_balancing_scheduler._pick_worker(_make_request()) is chosen

    def test_length_sum_policy_picks_worker_with_lowest_projected_sum(
        self, length_sum_scheduler
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        length_sum_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    active_request_length_sum=100,
                ),
                WorkerInfo(
                    "cw-1",
                    WorkerType.COLOCATED,
                    "addr:1",
                    dist_rank=1,
                    gpu_id=1,
                    active_request_length_sum=10,
                ),
            ]
        )
        assert length_sum_scheduler._pick_worker(req).worker_id == "cw-1"

    def test_length_sum_policy_uses_random_tiebreak(
        self, length_sum_scheduler, monkeypatch
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        chosen = WorkerInfo(
            "cw-1",
            WorkerType.COLOCATED,
            "addr:1",
            dist_rank=1,
            gpu_id=1,
            active_request_length_sum=10,
        )
        length_sum_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    active_request_length_sum=10,
                ),
                chosen,
            ]
        )
        monkeypatch.setattr(
            colocated_scheduler_module.random, "choice", lambda seq: seq[1]
        )
        assert length_sum_scheduler._pick_worker(req) is chosen

    def test_balanced_length_clustering_prefers_closer_mean_within_slack_window(
        self, balanced_length_clustering_scheduler
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        balanced_length_clustering_scheduler._decode_grouping_slack_ratio = 0.40
        balanced_length_clustering_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_requests=4,
                    active_request_length_sum=160,
                ),
                WorkerInfo(
                    "cw-1",
                    WorkerType.COLOCATED,
                    "addr:1",
                    dist_rank=1,
                    gpu_id=1,
                    outstanding_requests=2,
                    active_request_length_sum=80,
                ),
            ]
        )

        assert (
            balanced_length_clustering_scheduler._pick_worker(req).worker_id == "cw-1"
        )

    def test_balanced_length_clustering_excludes_workers_outside_slack_window(
        self, balanced_length_clustering_scheduler
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        balanced_length_clustering_scheduler._decode_grouping_slack_ratio = 0.10
        balanced_length_clustering_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_requests=1,
                    active_request_length_sum=100,
                ),
                WorkerInfo(
                    "cw-1",
                    WorkerType.COLOCATED,
                    "addr:1",
                    dist_rank=1,
                    gpu_id=1,
                    outstanding_requests=2,
                    active_request_length_sum=115,
                ),
            ]
        )

        assert (
            balanced_length_clustering_scheduler._pick_worker(req).worker_id == "cw-0"
        )

    def test_balanced_length_clustering_zero_slack_matches_length_sum_behavior(
        self, monkeypatch
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        scheduler = ColocatedScheduler(
            _make_colocated_config(
                prefill_scheduler_policy="balanced_length_clustering",
                decode_grouping_slack_ratio=0.0,
            )
        )
        chosen = WorkerInfo(
            "cw-1",
            WorkerType.COLOCATED,
            "addr:1",
            dist_rank=1,
            gpu_id=1,
            active_request_length_sum=10,
        )
        scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    active_request_length_sum=10,
                ),
                chosen,
            ]
        )
        monkeypatch.setattr(
            colocated_scheduler_module.random, "choice", lambda seq: seq[1]
        )

        assert scheduler._pick_worker(req) is chosen

    def test_balanced_length_clustering_uses_random_choice_when_no_active_requests(
        self, balanced_length_clustering_scheduler, monkeypatch
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        chosen = WorkerInfo(
            "cw-1",
            WorkerType.COLOCATED,
            "addr:1",
            dist_rank=1,
            gpu_id=1,
            outstanding_requests=0,
            active_request_length_sum=0,
        )
        balanced_length_clustering_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_requests=0,
                    active_request_length_sum=0,
                ),
                chosen,
            ]
        )
        monkeypatch.setattr(
            colocated_scheduler_module.random, "choice", lambda seq: seq[1]
        )

        assert balanced_length_clustering_scheduler._pick_worker(req) is chosen

    def test_balanced_length_clustering_uses_random_tiebreak_for_equal_scores(
        self, balanced_length_clustering_scheduler, monkeypatch
    ):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        balanced_length_clustering_scheduler._decode_grouping_slack_ratio = 0.20
        chosen = WorkerInfo(
            "cw-1",
            WorkerType.COLOCATED,
            "addr:1",
            dist_rank=1,
            gpu_id=1,
            outstanding_requests=1,
            active_request_length_sum=40,
        )
        balanced_length_clustering_scheduler._workers.extend(
            [
                WorkerInfo(
                    "cw-0",
                    WorkerType.COLOCATED,
                    "addr:0",
                    dist_rank=0,
                    gpu_id=0,
                    outstanding_requests=1,
                    active_request_length_sum=30,
                ),
                chosen,
            ]
        )
        monkeypatch.setattr(
            colocated_scheduler_module.random, "choice", lambda seq: seq[1]
        )

        assert balanced_length_clustering_scheduler._pick_worker(req) is chosen


class TestSubmitAndPoll:
    def test_colocated_scheduler_ignores_kv_transfer_reports(
        self, scheduler, monkeypatch
    ):
        req = _make_request()
        req.status = RequestStatus.DECODING
        scheduler._requests[req.request_id] = req
        warning = MagicMock()
        monkeypatch.setattr(colocated_scheduler_module.logger, "warning", warning)

        scheduler._on_kv_transfer(
            sangam_pb2.KVTransferReport(
                worker_id="cw-0",
                request_id=req.request_id,
                block_index=0,
                success=True,
                transfer_start_time=10.0,
                transfer_end_time=11.0,
            )
        )

        assert req.status == RequestStatus.DECODING
        assert req.error_message is None
        assert req.block_states[0].kv_transfer_start_time is None
        assert req.block_states[0].kv_transfer_end_time is None
        warning.assert_called_once()

    def test_submit_stores_request(self, scheduler):
        req = _make_request()
        rid = scheduler.submit(req)
        assert rid == req.request_id
        assert scheduler.poll(rid) is req

    def test_poll_unknown_returns_none(self, scheduler):
        assert scheduler.poll("nonexistent") is None

    def test_submit_omitted_fixed_unmask_quota_stays_none(self, scheduler):
        servicer = ColocatedSchedulerServicer(scheduler)
        resp = servicer.Submit(
            sangam_pb2.GenerateRequest(
                prompt_token_ids=[1, 2, 3],
                gen_length=32,
                sampling_parameters={"unmasking_strategy": "random"},
            ),
            context=None,
        )

        req = scheduler.poll(resp.request_id)
        assert req is not None
        assert req.sampling_parameters.fixed_unmask_quota is None

    def test_reset_metrics_rpc_resets_store(self, scheduler):
        servicer = ColocatedSchedulerServicer(scheduler)
        scheduler._metrics_store.reset = MagicMock()

        resp = servicer.ResetMetrics(
            sangam_pb2.ResetMetricsRequest(),
            context=None,
        )

        assert resp.success is True
        scheduler._metrics_store.reset.assert_called_once()


class TestEventLoop:
    def test_length_sum_policy_sticky_counts_once_until_terminal_completion(
        self, sticky_length_sum_scheduler
    ):
        _add_worker(sticky_length_sum_scheduler)
        req = _make_request(gen_length=64, block_length=32)
        sticky_length_sum_scheduler._requests[req.request_id] = req

        sticky_length_sum_scheduler._on_request_arrived(req)
        assert sticky_length_sum_scheduler._workers[0].active_request_length_sum == 67

        sticky_length_sum_scheduler._on_batch_metrics(_make_prefill_batch_report(req))
        sticky_length_sum_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert sticky_length_sum_scheduler._workers[0].active_request_length_sum == 67

        sticky_length_sum_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req, block_index=1)
        )
        sticky_length_sum_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                block_index=1,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert req.status == RequestStatus.COMPLETED
        assert sticky_length_sum_scheduler._workers[0].active_request_length_sum == 0

    def test_length_sum_policy_nonsticky_updates_per_block(self, length_sum_scheduler):
        _add_worker(length_sum_scheduler)
        req = _make_request(gen_length=64, block_length=32)
        length_sum_scheduler._requests[req.request_id] = req

        length_sum_scheduler._on_request_arrived(req)
        assert length_sum_scheduler._workers[0].active_request_length_sum == 67

        length_sum_scheduler._on_batch_metrics(_make_prefill_batch_report(req))
        length_sum_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )
        assert length_sum_scheduler._workers[0].active_request_length_sum == 67

        length_sum_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req, block_index=1)
        )
        length_sum_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                block_index=1,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )
        assert req.status == RequestStatus.COMPLETED
        assert length_sum_scheduler._workers[0].active_request_length_sum == 0

    def test_length_sum_policy_nonsticky_decrements_on_prefill_failure(
        self, length_sum_scheduler
    ):
        _add_worker(length_sum_scheduler)
        req = _make_request()
        length_sum_scheduler._requests[req.request_id] = req
        length_sum_scheduler._on_request_arrived(req)
        assert length_sum_scheduler._workers[0].active_request_length_sum == 35

        length_sum_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req, success=False)
        )

        assert req.status == RequestStatus.ERROR
        assert length_sum_scheduler._workers[0].active_request_length_sum == 0

    def test_target_blocks_stops_request_before_buffer_end(self, scheduler):
        """With max_gen_len > requested gen_len, the request must finish after
        target_blocks blocks even though block_states has more entries."""
        _add_worker(scheduler)
        req = _make_request(gen_length=128, block_length=32, target_blocks=2)
        scheduler._requests[req.request_id] = req
        assert len(req.block_states) == 4
        assert req.target_blocks == 2

        scheduler._on_request_arrived(req)

        for block_index in range(2):
            scheduler._on_batch_metrics(
                _make_prefill_batch_report(req, block_index=block_index)
            )
            scheduler._on_batch_metrics(
                _make_decode_batch_report(
                    req,
                    block_index=block_index,
                    updated_sequence=list(range(len(req.sequence_ids))),
                    num_forward_evals=3,
                    decode_duration=0.5,
                    block_completed=True,
                )
            )

        assert req.status == RequestStatus.COMPLETED
        assert req.current_block_index == 2

    def test_request_arrived_without_workers_queues(self, scheduler):
        req = _make_request()
        scheduler._requests[req.request_id] = req

        scheduler._on_request_arrived(req)

        assert req.status == RequestStatus.PENDING
        assert req.request_id in scheduler._pending_requests

    def test_request_arrived_records_worker_registration_stall(
        self, scheduler, monkeypatch
    ):
        req = _make_request()
        scheduler._requests[req.request_id] = req
        _mock_time_with_tail(monkeypatch, [10.0, 11.0, 11.1])

        scheduler._on_request_arrived(req)
        _add_worker(scheduler)
        scheduler._on_request_arrived(req)

        assert req.current_block is not None
        assert req.current_block.scheduler_wait_duration == pytest.approx(1.0)
        assert req.current_block.prefill_scheduler_wait_duration == pytest.approx(1.0)
        assert req.current_block.decode_scheduler_wait_duration == pytest.approx(0.0)

    def test_request_arrived_enqueues_on_worker(self, scheduler):
        stub = _add_worker(scheduler)
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)

        assert req.status == RequestStatus.PREFILLING
        scheduler._drain_outbound()
        stub.EnqueueRequest.assert_called_once()

        # Verify arrival_time is set
        call_args = stub.EnqueueRequest.call_args
        enqueue_req = call_args[0][0]
        assert enqueue_req.arrival_time == req.submit_time
        assert enqueue_req.total_generation_blocks == len(req.block_states)
        assert scheduler._workers[0].outstanding_prefill_tokens == 35

    def test_prefill_batch_keeps_outstanding_tokens_until_block_completion(
        self, scheduler
    ):
        _add_worker(scheduler)
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)

        scheduler._on_batch_metrics(_make_prefill_batch_report(req))

        assert req.status == RequestStatus.DECODING
        assert scheduler._workers[0].outstanding_prefill_tokens == 35

    def test_prefill_batch_applies_prefill_result_to_request_state(self, scheduler):
        _add_worker(scheduler)
        scheduler._metrics_store.on_block_prefill_end = MagicMock()
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)

        sampled_sequence = list(req.sequence_ids)
        sampled_sequence[3] = 77

        scheduler._on_batch_metrics(
            _make_prefill_batch_report(
                req,
                updated_sequence=sampled_sequence,
                prefill_duration=0.1,
                prefill_queue_wait_duration=0.01,
                num_unmasked_tokens=1,
            )
        )

        block = req.block_states[0]
        assert req.status == RequestStatus.DECODING
        assert req.sequence_ids == sampled_sequence
        assert req.num_forward_evals == 1
        assert block.prefill_start_time == pytest.approx(
            block.prefill_enqueue_time + 0.01
        )
        assert block.prefill_end_time == pytest.approx(block.prefill_start_time + 0.1)
        scheduler._metrics_store.on_block_prefill_end.assert_called_once_with(
            req.request_id, 0, 0.1, block.prefill_worker_id
        )

    def test_decode_batch_single_block_completes_request(self, scheduler):
        _add_worker(scheduler)
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        scheduler._on_batch_metrics(_make_prefill_batch_report(req))
        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(35)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert req.status == RequestStatus.COMPLETED
        assert req.complete_time is not None
        assert req.num_forward_evals == 6
        assert scheduler._workers[0].outstanding_prefill_tokens == 0

    def test_prefill_completed_block_skips_decode_and_completes_request(
        self, scheduler
    ):
        stub = _add_worker(scheduler)
        scheduler._metrics_store.on_block_decode_end = MagicMock()
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)

        scheduler._on_batch_metrics(
            _make_prefill_batch_report(
                req,
                updated_sequence=list(range(35)),
                num_unmasked_tokens=32,
                block_completed=True,
            )
        )

        assert req.status == RequestStatus.COMPLETED
        assert req.num_forward_evals == 1
        assert stub.EnqueueRequest.call_count == 1
        assert scheduler._workers[0].outstanding_prefill_tokens == 0
        scheduler._metrics_store.on_block_decode_end.assert_called_once_with(
            req.request_id, 0, 0.0
        )

    def test_decode_batch_records_prefill_time_and_zero_kv_transfer(self, scheduler):
        _add_worker(scheduler)
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        scheduler._on_batch_metrics(
            _make_prefill_batch_report(
                req,
                prefill_duration=0.2,
                prefill_queue_wait_duration=0.01,
            )
        )
        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(35)),
                num_forward_evals=5,
                decode_duration=0.5,
                decode_queue_wait_duration=0.02,
                prefill_duration=0.2,
                prefill_queue_wait_duration=0.01,
                block_completed=True,
            )
        )

        assert req.total_prefill_time == pytest.approx(0.2)
        assert req.total_kv_transfer_time == 0.0

    def test_decode_batch_records_request_token_gaps(self, scheduler):
        _add_worker(scheduler)
        scheduler._metrics_store.on_request_visibility = MagicMock()
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        prefill_sequence = list(req.sequence_ids)
        prefill_sequence[3] = 77
        scheduler._on_batch_metrics(
            _make_prefill_batch_report(
                req,
                updated_sequence=prefill_sequence,
                prefill_duration=0.2,
                prefill_queue_wait_duration=0.01,
                num_unmasked_tokens=1,
            )
        )

        updated_sequence = list(range(35))
        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=updated_sequence,
                num_forward_evals=5,
                decode_duration=0.5,
                decode_queue_wait_duration=0.02,
                num_unmasked_tokens=2,
                prefill_duration=0.2,
                prefill_queue_wait_duration=0.01,
                block_completed=True,
            )
        )

        assert scheduler._metrics_store.on_request_visibility.call_count >= 1

    def test_mixed_batch_applies_prefill_and_decode_updates(self, scheduler):
        _add_worker(scheduler)

        prefill_req = _make_request()
        decode_req = _make_request()

        scheduler._requests[prefill_req.request_id] = prefill_req
        scheduler._requests[decode_req.request_id] = decode_req

        scheduler._on_request_arrived(prefill_req)
        scheduler._on_request_arrived(decode_req)
        scheduler._on_batch_metrics(_make_prefill_batch_report(decode_req))

        mixed_prefill_sequence = list(prefill_req.sequence_ids)
        mixed_prefill_sequence[3] = 77
        mixed_decode_sequence = list(range(35))

        scheduler._on_batch_metrics(
            _make_mixed_batch_report(
                prefill_req,
                decode_req,
                prefill_sequence=mixed_prefill_sequence,
                decode_sequence=mixed_decode_sequence,
                decode_num_forward_evals=2,
                decode_block_completed=False,
            )
        )

        assert prefill_req.status == RequestStatus.DECODING
        assert prefill_req.sequence_ids == mixed_prefill_sequence
        assert prefill_req.block_states[0].prefill_end_time is not None
        assert decode_req.status == RequestStatus.DECODING
        assert decode_req.sequence_ids == mixed_decode_sequence
        assert decode_req.num_forward_evals == 3

    def test_decode_batch_multi_block_continues(self, scheduler):
        stub = _add_worker(scheduler)
        req = _make_request(gen_length=64, block_length=32)
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        scheduler._on_batch_metrics(_make_prefill_batch_report(req))

        # Complete block 0
        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        # Should have enqueued request for block 1
        assert req.current_block_index == 1
        assert req.status == RequestStatus.PREFILLING
        scheduler._drain_outbound()
        assert stub.EnqueueRequest.call_count == 2
        assert scheduler._workers[0].outstanding_prefill_tokens == 67

        scheduler._on_batch_metrics(_make_prefill_batch_report(req, block_index=1))

        # Complete block 1
        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                block_index=1,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert req.status == RequestStatus.COMPLETED

    def test_block_failure_sets_error(self, scheduler):
        _add_worker(scheduler)
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)

        scheduler._on_batch_metrics(_make_prefill_batch_report(req, success=False))

        assert req.status == RequestStatus.ERROR
        assert scheduler._workers[0].outstanding_prefill_tokens == 0

    def test_decode_failure_after_prefill_does_not_redecrement_outstanding_tokens(
        self, scheduler
    ):
        _add_worker(scheduler)
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        scheduler._on_batch_metrics(_make_prefill_batch_report(req))

        assert scheduler._workers[0].outstanding_prefill_tokens == 35

        scheduler._on_batch_metrics(_make_decode_batch_report(req, success=False))

        assert req.status == RequestStatus.ERROR
        assert scheduler._workers[0].outstanding_prefill_tokens == 0

    def test_round_robin_across_blocks(self, scheduler):
        """Multi-block request should round-robin blocks across workers."""
        stub0 = _add_worker(scheduler, "cw-0", dist_rank=0)
        stub1 = _add_worker(scheduler, "cw-1", dist_rank=1)

        req = _make_request(gen_length=64, block_length=32)
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        scheduler._on_batch_metrics(_make_prefill_batch_report(req))

        # Block 0 assigned to cw-0
        assert req.block_states[0].decode_worker_id == "cw-0"
        scheduler._drain_outbound()
        stub0.EnqueueRequest.assert_called_once()

        # Complete block 0
        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        # Block 1 assigned to cw-1
        assert req.block_states[1].decode_worker_id == "cw-1"
        scheduler._drain_outbound()
        stub1.EnqueueRequest.assert_called_once()

    def test_multi_block_request_releases_outstanding_tokens_on_block_completion(
        self, scheduler
    ):
        stub = _add_worker(scheduler)
        req = _make_request(gen_length=64, block_length=32)
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)

        assert scheduler._workers[0].outstanding_prefill_tokens == 67
        scheduler._on_batch_metrics(_make_prefill_batch_report(req))
        assert scheduler._workers[0].outstanding_prefill_tokens == 67

        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert scheduler._workers[0].outstanding_prefill_tokens == 67
        scheduler._drain_outbound()
        assert stub.EnqueueRequest.call_count == 2
        assert scheduler._workers[0].outstanding_prefill_tokens == 67

    def test_request_policy_nonsticky_counts_assigned_block_per_worker(
        self, request_balancing_scheduler
    ):
        stub0 = _add_worker(request_balancing_scheduler, "cw-0", dist_rank=0)
        stub1 = _add_worker(request_balancing_scheduler, "cw-1", dist_rank=1)

        req = _make_request(gen_length=64, block_length=32)
        request_balancing_scheduler._requests[req.request_id] = req
        request_balancing_scheduler._on_request_arrived(req)

        first_block_worker_id = req.block_states[0].prefill_worker_id
        assert first_block_worker_id in {"cw-0", "cw-1"}
        outstanding_after_first = {
            worker.worker_id: worker.outstanding_requests
            for worker in request_balancing_scheduler._workers
        }
        assert outstanding_after_first[first_block_worker_id] == 1
        assert sum(outstanding_after_first.values()) == 1

        request_balancing_scheduler._on_batch_metrics(_make_prefill_batch_report(req))
        request_balancing_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        second_block_worker_id = req.block_states[1].prefill_worker_id
        assert second_block_worker_id in {"cw-0", "cw-1"}
        outstanding_after_second = {
            worker.worker_id: worker.outstanding_requests
            for worker in request_balancing_scheduler._workers
        }
        assert outstanding_after_second[second_block_worker_id] == 1
        assert sum(outstanding_after_second.values()) == 1
        request_balancing_scheduler._drain_outbound()
        assert (stub0.EnqueueRequest.call_count + stub1.EnqueueRequest.call_count) == 2

    def test_request_policy_nonsticky_decrements_on_prefill_failure(
        self, request_balancing_scheduler
    ):
        _add_worker(request_balancing_scheduler)
        req = _make_request()
        request_balancing_scheduler._requests[req.request_id] = req
        request_balancing_scheduler._on_request_arrived(req)

        assert request_balancing_scheduler._workers[0].outstanding_requests == 1

        request_balancing_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req, success=False)
        )

        assert req.status == RequestStatus.ERROR
        assert request_balancing_scheduler._workers[0].outstanding_requests == 0

    def test_request_policy_nonsticky_decrements_on_decode_failure(
        self, request_balancing_scheduler
    ):
        _add_worker(request_balancing_scheduler)
        req = _make_request()
        request_balancing_scheduler._requests[req.request_id] = req
        request_balancing_scheduler._on_request_arrived(req)
        request_balancing_scheduler._on_batch_metrics(_make_prefill_batch_report(req))

        assert request_balancing_scheduler._workers[0].outstanding_requests == 1

        request_balancing_scheduler._on_batch_metrics(
            _make_decode_batch_report(req, success=False)
        )

        assert req.status == RequestStatus.ERROR
        assert request_balancing_scheduler._workers[0].outstanding_requests == 0

    def test_request_policy_sticky_counts_request_once_until_terminal_completion(
        self, sticky_request_balancing_scheduler
    ):
        stub0 = _add_worker(sticky_request_balancing_scheduler, "cw-0", dist_rank=0)
        stub1 = _add_worker(sticky_request_balancing_scheduler, "cw-1", dist_rank=1)

        req = _make_request(gen_length=64, block_length=32)
        sticky_request_balancing_scheduler._requests[req.request_id] = req
        sticky_request_balancing_scheduler._on_request_arrived(req)

        first_block_worker_id = req.block_states[0].prefill_worker_id
        assert first_block_worker_id in {"cw-0", "cw-1"}
        outstanding_after_first = {
            worker.worker_id: worker.outstanding_requests
            for worker in sticky_request_balancing_scheduler._workers
        }
        assert outstanding_after_first[first_block_worker_id] == 1
        assert sum(outstanding_after_first.values()) == 1

        sticky_request_balancing_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req)
        )
        sticky_request_balancing_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert req.block_states[1].prefill_worker_id == first_block_worker_id
        outstanding_after_second = {
            worker.worker_id: worker.outstanding_requests
            for worker in sticky_request_balancing_scheduler._workers
        }
        assert outstanding_after_second[first_block_worker_id] == 1
        assert sum(outstanding_after_second.values()) == 1

        sticky_request_balancing_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req, block_index=1)
        )
        sticky_request_balancing_scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                block_index=1,
                updated_sequence=list(range(67)),
                num_forward_evals=5,
                decode_duration=0.5,
                block_completed=True,
            )
        )

        assert req.status == RequestStatus.COMPLETED
        assert sticky_request_balancing_scheduler._workers[0].outstanding_requests == 0
        assert sticky_request_balancing_scheduler._workers[1].outstanding_requests == 0
        sticky_request_balancing_scheduler._drain_outbound()
        if first_block_worker_id == "cw-0":
            assert stub0.EnqueueRequest.call_count == 2
            assert stub1.EnqueueRequest.call_count == 0
        else:
            assert stub0.EnqueueRequest.call_count == 0
            assert stub1.EnqueueRequest.call_count == 2

    def test_request_policy_sticky_decrements_once_on_decode_failure(
        self, sticky_request_balancing_scheduler
    ):
        _add_worker(sticky_request_balancing_scheduler)
        req = _make_request()
        sticky_request_balancing_scheduler._requests[req.request_id] = req
        sticky_request_balancing_scheduler._on_request_arrived(req)
        sticky_request_balancing_scheduler._on_batch_metrics(
            _make_prefill_batch_report(req)
        )

        assert sticky_request_balancing_scheduler._workers[0].outstanding_requests == 1

        sticky_request_balancing_scheduler._on_batch_metrics(
            _make_decode_batch_report(req, success=False)
        )

        assert req.status == RequestStatus.ERROR
        assert sticky_request_balancing_scheduler._workers[0].outstanding_requests == 0

    def test_register_worker_drains_pending_requests(self, scheduler):
        req = _make_request()
        scheduler._requests[req.request_id] = req
        scheduler._on_request_arrived(req)
        assert req.request_id in scheduler._pending_requests

        scheduler._event_queue.put = MagicMock()
        scheduler._drain_pending_requests()
        scheduler._event_queue.put.assert_not_called()

        _add_worker(scheduler, "cw-0", dist_rank=0)
        scheduler._drain_pending_requests()
        scheduler._event_queue.put.assert_called_once()
        assert req.request_id not in scheduler._pending_requests

    def test_pending_requests_drain_in_submit_time_order_after_retry(self, scheduler):
        worker = WorkerInfo(
            "cw-0",
            WorkerType.COLOCATED,
            "addr:0",
            dist_rank=0,
            gpu_id=0,
            max_pages=8,
            page_size=16,
            free_pages=0,
        )
        scheduler._workers.append(worker)
        scheduler._worker_stubs["cw-0"] = MagicMock()

        first_req = _make_request(
            request_id="req-first",
            prompt_token_ids=[1, 2, 3],
            gen_length=32,
            block_length=32,
            submit_time=100.0,
        )
        second_req = _make_request(
            request_id="req-second",
            prompt_token_ids=[1],
            gen_length=32,
            block_length=32,
            submit_time=101.0,
        )
        scheduler._requests[first_req.request_id] = first_req
        scheduler._requests[second_req.request_id] = second_req

        # Force the scheduler's retry path by making it temporarily not ready,
        # then retrying the older request.
        scheduler._workers.clear()
        scheduler._on_request_arrived(first_req)
        scheduler._on_request_arrived(second_req)
        scheduler._on_request_arrived(first_req)
        scheduler._workers.append(worker)

        scheduler._event_queue.put = MagicMock()
        scheduler._drain_pending_requests()

        queued_request_ids = [
            call.args[0].payload.request_id
            for call in scheduler._event_queue.put.call_args_list
        ]
        assert queued_request_ids == [first_req.request_id, second_req.request_id]

    def test_batch_metrics_report_updates_store(self, scheduler):
        scheduler._metrics_store.on_batch_end = MagicMock()
        scheduler._on_batch_metrics(
            sangam_pb2.BatchMetricsReport(
                worker_id="cw-0",
                worker_type="colocated",
                batch_size=2,
                prompt_len=32,
                gen_len=64,
                batch_start_time=10.0,
                batch_end_time=10.2,
                kv_total_pages=256,
                kv_used_pages=100,
                kv_free_pages=156,
                num_unmasked_tokens=13,
            )
        )
        scheduler._metrics_store.on_batch_end.assert_called_once_with(
            worker_id="cw-0",
            worker_type="colocated",
            batch_size=2,
            prompt_len=32,
            gen_len=64,
            batch_start_time=10.0,
            batch_end_time=10.2,
            kv_total_pages=256,
            kv_used_pages=100,
            kv_free_pages=156,
            num_unmasked_tokens=13,
            batch_phase=sangam_pb2.BATCH_PHASE_PREFILL,
            sampling_duration=0.0,
            request_updates=[],
            batch_op_attn_time=None,
            batch_op_mlp_time=None,
            batch_op_qkv_time=None,
        )

    def test_batch_metrics_report_updates_request_visibility_store(self, scheduler):
        scheduler._metrics_store.on_request_visibility = MagicMock()
        req = _make_request(prompt_token_ids=[1, 2], gen_length=4, block_length=4)
        scheduler._requests[req.request_id] = req

        scheduler._on_batch_metrics(
            _make_decode_batch_report(
                req,
                num_forward_evals=1,
                num_unmasked_tokens=2,
                block_completed=False,
            )
        )

        scheduler._metrics_store.on_request_visibility.assert_called_once_with(
            request_id=req.request_id,
            timestamp=200.5,
            num_unmasked_tokens=2,
        )

    def test_worker_state_snapshot_in_batch_report_updates_store(self, scheduler):
        req = _make_request()
        scheduler._metrics_store.on_worker_state = MagicMock()
        snapshot = sangam_pb2.WorkerStateSnapshot(
            state=sangam_pb2.WORKER_STATE_QUEUED,
            timestamp=100.0,
            waiting_queue_depth=1,
            active_batch_size=0,
            kv_total_pages=128,
            kv_used_pages=16,
            kv_free_pages=112,
        )
        report = _make_prefill_batch_report(req)
        report2 = sangam_pb2.BatchMetricsReport()
        report2.CopyFrom(report)
        report2.worker_state_after.CopyFrom(snapshot)
        scheduler._on_batch_metrics(report2)
        assert scheduler._metrics_store.on_worker_state.called
