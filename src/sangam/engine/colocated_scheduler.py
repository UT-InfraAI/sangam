"""Colocated baseline scheduler: round-robin distribution to colocated workers.

Simplified scheduler for the colocated baseline where each worker does both
prefill and decode. The scheduler only needs to:
  1. Distribute incoming requests round-robin to colocated workers.
  2. Reconcile prefill/decode progress from worker batch reports.
"""

import random
import time

import grpc

from sangam.engine.base_scheduler import (
    BaseScheduler,
    BaseSchedulerServicer,
    WorkerInfo,
    serve_scheduler_instance,
)
from sangam.engine.scheduler_config import ColocatedSchedulerConfig
from sangam.request import Request, RequestStatus
from sangam.logger import init_logger
from sangam.metrics.constants import WorkerStateTimeline
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.types import PrefillSchedulerPolicy, WorkerType

logger = init_logger(__name__)


class ColocatedScheduler(BaseScheduler):
    """Round-robin scheduler for colocated workers."""

    def __init__(self, config: ColocatedSchedulerConfig):
        super().__init__(config, event_thread_name="colocated-scheduler-loop")

        # Worker registry
        self._workers: list[WorkerInfo] = []
        self._worker_rr = 0
        self._prefill_scheduler_policy = PrefillSchedulerPolicy(
            config.prefill_scheduler_policy
        )
        self._decode_grouping_slack_ratio = config.decode_grouping_slack_ratio

        # Sticky worker mode: pin each request to its first-assigned worker
        self._sticky_worker: bool = config.colocated_sticky_worker
        # Maps request_id -> worker_id; populated on first block, cleared on completion/error
        self._sticky_worker_map: dict[str, str] = {}

        # gRPC stubs cached per worker
        self._worker_stubs: dict[str, sangam_pb2_grpc.ColocatedWorkerServiceStub] = {}

    # ----- Worker management -----

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
        if is_conversion:
            logger.error("ColocatedScheduler does not support worker conversion")
            return False
        try:
            wtype = WorkerType(worker_type)
        except ValueError:
            logger.error(f"Unknown worker type: {worker_type}")
            return False

        if wtype is not WorkerType.COLOCATED:
            logger.error(
                f"ColocatedScheduler only accepts colocated workers, got {wtype}"
            )
            return False

        info = WorkerInfo(
            worker_id=worker_id,
            worker_type=wtype,
            address=address,
            dist_rank=dist_rank,
            gpu_id=gpu_id,
            max_pages=max_pages,
            page_size=page_size,
            free_pages=max_pages,
        )
        channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", self._config.max_grpc_message_length),
                (
                    "grpc.max_receive_message_length",
                    self._config.max_grpc_message_length,
                ),
            ],
        )
        self._workers.append(info)
        self._worker_stubs[worker_id] = sangam_pb2_grpc.ColocatedWorkerServiceStub(
            channel,
        )
        logger.info(
            f"Registered colocated worker {worker_id} at {address} (rank={dist_rank})"
        )
        self._metrics_store.on_worker_state(
            worker_id=worker_id,
            worker_type=worker_type,
            state=WorkerStateTimeline.IDLE,
            timestamp=time.time(),
            waiting_queue_depth=0,
            active_batch_size=0,
            kv_total_pages=max_pages or 0,
            kv_used_pages=0,
            kv_free_pages=max_pages or 0,
        )
        self._drain_pending_requests()
        return True

    def _pick_worker_round_robin(self) -> WorkerInfo:
        workers = self._workers
        worker = workers[self._worker_rr % len(workers)]
        self._worker_rr += 1
        return worker

    def _pick_worker_least_outstanding(self) -> WorkerInfo:
        if (
            self._prefill_scheduler_policy
            is PrefillSchedulerPolicy.LEAST_OUTSTANDING_REQUESTS
        ):
            min_outstanding = min(w.outstanding_requests for w in self._workers)
            candidates = [
                w for w in self._workers if w.outstanding_requests == min_outstanding
            ]
            return random.choice(candidates)

        min_outstanding = min(w.outstanding_prefill_tokens for w in self._workers)
        candidates = [
            w for w in self._workers if w.outstanding_prefill_tokens == min_outstanding
        ]
        return random.choice(candidates)

    def _pick_worker_least_request_length_sum(self, req: Request) -> WorkerInfo:
        request_length = req.request_accounting_tokens
        min_projected = min(
            worker.active_request_length_sum + request_length
            for worker in self._workers
        )
        candidates = [
            worker
            for worker in self._workers
            if worker.active_request_length_sum + request_length == min_projected
        ]
        return random.choice(candidates)

    def _pick_worker_balanced_length_clustering(self, req: Request) -> WorkerInfo:
        if self._decode_grouping_slack_ratio == 0:
            return self._pick_worker_least_request_length_sum(req)

        request_length = req.request_accounting_tokens
        projected_sums = {
            worker.worker_id: worker.active_request_length_sum + request_length
            for worker in self._workers
        }
        min_projected = min(projected_sums.values())
        projected_limit = min_projected * (1 + self._decode_grouping_slack_ratio)
        candidates = [
            worker
            for worker in self._workers
            if projected_sums[worker.worker_id] <= projected_limit
        ]
        if not candidates:
            return self._pick_worker_least_request_length_sum(req)

        scored_candidates = [
            (
                abs(
                    request_length
                    - (worker.active_request_length_sum / worker.outstanding_requests)
                ),
                worker,
            )
            for worker in candidates
            if worker.outstanding_requests > 0
        ]
        if not scored_candidates:
            return random.choice(candidates)

        min_distance = min(score for score, _ in scored_candidates)
        best_candidates = [
            worker for score, worker in scored_candidates if score == min_distance
        ]
        return random.choice(best_candidates)

    def _pick_worker(self, req: Request) -> WorkerInfo:
        if self._prefill_scheduler_policy is PrefillSchedulerPolicy.ROUND_ROBIN:
            return self._pick_worker_round_robin()
        if (
            self._prefill_scheduler_policy
            is PrefillSchedulerPolicy.LEAST_REQUEST_LENGTH_SUM
        ):
            return self._pick_worker_least_request_length_sum(req)
        if (
            self._prefill_scheduler_policy
            is PrefillSchedulerPolicy.BALANCED_LENGTH_CLUSTERING
        ):
            return self._pick_worker_balanced_length_clustering(req)
        return self._pick_worker_least_outstanding()

    def _increment_prefill_outstanding_tokens(
        self,
        worker_id: str | None,
        num_tokens: int,
    ) -> None:
        if worker_id is None:
            return
        for worker in self._workers:
            if worker.worker_id == worker_id:
                worker.outstanding_prefill_tokens += num_tokens
                self._metrics_store.on_outstanding_prefill_tokens(
                    worker.worker_id,
                    worker.outstanding_prefill_tokens,
                    time.time(),
                )
                return

    def _decrement_prefill_outstanding_tokens(
        self,
        worker_id: str | None,
        num_tokens: int,
    ) -> None:
        if worker_id is None:
            return
        for worker in self._workers:
            if worker.worker_id == worker_id:
                worker.outstanding_prefill_tokens = max(
                    0, worker.outstanding_prefill_tokens - num_tokens
                )
                self._metrics_store.on_outstanding_prefill_tokens(
                    worker.worker_id,
                    worker.outstanding_prefill_tokens,
                    time.time(),
                )
                return

    def _increment_outstanding_requests(self, worker_id: str | None) -> None:
        if worker_id is None:
            return
        for worker in self._workers:
            if worker.worker_id == worker_id:
                worker.outstanding_requests += 1
                return

    def _decrement_outstanding_requests(self, worker_id: str | None) -> None:
        if worker_id is None:
            return
        for worker in self._workers:
            if worker.worker_id == worker_id:
                worker.outstanding_requests = max(0, worker.outstanding_requests - 1)
                return

    def _increment_active_request_length_sum(
        self, worker_id: str | None, request_length: int
    ) -> None:
        if worker_id is None:
            return
        for worker in self._workers:
            if worker.worker_id == worker_id:
                worker.active_request_length_sum += request_length
                return

    def _decrement_active_request_length_sum(
        self, worker_id: str | None, request_length: int
    ) -> None:
        if worker_id is None:
            return
        for worker in self._workers:
            if worker.worker_id == worker_id:
                worker.active_request_length_sum = max(
                    0, worker.active_request_length_sum - request_length
                )
                return

    def _release_sticky_worker(self, req: Request) -> None:
        """Remove sticky pin and release the full-request upfront accounting.

        Idempotent: safe to call more than once on the same request (the second
        call is a no-op because pop returns None on a missing key).
        """
        worker_id = self._sticky_worker_map.pop(req.request_id, None)
        if worker_id is None:
            return
        self._decrement_outstanding_requests(worker_id)
        num_blocks = req.target_blocks
        self._decrement_prefill_outstanding_tokens(
            worker_id, num_blocks * req.request_accounting_tokens
        )
        self._decrement_active_request_length_sum(
            worker_id, req.request_accounting_tokens
        )

    def _release_nonsticky_block(self, req: Request, block) -> None:
        """Release non-sticky outstanding tokens for a single terminal block."""
        worker_id = block.prefill_worker_id
        if worker_id is None:
            return
        block.prefill_worker_id = None
        self._decrement_outstanding_requests(worker_id)
        self._decrement_prefill_outstanding_tokens(
            worker_id, req.request_accounting_tokens
        )
        self._decrement_active_request_length_sum(
            worker_id, req.request_accounting_tokens
        )

    def _is_ready_for_requests(self) -> bool:
        return len(self._workers) > 0

    def _on_request_arrived(self, req: Request) -> None:
        """Pick a worker and enqueue the current block."""
        request_id = req.request_id
        block = req.current_block
        if block is None:
            return
        now = time.time()
        if not self._is_ready_for_requests():
            self._mark_scheduler_wait(block, now, "prefill")
            if request_id not in self._pending_requests:
                logger.warning(
                    f"[{request_id}] No colocated workers registered yet; "
                    "request queued until workers are ready"
                )
            self._queue_pending_request(req, now=now)
            return
        self._remove_pending_request(request_id, now=now)
        self._clear_scheduler_wait(request_id, block, now)

        if self._sticky_worker and request_id in self._sticky_worker_map:
            # Continuation block: reuse the pinned worker, no accounting change.
            pinned_id = self._sticky_worker_map[request_id]
            worker = next((w for w in self._workers if w.worker_id == pinned_id), None)
            if worker is None:
                logger.error(
                    "[%s] Sticky worker %s is no longer registered; "
                    "falling back to _pick_worker()",
                    request_id,
                    pinned_id,
                )
                worker = self._pick_worker(req)
                self._sticky_worker_map[request_id] = worker.worker_id
            # No outstanding_prefill_tokens adjustment — charged upfront on first block.
        else:
            worker = self._pick_worker(req)
            if self._sticky_worker:
                # First block: pin the worker and charge the full request upfront.
                self._sticky_worker_map[request_id] = worker.worker_id
                self._increment_outstanding_requests(worker.worker_id)
                self._increment_prefill_outstanding_tokens(
                    worker.worker_id,
                    req.target_blocks * req.request_accounting_tokens,
                )
                self._increment_active_request_length_sum(
                    worker.worker_id, req.request_accounting_tokens
                )
            else:
                self._increment_outstanding_requests(worker.worker_id)
                self._increment_prefill_outstanding_tokens(
                    worker.worker_id, req.request_accounting_tokens
                )
                self._increment_active_request_length_sum(
                    worker.worker_id, req.request_accounting_tokens
                )
        block.prefill_worker_id = worker.worker_id
        block.decode_worker_id = worker.worker_id

        req.status = RequestStatus.PREFILLING
        block.prefill_enqueue_time = time.time()

        logger.debug(
            f"[{request_id}] Block {block.block_index}: assigned to {worker.worker_id}"
        )

        stub = self._worker_stubs[worker.worker_id]

        enqueue_req = sangam_pb2.EnqueueColocatedRequest(
            request_id=request_id,
            sequence_ids=req.sequence_ids,
            block_start=block.block_start,
            block_end=block.block_end,
            block_index=block.block_index,
            total_generation_blocks=req.target_blocks,
            request_seed=req.request_seed,
            mask_id=req.mask_id,
            arrival_time=req.submit_time,
            prefill_enqueue_time=block.prefill_enqueue_time,
        )
        enqueue_req.sampling_parameters.CopyFrom(req.sampling_parameters.to_proto())

        self._submit_enqueue_async(
            stub.EnqueueRequest,
            enqueue_req,
            worker.worker_id,
            WorkerType.COLOCATED.value,
        )

    def _apply_request_visibility(
        self, request_id: str, timestamp: float, num_unmasked_tokens: int
    ) -> None:
        self._metrics_store.on_request_visibility(
            request_id=request_id,
            timestamp=timestamp,
            num_unmasked_tokens=num_unmasked_tokens,
        )

    def _complete_block(
        self,
        req: Request,
        block,
        *,
        completion_time: float,
        decode_duration: float,
    ) -> None:
        if block.decode_start_time is None:
            block.decode_start_time = completion_time - decode_duration
        block.decode_end_time = completion_time
        block.completed = True
        self._metrics_store.on_block_decode_end(
            req.request_id, block.block_index, decode_duration
        )
        block_total_time = (block.decode_end_time or time.time()) - (
            block.prefill_start_time or 0.0
        )
        self._metrics_store.on_block_end(
            req.request_id, block.block_index, block_total_time, block.decode_worker_id
        )

        logger.debug(
            "[%s] Block %s: complete, %s fwd evals, %.3fs decode",
            req.request_id,
            block.block_index,
            block.decode_forward_evals_applied,
            decode_duration,
        )

        if not self._sticky_worker:
            self._release_nonsticky_block(req, block)

        req.current_block_index += 1
        if req.current_block_index < req.target_blocks:
            req.status = RequestStatus.WAITING_NEXT_BLOCK
            self._on_request_arrived(req)
        else:
            if self._sticky_worker:
                self._release_sticky_worker(req)
            req.status = RequestStatus.COMPLETED
            req.complete_time = completion_time
            req.done_event.set()
            self._metrics_store.on_request_end(req)
            logger.debug(
                "[%s] Completed: %s fwd evals, %.2fs e2e, %.2fs accounted, %.2fs unaccounted "
                "(prefill=%.2fs, decode=%.2fs, scheduling=%.2fs, kv_nonoverlapped=%.2fs)",
                req.request_id,
                req.num_forward_evals,
                req.complete_time - req.submit_time,
                req.verification_component_sum,
                req.unaccounted_time,
                req.total_prefill_time,
                req.total_decode_time,
                req.total_queue_wait_time,
                req.total_kv_transfer_time_nonoverlapped,
            )

    def _apply_prefill_batch_update(
        self,
        report: sangam_pb2.BatchMetricsReport,
        update: sangam_pb2.BatchRequestUpdate,
    ) -> None:
        request_id = update.request_id
        req = self._requests.get(request_id)
        if req is None:
            logger.error("[%s] Prefill batch update but request not found", request_id)
            return

        block = req.block_states[update.block_index]

        if not update.success:
            if self._sticky_worker:
                self._release_sticky_worker(req)
            else:
                self._release_nonsticky_block(req, block)
            req.status = RequestStatus.ERROR
            req.error_message = f"Prefill failed for block {update.block_index}"
            req.done_event.set()
            return

        block.prefill_queue_wait_duration = update.prefill_queue_wait_duration
        if block.prefill_enqueue_time is not None:
            block.prefill_start_time = (
                block.prefill_enqueue_time + update.prefill_queue_wait_duration
            )
            block.prefill_end_time = block.prefill_start_time + update.prefill_duration
        else:
            block.prefill_end_time = report.batch_end_time
        req.sequence_ids = list(update.updated_sequence)
        block.prefill_forward_evals_applied = max(
            block.prefill_forward_evals_applied,
            update.num_forward_evals_in_batch_phase,
        )
        req.recompute_num_forward_evals()
        self._apply_request_visibility(
            request_id=request_id,
            timestamp=report.batch_end_time,
            num_unmasked_tokens=update.num_unmasked_tokens,
        )
        self._metrics_store.on_block_prefill_end(
            request_id,
            block.block_index,
            update.prefill_duration,
            report.worker_id,
        )
        if update.block_completed:
            req.status = RequestStatus.DECODING
            block.decode_start_time = block.prefill_end_time or report.batch_end_time
            self._complete_block(
                req,
                block,
                completion_time=block.decode_start_time,
                decode_duration=0.0,
            )
            return
        req.status = RequestStatus.DECODING

    def _apply_decode_batch_update(
        self,
        report: sangam_pb2.BatchMetricsReport,
        update: sangam_pb2.BatchRequestUpdate,
    ) -> None:
        request_id = update.request_id
        req = self._requests.get(request_id)
        if req is None:
            logger.error("[%s] Decode batch update but request not found", request_id)
            return

        block = req.block_states[update.block_index]
        if not update.success:
            if self._sticky_worker:
                self._release_sticky_worker(req)
            else:
                self._release_nonsticky_block(req, block)
            req.status = RequestStatus.ERROR
            req.error_message = f"Block {update.block_index} failed"
            req.done_event.set()
            return

        req.status = RequestStatus.DECODING
        req.sequence_ids = list(update.updated_sequence)
        if block.prefill_end_time is None and block.prefill_enqueue_time is not None:
            block.prefill_queue_wait_duration = update.prefill_queue_wait_duration
            block.prefill_start_time = (
                block.prefill_enqueue_time + update.prefill_queue_wait_duration
            )
            block.prefill_end_time = block.prefill_start_time + update.prefill_duration
        block.decode_queue_wait_duration = max(
            block.decode_queue_wait_duration,
            update.decode_queue_wait_duration,
        )
        if block.prefill_start_time is not None:
            block.decode_start_time = (
                block.prefill_start_time
                + update.prefill_duration
                + block.decode_queue_wait_duration
            )
        elif block.prefill_end_time is not None:
            block.decode_start_time = (
                block.prefill_end_time + block.decode_queue_wait_duration
            )
        block.decode_forward_evals_applied = max(
            block.decode_forward_evals_applied,
            update.num_forward_evals_in_batch_phase,
        )
        req.recompute_num_forward_evals()
        self._apply_request_visibility(
            request_id=request_id,
            timestamp=report.batch_end_time,
            num_unmasked_tokens=update.num_unmasked_tokens,
        )

        if not update.block_completed:
            return

        self._complete_block(
            req,
            block,
            completion_time=(block.decode_start_time or report.batch_end_time)
            + update.decode_duration,
            decode_duration=update.decode_duration,
        )

    def _on_batch_metrics(self, report: sangam_pb2.BatchMetricsReport) -> None:
        # Synthesize BUSY state retroactively at batch_start_time
        self._metrics_store.on_worker_state(
            worker_id=report.worker_id,
            worker_type=report.worker_type,
            state=WorkerStateTimeline.BUSY,
            timestamp=report.batch_start_time,
            waiting_queue_depth=0,
            active_batch_size=report.batch_size,
            kv_total_pages=report.kv_total_pages,
            kv_used_pages=report.kv_used_pages,
            kv_free_pages=report.kv_free_pages,
        )
        self._metrics_store.on_batch_end(
            worker_id=report.worker_id,
            worker_type=report.worker_type,
            batch_size=report.batch_size,
            prompt_len=report.prompt_len,
            gen_len=report.gen_len,
            batch_start_time=report.batch_start_time,
            batch_end_time=report.batch_end_time,
            kv_total_pages=report.kv_total_pages,
            kv_used_pages=report.kv_used_pages,
            kv_free_pages=report.kv_free_pages,
            num_unmasked_tokens=report.num_unmasked_tokens,
            batch_phase=report.batch_phase,
            sampling_duration=report.sampling_duration,
            request_updates=report.request_updates,
            batch_op_attn_time=(
                report.batch_op_attn_time
                if report.HasField("batch_op_attn_time")
                else None
            ),
            batch_op_mlp_time=(
                report.batch_op_mlp_time
                if report.HasField("batch_op_mlp_time")
                else None
            ),
            batch_op_qkv_time=(
                report.batch_op_qkv_time
                if report.HasField("batch_op_qkv_time")
                else None
            ),
        )
        for update in report.request_updates:
            update_phase = update.request_phase
            if update_phase == sangam_pb2.BATCH_PHASE_PREFILL:
                self._apply_prefill_batch_update(report, update)
            elif update_phase == sangam_pb2.BATCH_PHASE_DECODE:
                self._apply_decode_batch_update(report, update)
            elif update_phase == sangam_pb2.BATCH_PHASE_MIXED:
                logger.error(
                    "[%s] Mixed request update phase is unsupported",
                    update.request_id,
                )
        if report.HasField("worker_state_after"):
            self._process_worker_state_snapshot(
                report.worker_state_after, report.worker_id, report.worker_type
            )
            self._metrics_store.on_worker_deficit_tokens(
                worker_id=report.worker_id,
                worker_type=report.worker_type,
                deficit_tokens=report.worker_state_after.deficit_tokens,
                timestamp=report.batch_end_time,
            )

    def _on_kv_transfer(self, report: sangam_pb2.KVTransferReport) -> None:
        logger.warning(
            "[%s] Ignoring unexpected KV transfer report in colocated mode "
            "(block=%s, worker=%s)",
            report.request_id,
            report.block_index,
            report.worker_id,
        )

    def _scheduler_status_counts(self) -> tuple[int, int, int]:
        return 0, 0, len(self._workers)

    def _on_conversion_rpc_finished(self, payload) -> None:
        logger.error("ColocatedScheduler does not handle conversion events")


class ColocatedSchedulerServicer(BaseSchedulerServicer):
    """gRPC servicer wrapping the colocated scheduler."""


def serve_colocated_scheduler(port: int, config: ColocatedSchedulerConfig) -> None:
    """Start the colocated scheduler gRPC server."""
    scheduler = ColocatedScheduler(config)
    serve_scheduler_instance(
        scheduler=scheduler,
        servicer_cls=ColocatedSchedulerServicer,
        port=port,
        service_name="colocated scheduler",
        startup_log_name="Colocated scheduler",
        logger_instance=logger,
    )
