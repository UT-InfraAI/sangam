"""Hybrid scheduler: prefill workers + colocated workers with overflow.

Combines dedicated prefill workers with colocated workers:
  - Normal flow: request → prefill worker → KV transfer → colocated worker (decode)
  - Overflow: when prefill workers are overloaded, requests go directly to
    colocated workers via EnqueueRequest (local prefill + decode, no transfer)

Colocated workers retain full token-budgeted behavior AND receive KV caches
from prefill workers for externally-prefilled requests.
"""

import heapq
import math
import random
import time

import grpc

from sangam.engine.base_scheduler import (
    BaseScheduler,
    BaseSchedulerServicer,
    EnqueueWorkerResult,
    EventType,
    WorkerInfo,
    serve_scheduler_instance,
)
from sangam.engine.scheduler_config import HybridSchedulerConfig
from sangam.engine.topology_policy import is_fast_pair, parse_kv_fast_pairs
from sangam.request import Request, RequestStatus
from sangam.logger import init_logger
from sangam.metrics.constants import SchedulerQueueTimeSeries, WorkerStateTimeline
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.types import (
    DecodeSchedulerPolicy,
    PrefillSchedulerPolicy,
    WorkerType,
)

logger = init_logger(__name__)


class HybridScheduler(BaseScheduler):
    """Manages prefill workers + colocated workers with overflow routing."""

    _MAX_KV_TRANSFER_RETRIES = 3

    def __init__(self, config: HybridSchedulerConfig):
        super().__init__(config, event_thread_name="hybrid-scheduler-loop")

        # Worker registries
        self._prefill_workers: list[WorkerInfo] = []
        self._colocated_workers: list[WorkerInfo] = []
        self._prefill_rr = 0
        self._colocated_rr = 0
        self._prefill_scheduler_policy = PrefillSchedulerPolicy(
            config.prefill_scheduler_policy
        )
        self._decode_scheduler_policy = DecodeSchedulerPolicy(
            config.decode_scheduler_policy
        )
        self._decode_grouping_slack_ratio = config.decode_grouping_slack_ratio
        self._kv_fast_pairs = parse_kv_fast_pairs(config.kv_fast_pairs)
        self._kv_topology_alpha = config.kv_topology_alpha

        # gRPC stubs cached per worker
        self._prefill_stubs: dict[str, sangam_pb2_grpc.PrefillWorkerServiceStub] = {}
        self._colocated_stubs: dict[
            str, sangam_pb2_grpc.ColocatedWorkerServiceStub
        ] = {}
        self._decode_stubs: dict[str, sangam_pb2_grpc.DecodeWorkerServiceStub] = {}

        # Decode-ready queue: requests waiting for KV transfer to colocated worker
        self._decode_ready_requests: list[tuple[float, str, Request]] = []
        self._decode_ready_request_ids: set[str] = set()

        # Overflow threshold
        self._prefill_overload_threshold = config.prefill_overload_threshold
        self._enable_prefill_overflow = config.enable_prefill_overflow

        # Throttle for the divergence warning emitted when a worker's reported
        # kv_free_pages disagrees with the scheduler's authoritative tally.
        self._last_divergence_warning_time: dict[str, float] = {}

    # ----- Worker management -----

    _DIVERGENCE_WARNING_INTERVAL_SEC = 5.0

    def _get_worker_info(
        self,
        worker_id: str,
    ) -> WorkerInfo | None:
        for worker in self._prefill_workers + self._colocated_workers:
            if worker.worker_id == worker_id:
                return worker
        return None

    def _check_free_pages_divergence(
        self,
        worker: WorkerInfo,
        reported_free: int,
        context: str,
    ) -> None:
        """Log a throttled warning when the worker reports *fewer* free
        pages than the scheduler's tally implies (i.e. scheduler's
        free_pages > reported_free by more than the tolerance).

        This is the OOM-causing direction: the scheduler believes the
        worker has capacity it does not actually have, so a future
        dispatch may push the worker's allocator past empty and surface
        as a KV transfer failure. The opposite direction (reported_free
        > scheduler_free) is benign — either eager-release lag on
        completion or a stale snapshot — and not worth surfacing.
        """
        if worker.free_pages is None or worker.max_pages is None:
            return
        tolerance = max(4, int(0.05 * worker.max_pages))
        diff = worker.free_pages - reported_free
        if diff <= tolerance:
            return
        now = time.time()
        last = self._last_divergence_warning_time.get(worker.worker_id, 0.0)
        if now - last < self._DIVERGENCE_WARNING_INTERVAL_SEC:
            return
        self._last_divergence_warning_time[worker.worker_id] = now
        logger.warning(
            "free_pages divergence on %s (%s): scheduler=%d reported=%d "
            "scheduler-reported=%+d (tolerance=%d); worker has fewer free "
            "pages than the scheduler tracks — next dispatch may OOM",
            worker.worker_id,
            context,
            worker.free_pages,
            reported_free,
            diff,
            tolerance,
        )

    def _process_worker_state_snapshot(
        self,
        snapshot: sangam_pb2.WorkerStateSnapshot,
        worker_id: str,
        worker_type: str,
    ) -> None:
        worker = self._get_worker_info(worker_id)
        if worker is not None and snapshot.kv_total_pages > 0:
            worker.max_pages = snapshot.kv_total_pages
            self._check_free_pages_divergence(
                worker, snapshot.kv_free_pages, "state_snapshot"
            )
        super()._process_worker_state_snapshot(snapshot, worker_id, worker_type)

    @staticmethod
    def _decode_ready_sort_key(req: Request) -> tuple[float, str]:
        return (req.submit_time, req.request_id)

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
            logger.error("HybridScheduler does not support worker conversion")
            return False
        try:
            wtype = WorkerType(worker_type)
        except ValueError:
            logger.error(f"Unknown worker type: {worker_type}")
            return False

        if wtype not in (WorkerType.PREFILL, WorkerType.COLOCATED):
            logger.error(
                f"HybridScheduler only accepts prefill and colocated workers, "
                f"got {wtype}"
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

        if wtype is WorkerType.PREFILL:
            self._prefill_workers.append(info)
            self._prefill_stubs[worker_id] = sangam_pb2_grpc.PrefillWorkerServiceStub(
                channel
            )
            logger.info(
                f"Registered prefill worker {worker_id} at {address} (rank={dist_rank})"
            )
        elif wtype is WorkerType.COLOCATED:
            self._colocated_workers.append(info)
            self._colocated_stubs[worker_id] = (
                sangam_pb2_grpc.ColocatedWorkerServiceStub(channel)
            )
            self._decode_stubs[worker_id] = sangam_pb2_grpc.DecodeWorkerServiceStub(
                channel
            )
            logger.info(
                f"Registered colocated worker {worker_id} at {address} "
                f"(rank={dist_rank})"
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
        if wtype is WorkerType.COLOCATED:
            self._drain_decode_ready_requests()
        return True

    def _is_ready_for_requests(self) -> bool:
        return (
            any(not w.draining for w in self._prefill_workers)
            or len(self._colocated_workers) > 0
        )

    # ----- Prefill worker selection -----

    def _pick_prefill_worker(
        self, required_seq_length: int
    ) -> tuple[WorkerInfo | None, int]:
        """Pick a prefill worker with enough memory, or return (None, 0)."""
        available = [w for w in self._prefill_workers if not w.draining]
        if not available:
            return None, 0

        tracked_eligible: list[WorkerInfo] = []
        untracked_workers: list[WorkerInfo] = []
        required_pages_by_id: dict[str, int] = {}

        for w in available:
            if w.page_size is None or w.free_pages is None:
                untracked_workers.append(w)
                continue
            required_pages = math.ceil(required_seq_length / w.page_size)
            required_pages_by_id[w.worker_id] = required_pages
            if w.free_pages >= required_pages:
                tracked_eligible.append(w)

        candidates = tracked_eligible if tracked_eligible else untracked_workers
        if not candidates:
            return None, 0

        if self._prefill_scheduler_policy is PrefillSchedulerPolicy.ROUND_ROBIN:
            w = candidates[self._prefill_rr % len(candidates)]
            self._prefill_rr += 1
        else:
            min_outstanding = min(w.outstanding_prefill_tokens for w in candidates)
            min_candidates = [
                w for w in candidates if w.outstanding_prefill_tokens == min_outstanding
            ]
            w = random.choice(min_candidates)

        required_pages = required_pages_by_id.get(w.worker_id, 0)
        if required_pages > 0:
            w.free_pages -= required_pages  # type: ignore[operator]
        return w, required_pages

    # ----- Colocated worker selection (for decode / overflow) -----

    def _pick_colocated_round_robin(
        self, candidates: list[WorkerInfo]
    ) -> WorkerInfo | None:
        if not candidates:
            return None
        candidate_ids = {w.worker_id for w in candidates}
        workers = self._colocated_workers
        num_workers = len(workers)
        for _ in range(num_workers):
            worker = workers[self._colocated_rr % num_workers]
            self._colocated_rr += 1
            if worker.worker_id in candidate_ids:
                return worker
        return None

    def _pick_colocated_max_free_memory(
        self, candidates: list[WorkerInfo]
    ) -> WorkerInfo | None:
        if not candidates:
            return None
        max_free_pages = max(w.free_pages for w in candidates)
        max_free_candidates = [w for w in candidates if w.free_pages == max_free_pages]
        return random.choice(max_free_candidates)

    def _pick_colocated_balanced_length_clustering(
        self,
        candidates: list[WorkerInfo],
        request_length: int,
    ) -> WorkerInfo | None:
        if not candidates:
            return None
        if self._decode_grouping_slack_ratio == 0:
            return self._pick_colocated_round_robin(candidates)

        projected_sums = {
            worker.worker_id: worker.active_request_length_sum + request_length
            for worker in candidates
        }
        min_projected = min(projected_sums.values())
        projected_limit = min_projected * (1 + self._decode_grouping_slack_ratio)
        slack_candidates = [
            worker
            for worker in candidates
            if projected_sums[worker.worker_id] <= projected_limit
        ]
        if not slack_candidates:
            return self._pick_colocated_round_robin(candidates)

        scored_candidates = [
            (
                abs(
                    request_length
                    - (worker.active_request_length_sum / worker.outstanding_requests)
                ),
                worker,
            )
            for worker in slack_candidates
            if worker.outstanding_requests > 0
        ]
        if not scored_candidates:
            return random.choice(slack_candidates)

        min_distance = min(score for score, _ in scored_candidates)
        best_candidates = [
            worker for score, worker in scored_candidates if score == min_distance
        ]
        return random.choice(best_candidates)

    def _pick_colocated_worker_for_decode(
        self, required_seq_length: int, prefill_gpu_id: int | None = None
    ) -> tuple[WorkerInfo | None, int]:
        """Pick a colocated worker for decode (KV transfer target)."""
        tracked_eligible: list[WorkerInfo] = []
        untracked_workers: list[WorkerInfo] = []
        required_pages_by_id: dict[str, int] = {}

        for w in self._colocated_workers:
            if w.draining:
                continue
            if w.page_size is None or w.free_pages is None:
                untracked_workers.append(w)
                continue
            required_pages = math.ceil(required_seq_length / w.page_size)
            required_pages_by_id[w.worker_id] = required_pages
            if w.free_pages >= required_pages:
                tracked_eligible.append(w)

        if tracked_eligible:
            if self._decode_scheduler_policy is DecodeSchedulerPolicy.ROUND_ROBIN:
                worker = self._pick_colocated_round_robin(tracked_eligible)
            elif (
                self._decode_scheduler_policy
                is DecodeSchedulerPolicy.TOPOLOGY_GUARDED_MEMORY
            ):
                worker = self._pick_colocated_topology_guarded_memory(
                    tracked_eligible, prefill_gpu_id
                )
            elif (
                self._decode_scheduler_policy
                is DecodeSchedulerPolicy.BALANCED_LENGTH_CLUSTERING
            ):
                worker = self._pick_colocated_balanced_length_clustering(
                    tracked_eligible, required_seq_length
                )
            else:
                worker = self._pick_colocated_max_free_memory(tracked_eligible)
            if worker is None:
                return None, 0
            required_pages = required_pages_by_id[worker.worker_id]
            worker.free_pages -= required_pages
            return worker, required_pages

        if untracked_workers:
            worker = self._pick_colocated_round_robin(untracked_workers)
            if worker is not None:
                return worker, 0

        return None, 0

    def _pick_colocated_topology_guarded_memory(
        self,
        candidates: list[WorkerInfo],
        prefill_gpu_id: int | None,
    ) -> WorkerInfo | None:
        if not candidates:
            return None
        mem_best = self._pick_colocated_max_free_memory(candidates)
        if mem_best is None or prefill_gpu_id is None:
            return mem_best

        topo_candidates = [
            worker
            for worker in candidates
            if is_fast_pair(self._kv_fast_pairs, prefill_gpu_id, worker.gpu_id)
        ]
        topo_best = self._pick_colocated_max_free_memory(topo_candidates)
        if topo_best is None:
            return mem_best
        if topo_best.free_pages is None or mem_best.free_pages is None:
            return mem_best
        if topo_best.free_pages >= self._kv_topology_alpha * mem_best.free_pages:
            return topo_best
        return mem_best

    def _decode_ready_reserved_pages(self, page_size: int) -> int:
        """Aggregate KV-page demand of completed prefills awaiting colocated
        memory (the decode-ready backlog).

        These requests finished prefill on a dedicated prefill worker and only
        need to land on a colocated worker to decode; until they do, their KV
        stays pinned on the prefill worker. They must out-rank brand-new
        overflow prefills for colocated memory, so overflow admission reserves
        this many pages as headroom. Stale heap entries (already scheduled or
        no longer waiting) are skipped, mirroring _drain_decode_ready_requests.
        """
        reserved = 0
        for _, _, req in self._decode_ready_requests:
            if (
                req.request_id not in self._decode_ready_request_ids
                or req.status is not RequestStatus.WAITING_DECODE
                or req.current_block is None
            ):
                continue
            reserved += math.ceil(len(req.sequence_ids) / page_size)
        return reserved

    def _pick_overflow_colocated_candidate(
        self, required_seq_length: int
    ) -> tuple[WorkerInfo | None, int]:
        """Pick a colocated worker for an overflow prefill, ranked by least
        outstanding_prefill_tokens among those with enough free pages.

        Overflow never targets a dedicated prefill worker. The normal path
        already routed to any prefill worker that was both under-threshold and
        had memory, so reaching overflow means every prefill worker is either
        overloaded or out of memory; adding prefills there only deepens the
        overload. Overflow therefore stays colocated-only.

        Completed prefills waiting for colocated memory (the decode-ready
        backlog) out-rank brand-new overflow prefills: their KV is pinned on a
        prefill worker and they only need to decode. Admission holds back
        overflow unless aggregate free colocated memory exceeds the backlog's
        reserved headroom. Reserves the chosen worker's pages on success.

        Returns (worker, required_pages), or (None, 0) when held back so the
        caller queues the request as pending.
        """
        tracked_eligible: list[tuple[WorkerInfo, int]] = []
        untracked_colocated: list[WorkerInfo] = []
        representative_page_size: int | None = None
        available_pages = 0

        for w in self._colocated_workers:
            if w.draining:
                continue
            # Symmetric overload gate: a colocated worker may receive a new
            # overflow prefill only while its prefill load is below the same
            # threshold that marks a prefill worker overloaded. Overflow is
            # entered only when every prefill worker is at/over that threshold
            # (or out of memory), so an eligible colocated worker sits strictly
            # below it, keeping colocated prefill load from overtaking the
            # prefill workers it is overflowing from. Overloaded colocated
            # workers still count toward available_pages: the decode-ready
            # backlog is not overload-gated and can still land on them.
            overloaded = (
                w.outstanding_prefill_tokens >= self._prefill_overload_threshold
            )
            if w.page_size is None or w.free_pages is None:
                if not overloaded:
                    untracked_colocated.append(w)
                continue
            representative_page_size = w.page_size
            available_pages += w.free_pages
            required_pages = math.ceil(required_seq_length / w.page_size)
            if not overloaded and w.free_pages >= required_pages:
                tracked_eligible.append((w, required_pages))

        if tracked_eligible and representative_page_size is not None:
            reserved_pages = self._decode_ready_reserved_pages(representative_page_size)
            demand = math.ceil(required_seq_length / representative_page_size)
            if available_pages - reserved_pages < demand:
                # Hold back: the completed-prefill backlog needs this memory.
                return None, 0
            min_outstanding = min(
                w.outstanding_prefill_tokens for w, _ in tracked_eligible
            )
            candidates = [
                (w, pages)
                for w, pages in tracked_eligible
                if w.outstanding_prefill_tokens == min_outstanding
            ]
            worker, required_pages = random.choice(candidates)
            worker.free_pages -= required_pages  # type: ignore[operator]
            return worker, required_pages

        if untracked_colocated:
            num_workers = len(self._colocated_workers)
            untracked_ids = {w.worker_id for w in untracked_colocated}
            for _ in range(num_workers):
                worker = self._colocated_workers[self._colocated_rr % num_workers]
                self._colocated_rr += 1
                if worker.worker_id in untracked_ids:
                    return worker, 0

        return None, 0

    # ----- Overflow detection -----

    def _is_prefill_overloaded(self) -> bool:
        """Check if prefill workers are overloaded and we should overflow."""
        non_draining = [w for w in self._prefill_workers if not w.draining]
        if not non_draining:
            return True  # no prefill workers = always overflow
        return all(
            w.outstanding_prefill_tokens >= self._prefill_overload_threshold
            for w in non_draining
        )

    # ----- Outstanding token tracking -----

    def _increment_prefill_outstanding_tokens(
        self,
        worker_id: str | None,
        num_tokens: int,
    ) -> None:
        if worker_id is None:
            return
        for worker in self._prefill_workers + self._colocated_workers:
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
        for worker in self._prefill_workers + self._colocated_workers:
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

    def _increment_decode_assignment_load(
        self,
        worker_id: str | None,
        request_length: int,
    ) -> None:
        if worker_id is None:
            return
        for worker in self._colocated_workers:
            if worker.worker_id == worker_id:
                worker.outstanding_requests += 1
                worker.active_request_length_sum += request_length
                return

    def _decrement_decode_assignment_load(
        self,
        worker_id: str | None,
        request_length: int,
    ) -> None:
        if worker_id is None:
            return
        for worker in self._colocated_workers:
            if worker.worker_id == worker_id:
                worker.outstanding_requests = max(0, worker.outstanding_requests - 1)
                worker.active_request_length_sum = max(
                    0, worker.active_request_length_sum - request_length
                )
                return

    # ----- Reservation release -----

    def _release_prefill_reservation(self, block) -> None:
        if not block.prefill_worker_id:
            return
        for worker in self._prefill_workers:
            if (
                worker.worker_id == block.prefill_worker_id
                and worker.free_pages is not None
            ):
                worker.free_pages += block.prefill_reserved_pages
                break
        block.prefill_reserved_pages = 0
        self._drain_pending_requests()

    def _release_decode_reservation(self, block) -> None:
        if not block.decode_worker_id:
            return
        assigned_decode_worker_id = block.decode_worker_id
        request_length = block.reserved_decode_request_length
        for worker in self._colocated_workers:
            if (
                worker.worker_id == block.decode_worker_id
                and worker.free_pages is not None
            ):
                worker.free_pages += block.reserved_pages
                break
        if request_length > 0:
            self._decrement_decode_assignment_load(
                assigned_decode_worker_id, request_length
            )
        block.reserved_pages = 0
        block.reserved_decode_request_length = 0
        block.decode_worker_id = None
        # Drain completed prefills first so they claim the freed pages ahead of
        # held-back overflow prefills, then retry overflow against the headroom
        # that remains. Order encodes the decode-ready > new-prefill priority.
        self._drain_decode_ready_requests()
        self._drain_pending_requests()

    # ----- Request dispatch -----

    def _on_request_arrived(self, req: Request) -> None:
        """Queue the request, then drain pending in strict FIFO order.

        Admission is never performed inline here: the request is enqueued onto
        the submit-time-ordered pending heap and `_drain_pending_requests`
        decides placement head-first. This guarantees a fresh arrival is
        ordered against the existing backlog instead of leapfrogging it.
        """
        request_id = req.request_id
        block = req.current_block
        if block is None:
            return
        now = time.time()

        self._mark_scheduler_wait(block, now, "prefill")
        if (
            not self._is_ready_for_requests()
            and request_id not in self._pending_requests
        ):
            logger.warning(f"[{request_id}] No workers registered yet; request queued")
        self._queue_pending_request(req, now=now)
        self._drain_pending_requests()

    def _try_admit_request(self, req: Request, *, now: float) -> bool:
        """Attempt to place `req` on a prefill or overflow colocated worker.

        Returns True if the request was dispatched, False if no path had
        capacity. Performs no pending-queue bookkeeping: the caller
        (`_drain_pending_requests`) owns enqueue/dequeue so head-of-line
        ordering stays centralized.
        """
        # Decide: normal prefill path or overflow via unified candidate pool.
        if not self._is_prefill_overloaded():
            pw, prefill_reserved_pages = self._pick_prefill_worker(
                required_seq_length=len(req.sequence_ids)
            )
            if pw is not None:
                self._dispatch_to_prefill_worker(req, pw, prefill_reserved_pages, now)
                return True
            # Prefill workers full on memory — fall through to overflow

        if self._enable_prefill_overflow and self._colocated_workers:
            worker, reserved_pages = self._pick_overflow_colocated_candidate(
                len(req.sequence_ids)
            )
            if worker is not None:
                self._dispatch_to_colocated_worker_overflow(
                    req, worker, reserved_pages, now
                )
                return True

        return False

    def _drain_pending_requests(self) -> None:
        """Admit pending requests in strict FIFO head-of-line order.

        Pops the submit-time-ordered heap and stops at the first request that
        cannot be placed on any path, re-pushing it and leaving the tail
        untouched. Mirrors the break-on-blocked-head semantics of
        `_drain_decode_ready_requests` and the colocated worker's prefill
        admission (`_collect_prefill_batch`).
        """
        if not self._is_ready_for_requests() or not self._pending_requests:
            return
        now = time.time()
        while self._pending_request_heap:
            key = heapq.heappop(self._pending_request_heap)
            _, request_id = key
            req = self._pending_requests.get(request_id)
            if req is None:
                # Stale heap entry (already removed elsewhere).
                continue
            if req.status.is_finished():
                self._pending_requests.pop(request_id, None)
                continue
            if self._try_admit_request(req, now=now):
                self._pending_requests.pop(request_id, None)
                continue
            # Head is blocked: stop, holding it and everything behind it.
            heapq.heappush(self._pending_request_heap, key)
            break
        self._record_pending_requests_depth(now)

    def _dispatch_to_prefill_worker(
        self,
        req: Request,
        pw: WorkerInfo,
        prefill_reserved_pages: int,
        now: float,
    ) -> None:
        """Send request to a dedicated prefill worker."""
        block = req.current_block
        self._clear_scheduler_wait(req.request_id, block, now)

        block.prefill_worker_id = pw.worker_id
        block.prefill_reserved_pages = prefill_reserved_pages
        self._increment_prefill_outstanding_tokens(
            pw.worker_id, req.request_accounting_tokens
        )

        req.status = RequestStatus.PREFILLING
        block.prefill_enqueue_time = time.time()

        logger.debug(
            "[%s] Block %s: prefill on %s",
            req.request_id,
            block.block_index,
            pw.worker_id,
        )

        stub = self._prefill_stubs[pw.worker_id]
        enqueue_req = sangam_pb2.EnqueuePrefillRequest(
            request_id=req.request_id,
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
            stub.EnqueuePrefill,
            enqueue_req,
            pw.worker_id,
            WorkerType.PREFILL.value,
        )

    def _dispatch_to_colocated_worker_overflow(
        self,
        req: Request,
        worker: WorkerInfo,
        reserved_pages: int,
        now: float,
    ) -> None:
        """Send request to a colocated worker for local prefill + decode."""
        block = req.current_block
        self._clear_scheduler_wait(req.request_id, block, now)

        block.prefill_worker_id = worker.worker_id
        block.decode_worker_id = worker.worker_id
        block.reserved_pages = reserved_pages
        block.reserved_decode_request_length = len(req.sequence_ids)
        self._increment_decode_assignment_load(worker.worker_id, len(req.sequence_ids))
        self._increment_prefill_outstanding_tokens(
            worker.worker_id, req.request_accounting_tokens
        )

        req.status = RequestStatus.PREFILLING
        block.prefill_enqueue_time = time.time()

        logger.debug(
            "[%s] Block %s: overflow to colocated %s",
            req.request_id,
            block.block_index,
            worker.worker_id,
        )

        stub = self._colocated_stubs[worker.worker_id]
        enqueue_req = sangam_pb2.EnqueueColocatedRequest(
            request_id=req.request_id,
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

    # ----- KV transfer scheduling (prefill → colocated) -----

    def _try_schedule_decode_transfer(
        self,
        req: Request,
        *,
        now: float,
    ) -> str:
        block = req.current_block
        if block is None:
            return "failed"

        required_seq_length = len(req.sequence_ids)
        prefill_gpu_id = None
        if block.prefill_worker_id is not None:
            prefill_worker = self._get_worker_info(block.prefill_worker_id)
            if prefill_worker is not None:
                prefill_gpu_id = prefill_worker.gpu_id
        cw, required_pages = self._pick_colocated_worker_for_decode(
            required_seq_length=required_seq_length,
            prefill_gpu_id=prefill_gpu_id,
        )
        if cw is None:
            return "blocked"

        self._clear_scheduler_wait(req.request_id, block, now)
        block.decode_worker_id = cw.worker_id
        block.reserved_pages = required_pages
        block.reserved_decode_request_length = len(req.sequence_ids)
        self._increment_decode_assignment_load(cw.worker_id, len(req.sequence_ids))
        req.status = RequestStatus.WAITING_DECODE

        prefill_stub = self._prefill_stubs[block.prefill_worker_id]
        decode_request = sangam_pb2.EnqueueDecodeRequest(
            request_id=req.request_id,
            sequence_ids=req.sequence_ids,
            block_start=block.block_start,
            block_end=block.block_end,
            block_index=block.block_index,
            request_seed=req.request_seed,
            mask_id=req.mask_id,
            arrival_time=req.submit_time,
            decode_enqueue_time=0.0,
        )
        decode_request.sampling_parameters.CopyFrom(req.sampling_parameters.to_proto())
        response = self._timed_grpc_call(
            prefill_stub.TriggerKVTransfer,
            sangam_pb2.TriggerKVTransferRequest(
                request_id=req.request_id,
                block_index=block.block_index,
                decode_worker_id=cw.worker_id,
                decode_dst_rank=cw.dist_rank,
                decode_worker_address=cw.address,
                decode_request=decode_request,
            ),
            "TriggerKVTransfer",
        )
        if response.success:
            block.kv_transfer_trigger_time = now
            logger.debug(
                "[%s] Block %s: transfer from %s to %s",
                req.request_id,
                block.block_index,
                block.prefill_worker_id,
                cw.worker_id,
            )
            return "scheduled"

        req.status = RequestStatus.ERROR
        req.error_message = (
            f"Failed to trigger KV transfer for block {block.block_index}"
        )
        req.done_event.set()
        self._release_decode_reservation(block)
        return "failed"

    def _enqueue_decode_ready_request(self, req: Request, *, now: float) -> None:
        if req.request_id not in self._decode_ready_request_ids:
            req_key = self._decode_ready_sort_key(req)
            heapq.heappush(self._decode_ready_requests, (*req_key, req))
            self._decode_ready_request_ids.add(req.request_id)
            self._record_decode_ready_depth()
        block = req.current_block
        if block is not None:
            self._mark_scheduler_wait(block, now, "decode")

    def _drain_decode_ready_requests(self, *, now: float | None = None) -> None:
        if not self._colocated_workers or not self._decode_ready_requests:
            return
        initial_len = len(self._decode_ready_requests)
        while self._decode_ready_requests:
            _, _, req = self._decode_ready_requests[0]
            if (
                req.request_id not in self._decode_ready_request_ids
                or req.status is not RequestStatus.WAITING_DECODE
                or req.current_block is None
            ):
                _, _, stale_req = heapq.heappop(self._decode_ready_requests)
                self._decode_ready_request_ids.discard(stale_req.request_id)
                continue
            result = self._try_schedule_decode_transfer(
                req, now=time.time() if now is None else now
            )
            if result == "scheduled":
                _, _, scheduled_req = heapq.heappop(self._decode_ready_requests)
                self._decode_ready_request_ids.discard(scheduled_req.request_id)
                continue
            if result == "failed":
                _, _, failed_req = heapq.heappop(self._decode_ready_requests)
                self._decode_ready_request_ids.discard(failed_req.request_id)
                continue
            break
        if len(self._decode_ready_requests) != initial_len:
            self._record_decode_ready_depth()

    def _record_decode_ready_depth(self) -> None:
        self._metrics_store.on_scheduler_queue_depth(
            SchedulerQueueTimeSeries.DECODE_READY_REQUESTS,
            len(self._decode_ready_requests),
            time.time(),
        )

    # ----- Enqueue result handling -----

    def _handle_enqueue_worker_done(self, result: EnqueueWorkerResult) -> None:
        if result.success:
            return

        req = self._requests.get(result.request_id)
        if req is None or req.status.is_finished():
            return

        block = req.current_block
        if block is None or block.block_index != result.block_index:
            return

        if result.worker_type == WorkerType.PREFILL.value:
            if block.prefill_worker_id != result.worker_id:
                return
            logger.warning(
                "[%s] Requeueing block %s after failed prefill enqueue on %s: %s",
                req.request_id,
                block.block_index,
                result.worker_id,
                result.error_message or "enqueue failed",
            )
            self._decrement_prefill_outstanding_tokens(
                block.prefill_worker_id, req.request_accounting_tokens
            )
            self._release_prefill_reservation(block)
            block.prefill_worker_id = None
            req.status = RequestStatus.PENDING
            self._enqueue_event(EventType.REQUEST_ARRIVED, req)
        elif result.worker_type == WorkerType.COLOCATED.value:
            logger.warning(
                "[%s] Requeueing block %s after failed colocated enqueue on %s: %s",
                req.request_id,
                block.block_index,
                result.worker_id,
                result.error_message or "enqueue failed",
            )
            self._decrement_prefill_outstanding_tokens(
                block.prefill_worker_id, req.request_accounting_tokens
            )
            self._release_decode_reservation(block)
            block.prefill_worker_id = None
            req.status = RequestStatus.PENDING
            self._enqueue_event(EventType.REQUEST_ARRIVED, req)

    # ----- Batch metrics -----

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
        release_decode_reservation: bool,
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
        if release_decode_reservation:
            self._release_decode_reservation(block)

        logger.debug(
            "[%s] Block %s: complete, %s fwd evals, %.3fs decode",
            req.request_id,
            block.block_index,
            block.decode_forward_evals_applied,
            decode_duration,
        )

        req.current_block_index += 1
        if req.current_block_index < req.target_blocks:
            # Route the continuation block through the FIFO pending queue rather
            # than dispatching inline. The request keeps its original (oldest)
            # submit_time, so it head-sorts ahead of newer pending requests,
            # prioritizing in-flight work without leapfrogging the backlog.
            next_block = req.current_block
            if next_block is not None:
                self._mark_scheduler_wait(next_block, completion_time, "prefill")
            req.status = RequestStatus.WAITING_NEXT_BLOCK
            self._queue_pending_request(req, now=completion_time)
            self._drain_pending_requests()
        else:
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

    @staticmethod
    def _block_uses_external_decode(block) -> bool:
        return (
            block.prefill_worker_id is not None
            and block.decode_worker_id is not None
            and block.prefill_worker_id != block.decode_worker_id
        )

    def _apply_prefill_batch_update_from_prefill_worker(
        self,
        report: sangam_pb2.BatchMetricsReport,
        update: sangam_pb2.BatchRequestUpdate,
    ) -> None:
        """Handle prefill completion on a dedicated prefill worker.

        After prefill, schedule KV transfer to a colocated worker for decode.
        """
        request_id = update.request_id
        req = self._requests.get(request_id)
        if req is None:
            logger.error("[%s] Prefill batch update but request not found", request_id)
            return

        block = req.block_states[update.block_index]
        self._decrement_prefill_outstanding_tokens(
            block.prefill_worker_id, req.request_accounting_tokens
        )

        if not update.success:
            req.status = RequestStatus.ERROR
            req.error_message = f"Prefill failed for block {update.block_index}"
            req.done_event.set()
            self._release_prefill_reservation(block)
            return

        block.prefill_queue_wait_duration = update.prefill_queue_wait_duration
        prefill_duration = update.prefill_duration
        kv_transfer_duration = (
            update.kv_transfer_duration
            if update.HasField("kv_transfer_duration")
            else 0.0
        )
        if block.prefill_enqueue_time is not None:
            block.prefill_start_time = (
                block.prefill_enqueue_time + update.prefill_queue_wait_duration
            )
            block.prefill_end_time = block.prefill_start_time + prefill_duration
        else:
            block.prefill_end_time = report.batch_end_time
        if kv_transfer_duration > 0:
            block.kv_transfer_start_time = block.prefill_end_time
            block.kv_transfer_end_time = block.prefill_end_time + kv_transfer_duration
            self._metrics_store.on_block_kv_transfer_end(
                request_id, block.block_index, kv_transfer_duration
            )
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
            request_id, block.block_index, prefill_duration, report.worker_id
        )
        if update.block_completed:
            self._release_prefill_reservation(block)
            req.status = RequestStatus.DECODING
            block.decode_start_time = block.prefill_end_time or report.batch_end_time
            self._complete_block(
                req,
                block,
                completion_time=block.decode_start_time,
                decode_duration=0.0,
                release_decode_reservation=False,
            )
            return

        # Schedule KV transfer to colocated worker
        req.status = RequestStatus.WAITING_DECODE
        self._enqueue_decode_ready_request(req, now=report.batch_end_time)
        self._drain_decode_ready_requests(now=report.batch_end_time)

    def _apply_prefill_batch_update_from_colocated_worker(
        self,
        report: sangam_pb2.BatchMetricsReport,
        update: sangam_pb2.BatchRequestUpdate,
    ) -> None:
        """Handle prefill completion on a colocated worker (overflow path).

        The colocated worker did the prefill locally. Decode follows locally.
        """
        request_id = update.request_id
        req = self._requests.get(request_id)
        if req is None:
            logger.error("[%s] Prefill batch update but request not found", request_id)
            return

        block = req.block_states[update.block_index]
        self._decrement_prefill_outstanding_tokens(
            block.prefill_worker_id, req.request_accounting_tokens
        )
        self._drain_pending_requests()

        if not update.success:
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
                release_decode_reservation=True,
            )
            return
        # Decode follows locally on the same colocated worker
        req.status = RequestStatus.DECODING

    def _apply_decode_batch_update(
        self,
        report: sangam_pb2.BatchMetricsReport,
        update: sangam_pb2.BatchRequestUpdate,
    ) -> None:
        """Handle decode progress from a colocated worker."""
        request_id = update.request_id
        req = self._requests.get(request_id)
        if req is None:
            logger.error("[%s] Decode batch update but request not found", request_id)
            return

        block = req.block_states[update.block_index]
        if not update.success:
            req.status = RequestStatus.ERROR
            req.error_message = f"Block {update.block_index} failed"
            req.done_event.set()
            self._release_decode_reservation(block)
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
        if block.kv_transfer_end_time is not None:
            # Externally-prefilled: decode wait starts after KV transfer
            block.decode_start_time = (
                block.kv_transfer_end_time + block.decode_queue_wait_duration
            )
        elif self._block_uses_external_decode(block):
            block.decode_start_time = max(
                report.batch_end_time - update.decode_duration,
                0.0,
            )
        elif block.prefill_start_time is not None:
            # Locally-prefilled (overflow): decode wait starts after prefill
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
            release_decode_reservation=True,
        )

    def _on_batch_metrics(self, report: sangam_pb2.BatchMetricsReport) -> None:
        worker = self._get_worker_info(report.worker_id)
        if worker is not None and report.kv_total_pages > 0:
            worker.max_pages = report.kv_total_pages
            self._check_free_pages_divergence(
                worker, report.kv_free_pages, "batch_metrics"
            )
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
            effective_phase = update.request_phase
            if effective_phase == sangam_pb2.BATCH_PHASE_PREFILL:
                if report.worker_type == WorkerType.PREFILL.value:
                    self._apply_prefill_batch_update_from_prefill_worker(report, update)
                else:
                    self._apply_prefill_batch_update_from_colocated_worker(
                        report, update
                    )
            elif effective_phase == sangam_pb2.BATCH_PHASE_DECODE:
                self._apply_decode_batch_update(report, update)
            elif effective_phase == sangam_pb2.BATCH_PHASE_MIXED:
                logger.error(
                    "[%s] Mixed request update phase is unsupported",
                    update.request_id,
                )

        if report.HasField("worker_state_after"):
            self._process_worker_state_snapshot(
                report.worker_state_after, report.worker_id, report.worker_type
            )
            if report.worker_type == WorkerType.COLOCATED.value:
                self._metrics_store.on_worker_deficit_tokens(
                    worker_id=report.worker_id,
                    worker_type=report.worker_type,
                    deficit_tokens=report.worker_state_after.deficit_tokens,
                    timestamp=report.batch_end_time,
                )

    # ----- KV transfer reports -----

    def _on_kv_transfer(self, report: sangam_pb2.KVTransferReport) -> None:
        request_id = report.request_id
        req = self._requests.get(request_id)
        if req is None:
            logger.error("[%s] KV transfer update but request not found", request_id)
            return

        block = req.block_states[report.block_index]
        decode_progress_applied = (
            block.decode_forward_evals_applied > 0 or block.completed
        )
        if not report.success:
            if req.status.is_finished() or decode_progress_applied:
                logger.warning(
                    "[%s] Ignoring stale KV transfer failure for block %s after decode progress",
                    request_id,
                    report.block_index,
                )
                self._release_prefill_reservation(block)
                return
            block.kv_transfer_failures += 1
            if block.kv_transfer_failures < self._MAX_KV_TRANSFER_RETRIES:
                logger.warning(
                    "[%s] KV transfer failed for block %s (attempt %s/%s), "
                    "requeueing for re-prefill",
                    request_id,
                    report.block_index,
                    block.kv_transfer_failures,
                    self._MAX_KV_TRANSFER_RETRIES,
                )
                self._release_decode_reservation(block)
                self._requeue_prefill_request(req, block_index=report.block_index)
            else:
                req.status = RequestStatus.ERROR
                req.error_message = (
                    report.error_message
                    if report.HasField("error_message")
                    else f"KV transfer failed for block {report.block_index}"
                )
                req.done_event.set()
                self._release_prefill_reservation(block)
                self._release_decode_reservation(block)
            return

        if block.kv_transfer_end_time is None:
            block.kv_transfer_start_time = (
                block.kv_transfer_trigger_time
                if block.kv_transfer_trigger_time is not None
                else report.transfer_start_time
            )
            block.kv_transfer_end_time = report.transfer_end_time
            self._metrics_store.on_block_kv_transfer_end(
                request_id, block.block_index, block.kv_transfer_duration
            )

        if not req.status.is_finished() and not decode_progress_applied:
            req.status = RequestStatus.WAITING_DECODE
            if block.decode_enqueue_time is None:
                block.decode_enqueue_time = report.transfer_end_time
        self._release_prefill_reservation(block)
        if report.HasField("worker_state_after"):
            self._process_worker_state_snapshot(
                report.worker_state_after,
                report.worker_id,
                WorkerType.PREFILL.value,
            )

    def _requeue_prefill_request(self, req: Request, *, block_index: int) -> None:
        if block_index >= len(req.block_states):
            return
        block = req.block_states[block_index]
        self._decrement_prefill_outstanding_tokens(
            block.prefill_worker_id, req.request_accounting_tokens
        )
        self._release_prefill_reservation(block)
        block.prefill_worker_id = None
        block.prefill_progress_applied = False
        block.decode_enqueue_time = None
        block.kv_transfer_trigger_time = None
        block.kv_transfer_start_time = None
        block.kv_transfer_end_time = None
        if req.current_block_index != block_index or req.status.is_finished():
            return
        req.status = RequestStatus.PENDING
        self._enqueue_event(EventType.REQUEST_ARRIVED, req)

    def _on_conversion_rpc_finished(self, payload) -> None:
        logger.error("HybridScheduler does not handle conversion events")

    def _on_prefill_redistribution_done(self, payload) -> None:
        logger.error("HybridScheduler does not handle redistribution events")

    def _scheduler_status_counts(self) -> tuple[int, int, int]:
        return len(self._prefill_workers), 0, len(self._colocated_workers)


class HybridSchedulerServicer(BaseSchedulerServicer):
    """gRPC servicer wrapping the hybrid scheduler."""


def serve_hybrid_scheduler(port: int, config: HybridSchedulerConfig) -> None:
    """Start the hybrid scheduler gRPC server."""
    scheduler = HybridScheduler(config)
    serve_scheduler_instance(
        scheduler=scheduler,
        servicer_cls=HybridSchedulerServicer,
        port=port,
        service_name="hybrid scheduler",
        startup_log_name="Hybrid scheduler",
        logger_instance=logger,
    )
