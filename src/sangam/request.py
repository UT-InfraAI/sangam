import itertools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from sangam.sampling_parameters import SamplingParameters

_request_id_counter = itertools.count(1)


class RequestStatus(Enum):
    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    WAITING_DECODE = "WAITING_DECODE"
    DECODING = "DECODING"
    WAITING_NEXT_BLOCK = "WAITING_NEXT_BLOCK"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

    def is_finished(self) -> bool:
        return self in (RequestStatus.COMPLETED, RequestStatus.ERROR)

    def is_executing(self) -> bool:
        return self in (RequestStatus.PREFILLING, RequestStatus.DECODING)

    def is_waiting(self) -> bool:
        return self in (
            RequestStatus.PENDING,
            RequestStatus.WAITING_NEXT_BLOCK,
            RequestStatus.WAITING_DECODE,
        )


@dataclass
class BlockState:
    """Tracks the state of a single generation block within a request.

    Attributes:
        block_index: Zero-based index of this block.
        block_start: Absolute start position in the full sequence.
        block_end: Absolute end position (exclusive) in the full sequence.
    """

    block_index: int
    block_start: int
    block_end: int

    decode_step: int = 0

    # Assigned worker IDs for this block
    prefill_worker_id: str | None = None
    decode_worker_id: str | None = None
    streaming_transfer: bool = False

    # Memory tracking (set by scheduler at assignment)
    prefill_reserved_pages: int = 0
    reserved_pages: int = 0
    reserved_decode_request_length: int = 0

    completed: bool = False

    # Per-block timing (set by scheduler during processing)
    prefill_enqueue_time: float | None = None
    scheduler_wait_start_time: float | None = None
    scheduler_wait_phase: str | None = None
    scheduler_wait_duration: float = 0.0
    prefill_scheduler_wait_duration: float = 0.0
    decode_scheduler_wait_duration: float = 0.0
    prefill_start_time: float | None = None
    prefill_end_time: float | None = None
    prefill_queue_wait_duration: float = 0.0
    decode_enqueue_time: float | None = None
    kv_transfer_trigger_time: float | None = None
    kv_transfer_start_time: float | None = None
    kv_transfer_end_time: float | None = None
    decode_start_time: float | None = None
    decode_end_time: float | None = None
    decode_queue_wait_duration: float = 0.0
    prefill_forward_evals_applied: int = 0
    decode_forward_evals_applied: int = 0
    prefill_progress_applied: bool = False
    decode_progress_applied: bool = False
    kv_transfer_failures: int = 0

    @property
    def prefill_duration(self) -> float:
        if self.prefill_start_time is not None and self.prefill_end_time is not None:
            return self.prefill_end_time - self.prefill_start_time
        return 0.0

    @property
    def kv_transfer_duration(self) -> float:
        if (
            self.kv_transfer_start_time is not None
            and self.kv_transfer_end_time is not None
        ):
            return self.kv_transfer_end_time - self.kv_transfer_start_time
        return 0.0

    @property
    def decode_duration(self) -> float:
        if self.decode_start_time is not None and self.decode_end_time is not None:
            return self.decode_end_time - self.decode_start_time
        return 0.0


@dataclass
class Request:
    """A single generation request, tracking its lifecycle from submission to completion.

    The request is decomposed into fixed-size blocks. Each block goes through
    prefill, KV-cache transfer, and iterative decode iterations.

    Attributes:
        prompt_token_ids: Input token IDs from the user prompt.
        gen_length: Number of mask tokens appended to the prompt (the
            generation buffer the model attends over).
        block_length: Number of tokens per generation block.
        target_blocks: Number of blocks the request must fully unmask before
            it is considered complete. Defaults to ``gen_length //
            block_length`` (i.e. unmask the entire buffer). When the server
            is configured with ``--max-gen-len`` and a per-request requested
            generation length, ``target_blocks`` may be smaller than the
            buffer so the model sees a uniform ``max_gen_len`` mask suffix
            but the scheduler stops decoding earlier.
    """

    prompt_token_ids: list[int]
    gen_length: int
    block_length: int
    sampling_parameters: SamplingParameters
    mask_id: int
    request_seed: int

    # Auto-generated fields
    request_id: str = field(default_factory=lambda: str(next(_request_id_counter)))
    status: RequestStatus = RequestStatus.PENDING

    # Mutable sequence: prompt + mask tokens, updated as blocks are decoded
    sequence_ids: list[int] = field(default_factory=list)

    # Block tracking
    block_states: list[BlockState] = field(default_factory=list)
    current_block_index: int = 0
    # Per-request stop point in fully-unmasked blocks. 0 means "default to
    # len(block_states)" and is filled in by __post_init__.
    target_blocks: int = 0

    # Metrics
    num_forward_evals: int = 0
    submit_time: float = field(default_factory=time.time)
    complete_time: float | None = None
    error_message: str | None = None
    last_visible_token_time: float | None = None
    visible_generated_tokens: int = 0

    # Signalled by the scheduler when the request reaches COMPLETED or ERROR.
    # The blocking Generate RPC parks on this so handler threads can wait on a
    # futex (GIL released) instead of polling Poll on a fixed interval.
    done_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def mark_terminal(
        self,
        status: RequestStatus,
        *,
        complete_time: float,
        error_message: str | None = None,
    ) -> None:
        """Transition to a terminal status (COMPLETED or ERROR) and wake any
        thread waiting on ``done_event``. Must be called from the scheduler's
        single-threaded event loop so the request is fully populated before
        the event fires."""
        assert status.is_finished(), f"mark_terminal called with {status}"
        self.status = status
        self.complete_time = complete_time
        if error_message is not None:
            self.error_message = error_message
        self.done_event.set()

    def __post_init__(self):
        self._verify_block_config()
        if not self.sequence_ids:
            self.sequence_ids = (
                list(self.prompt_token_ids) + [self.mask_id] * self.gen_length
            )
        if not self.block_states:
            self._init_block_states()
        if self.target_blocks <= 0:
            self.target_blocks = len(self.block_states)
        elif self.target_blocks > len(self.block_states):
            raise ValueError(
                f"target_blocks ({self.target_blocks}) exceeds the number of "
                f"allocated blocks ({len(self.block_states)})"
            )

    def _verify_block_config(self) -> None:
        """Validate block and step configuration at construction time."""
        if self.gen_length % self.block_length != 0:
            raise ValueError(
                f"gen_length ({self.gen_length}) must be divisible by "
                f"block_length ({self.block_length})"
            )

    def _init_block_states(self) -> None:
        """Create initial BlockState objects for each generation block."""
        prompt_len = len(self.prompt_token_ids)
        num_blocks = self.gen_length // self.block_length
        for i in range(num_blocks):
            s = prompt_len + i * self.block_length
            e = s + self.block_length
            self.block_states.append(
                BlockState(
                    block_index=i,
                    block_start=s,
                    block_end=e,
                )
            )

    @property
    def current_block(self) -> BlockState | None:
        if self.current_block_index < len(self.block_states):
            return self.block_states[self.current_block_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.status == RequestStatus.COMPLETED

    @cached_property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def target_gen_tokens(self) -> int:
        """Number of tokens the request will actually generate end-to-end.

        With ``--max-gen-len`` configured, the mask buffer is sized to the
        server-wide ``gen_length`` but only ``target_blocks`` worth of it is
        decoded; per-token throughput / latency normalization should use this
        rather than the buffer length.
        """
        return self.target_blocks * self.block_length

    @property
    def request_accounting_tokens(self) -> int:
        """Buffer footprint used by scheduler load / length-clustering policies.

        Equal to ``prompt_length + gen_length``, where ``gen_length`` is the
        mask-buffer size and matches the server-wide ``--max-gen-len`` when
        that flag is set. This intentionally ignores ``target_blocks``: the
        worker still holds the full buffer until the request exits, so
        scheduling decisions should be made against the buffer footprint, not
        against the (potentially smaller) number of tokens that will actually
        be emitted. For per-emitted-token normalization, use
        :attr:`target_gen_tokens` instead.
        """
        return self.prompt_length + self.gen_length

    @property
    def e2e_time(self) -> float:
        if self.complete_time is not None:
            return self.complete_time - self.submit_time
        return 0.0

    @property
    def e2e_time_normalized(self) -> float:
        if self.target_gen_tokens > 0:
            return self.e2e_time / self.target_gen_tokens
        return 0.0

    @property
    def total_prefill_time(self) -> float:
        return sum(b.prefill_duration for b in self.block_states)

    @property
    def total_kv_transfer_time(self) -> float:
        return sum(b.kv_transfer_duration for b in self.block_states)

    @property
    def total_kv_transfer_time_nonoverlapped(self) -> float:
        total = 0.0
        for block in self.block_states:
            if (
                block.kv_transfer_start_time is None
                or block.kv_transfer_end_time is None
            ):
                continue
            if block.prefill_end_time is None:
                total += block.kv_transfer_duration
                continue
            total += max(
                0.0,
                block.kv_transfer_end_time
                - max(block.kv_transfer_start_time, block.prefill_end_time),
            )
        return total

    @property
    def total_decode_time(self) -> float:
        return sum(b.decode_duration for b in self.block_states)

    @property
    def total_decode_time_normalized(self) -> float:
        if self.target_gen_tokens > 0:
            return self.total_decode_time / self.target_gen_tokens
        return 0.0

    @property
    def total_queue_wait_time(self) -> float:
        return self.total_prefill_queue_wait_time + self.total_decode_queue_wait_time

    @property
    def total_prefill_queue_wait_time(self) -> float:
        return sum(
            b.prefill_scheduler_wait_duration + b.prefill_queue_wait_duration
            for b in self.block_states
        )

    @property
    def total_decode_queue_wait_time(self) -> float:
        return sum(
            b.decode_scheduler_wait_duration + b.decode_queue_wait_duration
            for b in self.block_states
        )

    @property
    def first_block_queue_wait_time(self) -> float:
        if not self.block_states:
            return 0.0
        first = self.block_states[0]
        return first.prefill_scheduler_wait_duration + first.prefill_queue_wait_duration

    @property
    def e2e_time_excl_first_block_queue(self) -> float:
        return max(0.0, self.e2e_time - self.first_block_queue_wait_time)

    @property
    def e2e_time_excl_first_block_queue_normalized(self) -> float:
        if self.target_gen_tokens > 0:
            return self.e2e_time_excl_first_block_queue / self.target_gen_tokens
        return 0.0

    @property
    def execution_time(self) -> float:
        return max(0.0, self.e2e_time - self.total_queue_wait_time)

    @property
    def execution_time_normalized(self) -> float:
        if self.target_gen_tokens > 0:
            return self.execution_time / self.target_gen_tokens
        return 0.0

    @property
    def verification_component_sum(self) -> float:
        return (
            self.total_prefill_time
            + self.total_decode_time
            + self.total_queue_wait_time
            + self.total_kv_transfer_time_nonoverlapped
        )

    @property
    def unaccounted_time(self) -> float:
        return self.e2e_time - self.verification_component_sum

    def recompute_num_forward_evals(self) -> None:
        self.num_forward_evals = sum(
            block.prefill_forward_evals_applied + block.decode_forward_evals_applied
            for block in self.block_states
        )
