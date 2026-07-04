"""Backend abstraction for the benchmark client.

A backend owns the wire-format details (which gRPC service, sync vs async,
metrics RPCs) for a given inference server. The benchmark runner is
backend-agnostic: it generates requests and asks the backend to execute
each one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tqdm import tqdm

from sangam.benchmark.entities import Request


@dataclass
class RequestResult:
    request_id: str
    prompt_len: int
    gen_len: int
    arrived_at: float
    submit_time: float
    complete_time: float
    latency: float
    num_forward_evals: int
    gen_tokens: int
    # Completion/output tokens per second, excluding prompt tokens.
    tokens_per_sec: float
    rendered_prompt: str | None = None
    generated_text: str | None = None
    error: str | None = None
    # Server-reported timestamps (None when the server did not report them).
    # sangam reports wall clock; fastdllm_serve reports monotonic seconds.
    # Use as deltas (start - arrival = queue, complete - start = execute);
    # absolute values are not comparable across backends or with client time.
    server_arrival_time: float | None = None
    server_start_time: float | None = None
    server_complete_time: float | None = None


class BenchmarkBackend(ABC):
    """Abstract benchmark backend. One instance per benchmark run."""

    @abstractmethod
    def connect(self) -> None:
        """Open any persistent connections to the inference server."""

    def prepare(
        self,
        tokenizer,
        prompt_token_ids_by_request_id: dict[int, list[int]],
        rendered_prompt_by_request_id: dict[int, str],
    ) -> None:
        """Hand off tokenizer and tokenized-prompt cache built by the runner.

        Default no-op. Backends that need to send token IDs or decode/count
        output tokens override this.
        """

    @abstractmethod
    def validate_requests(self, requests: list[Request]) -> None:
        """Reject requests this backend cannot serve. Raise ValueError on failure."""

    @abstractmethod
    def resolve_num_warmup_requests(self, configured: int | None) -> int:
        """Pick a warmup request count when the user did not set one explicitly."""

    @abstractmethod
    def submit_and_poll(self, request: Request, pbar: tqdm) -> RequestResult:
        """Execute one request end-to-end and return its result."""

    def reset_metrics(self) -> None:
        """Reset server-side metrics after warmup. Default: no-op."""

    def export_metrics(self) -> None:
        """Flush server-side metrics to disk. Default: no-op."""
