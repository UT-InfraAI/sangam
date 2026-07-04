"""Benchmark runner: generates requests, drives a backend, collects metrics."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from tqdm import tqdm
from transformers import AutoTokenizer

from sangam.benchmark.backends import BenchmarkBackend, RequestResult
from sangam.benchmark.config import BenchmarkConfig
from sangam.benchmark.entities import Request
from sangam.benchmark.request_generator import RequestGeneratorRegistry
from sangam.benchmark.utils.random import set_seeds
from sangam.logger import init_logger

logger = init_logger(__name__)

# Re-exported so external imports of `RequestResult` from this module continue
# to work after the dataclass moved to backends.base.
__all__ = ["BenchmarkRunner", "RequestResult"]


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BenchmarkBackend | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        if backend is None:
            from sangam.benchmark.backends import SangamBackend

            backend = SangamBackend(config)
        self.backend = backend
        self._on_abort = on_abort
        self._abort_invoked = False
        self._prompt_token_ids_by_request_id: dict[int, list[int]] = {}
        self._rendered_prompt_by_request_id: dict[int, str] = {}

        set_seeds(config.seed)

        # Generate requests
        gen_config = config.build_request_generator_config()
        generator = RequestGeneratorRegistry.get(gen_config.get_type(), gen_config)
        generated_requests = generator.generate()
        self.requests = self._normalize_requests_for_block_length(generated_requests)

        needs_tokenizer = any(request.messages is not None for request in self.requests)
        self.tokenizer = (
            AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
            if needs_tokenizer
            else None
        )
        if self.tokenizer is not None:
            self.requests = self._align_messages_requests(self.requests)

        backend.validate_requests(self.requests)
        backend.prepare(
            self.tokenizer,
            self._prompt_token_ids_by_request_id,
            self._rendered_prompt_by_request_id,
        )

    # ----- back-compat shims for the previous single-class API -----
    # Pre-refactor, BenchmarkRunner owned the gRPC submit/poll path directly.
    # Existing tests still call into those internals; forward them to the
    # backend so the sangam test suite keeps passing.

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value
        # Re-attach so the backend sees the new tokenizer.
        self.backend.prepare(
            value,
            self._prompt_token_ids_by_request_id,
            self._rendered_prompt_by_request_id,
        )

    def _build_grpc_request(self, request: Request):
        # Only meaningful for the sangam backend; older tests rely on this.
        return self.backend._build_grpc_request(request)  # type: ignore[attr-defined]

    def _submit_and_poll(self, stub, request: Request, pbar):
        # Older test signature passes the stub explicitly. Inject it into
        # the backend so existing tests keep working.
        self.backend._stub = stub  # type: ignore[attr-defined]
        return self.backend.submit_and_poll(request, pbar)

    def export_scheduler_metrics(self) -> None:
        """Back-compat wrapper around `self.backend.export_metrics()`."""
        self.backend.export_metrics()

        logger.info(
            f"Generated {len(self.requests)} requests "
            f"(type={self.config.request_generator_type}, "
            f"interval={self.config.interval_type}, "
            f"length={self.config.length_type})"
        )
        if self.requests:
            span = self.requests[-1].arrived_at - self.requests[0].arrived_at
            logger.info(
                f"Arrival span: {span:.2f}s, "
                f"effective QPS: {len(self.requests) / max(span, 1e-9):.2f}"
            )

    def _normalize_requests_for_block_length(
        self,
        requests: list[Request],
    ) -> list[Request]:
        block_length = self.config.block_length
        if block_length <= 0:
            raise ValueError(f"block_length must be > 0, got {block_length}")

        adjusted_count = 0
        normalized_requests: list[Request] = []
        for request in requests:
            normalized_gen_len = request.gen_len
            if request.gen_len % block_length != 0:
                adjusted_count += 1
                normalized_gen_len = (
                    (request.gen_len + block_length - 1) // block_length
                ) * block_length

            normalized_requests.append(
                Request(
                    arrived_at=request.arrived_at,
                    prompt_len=request.prompt_len,
                    gen_len=normalized_gen_len,
                    messages=request.messages,
                    request_seed=request.request_seed,
                    external_id=request.external_id,
                )
            )

        if adjusted_count:
            logger.warning(
                "Adjusted gen_len for %d/%d benchmark requests to "
                "be divisible by block_length=%d",
                adjusted_count,
                len(requests),
                block_length,
            )

        return normalized_requests

    def _align_messages_requests(
        self,
        requests: list[Request],
    ) -> list[Request]:
        if getattr(self.tokenizer, "chat_template", None) in (None, ""):
            raise ValueError(
                f"Tokenizer for model {self.config.model!r} has no chat_template; "
                "trace requests with `messages` require an instruct tokenizer."
            )

        aligned_requests: list[Request] = []
        adjusted_count = 0

        for request in requests:
            if request.messages is None:
                aligned_requests.append(request)
                continue

            prompt_token_ids = self.tokenizer.apply_chat_template(
                request.messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            rendered_prompt = self.tokenizer.apply_chat_template(
                request.messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_len = len(prompt_token_ids)
            if prompt_len != request.prompt_len:
                adjusted_count += 1

            aligned_requests.append(
                Request(
                    arrived_at=request.arrived_at,
                    prompt_len=prompt_len,
                    gen_len=request.gen_len,
                    messages=request.messages,
                    request_seed=request.request_seed,
                    external_id=request.external_id,
                )
            )
            new_id = aligned_requests[-1].id
            self._prompt_token_ids_by_request_id[new_id] = prompt_token_ids
            self._rendered_prompt_by_request_id[new_id] = rendered_prompt

        if adjusted_count:
            logger.warning(
                "Adjusted prompt_len for %d benchmark requests to match the "
                "tokenized chat-templated messages.",
                adjusted_count,
            )

        return aligned_requests

    def _invoke_abort(self) -> None:
        if self._abort_invoked or self._on_abort is None:
            return
        self._abort_invoked = True
        try:
            self._on_abort()
        except Exception as e:
            logger.warning(f"Benchmark abort hook raised: {e}")

    def run(self) -> list[RequestResult]:
        if not self.requests:
            logger.warning("No requests to run")
            return []

        self.backend.connect()

        # Warmup: submit multiple requests concurrently to exercise all workers
        num_warmup = self.backend.resolve_num_warmup_requests(
            self.config.num_warmup_requests
        )
        num_warmup = min(num_warmup, len(self.requests))
        logger.info(f"Warming up with {num_warmup} request(s)...")
        warmup_pbar = tqdm(total=num_warmup, desc="Warmup")
        warmup_errors: list[str] = []
        warmup_pool = ThreadPoolExecutor(max_workers=num_warmup)
        try:
            futures = [
                warmup_pool.submit(
                    self.backend.submit_and_poll, self.requests[i], warmup_pbar
                )
                for i in range(num_warmup)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result.error:
                    warmup_errors.append(result.error)
            warmup_pool.shutdown(wait=True)
        except BaseException:
            self._invoke_abort()
            warmup_pool.shutdown(wait=False, cancel_futures=True)
            raise
        warmup_pbar.close()
        if warmup_errors:
            logger.error(
                f"Warmup failed ({len(warmup_errors)} error(s)): {warmup_errors[0]}"
            )
            return []
        logger.info("Warmup done")
        self.backend.reset_metrics()

        # Main run: submit all requests respecting arrival times
        results: list[RequestResult] = []
        pbar = tqdm(total=len(self.requests), desc="Completed requests")

        run_start = time.monotonic()

        executor = ThreadPoolExecutor(max_workers=len(self.requests))
        try:
            futures = {}

            for i, request in enumerate(self.requests):
                # Wait until arrival time
                target = run_start + request.arrived_at
                now = time.monotonic()
                if target > now:
                    time.sleep(target - now)

                future = executor.submit(self.backend.submit_and_poll, request, pbar)
                futures[future] = i

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
            executor.shutdown(wait=True)
        except BaseException:
            self._invoke_abort()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        pbar.close()

        # Sort by submit time
        results.sort(key=lambda r: r.submit_time)

        return results

    def save_results(self, results: list[RequestResult]) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)

        results_path = os.path.join(self.config.output_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "config": self.config.to_dict(),
                    "results": [asdict(r) for r in results],
                    "summary": _compute_summary(results),
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {results_path}")


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def _compute_summary(results: list[RequestResult]) -> dict:
    successful = [r for r in results if r.error is None]
    if not successful:
        return {"num_successful": 0, "num_failed": len(results)}

    latencies = sorted(r.latency for r in successful)
    total_gen_tokens = sum(r.gen_tokens for r in successful)
    total_time = max(r.complete_time for r in successful) - min(
        r.submit_time for r in successful
    )

    return {
        "num_successful": len(successful),
        "num_failed": len(results) - len(successful),
        "latency_min": latencies[0],
        "latency_p50": _percentile(latencies, 50),
        "latency_p90": _percentile(latencies, 90),
        "latency_p99": _percentile(latencies, 99),
        "latency_max": latencies[-1],
        "latency_mean": sum(latencies) / len(latencies),
        "total_gen_tokens": total_gen_tokens,
        "wall_clock_time": total_time,
        "tokens_per_sec": total_gen_tokens / total_time if total_time > 0 else 0,
        "requests_per_sec": len(successful) / total_time if total_time > 0 else 0,
    }
