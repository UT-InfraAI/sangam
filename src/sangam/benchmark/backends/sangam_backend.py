"""Backend that talks to sangam's SchedulerService."""

from __future__ import annotations

import time

import grpc
from tqdm import tqdm

from sangam.benchmark.backends.base import BenchmarkBackend, RequestResult
from sangam.benchmark.config import BenchmarkConfig
from sangam.benchmark.entities import Request
from sangam.grpc_utils import grpc_message_length_options
from sangam.logger import init_logger
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.sampling_parameters import SamplingParameters

logger = init_logger(__name__)
PROMPT_TEXT_DEFAULT_CONFIDENCE_THRESHOLD = 0.9


class SangamBackend(BenchmarkBackend):
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._stub: sangam_pb2_grpc.SchedulerServiceStub | None = None
        self._tokenizer = None
        self._prompt_token_ids_by_request_id: dict[int, list[int]] = {}
        self._rendered_prompt_by_request_id: dict[int, str] = {}
        self._warned_prompt_text_sampling_default = False

    def connect(self) -> None:
        channel = grpc.insecure_channel(
            self.config.scheduler_address,
            options=grpc_message_length_options(self.config.max_grpc_message_length),
        )
        self._stub = sangam_pb2_grpc.SchedulerServiceStub(channel)

    def prepare(
        self,
        tokenizer,
        prompt_token_ids_by_request_id: dict[int, list[int]],
        rendered_prompt_by_request_id: dict[int, str],
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_token_ids_by_request_id = prompt_token_ids_by_request_id
        self._rendered_prompt_by_request_id = rendered_prompt_by_request_id

    def validate_requests(self, requests: list[Request]) -> None:
        return

    def resolve_num_warmup_requests(self, configured: int | None) -> int:
        if configured is not None:
            return configured
        try:
            status = self._stub.GetSchedulerStatus(
                sangam_pb2.GetSchedulerStatusRequest()
            )
            worker_counts = (
                _coerce_worker_count(status.num_prefill_workers),
                _coerce_worker_count(status.num_decode_workers),
                _coerce_worker_count(status.num_colocated_workers),
            )
            if any(count is None for count in worker_counts):
                raise TypeError("Scheduler status returned non-integer worker counts")
            total = sum(worker_counts)
            return max(total, 1)
        except (grpc.RpcError, TypeError) as e:
            logger.warning(f"Failed to query scheduler status for warmup count: {e}")
            return 1

    def reset_metrics(self) -> None:
        try:
            resp = self._stub.ResetMetrics(sangam_pb2.ResetMetricsRequest())
        except grpc.RpcError as e:
            logger.warning(f"Failed to reset scheduler metrics after warmup: {e}")
            return
        if not resp.success:
            logger.warning(
                "Scheduler metrics reset returned success=False after warmup"
            )

    def export_metrics(self) -> None:
        # Open a fresh channel — the original code did this too, to survive any
        # partial close after the run.
        channel = grpc.insecure_channel(
            self.config.scheduler_address,
            options=grpc_message_length_options(self.config.max_grpc_message_length),
        )
        stub = sangam_pb2_grpc.SchedulerServiceStub(channel)
        try:
            resp = stub.ExportMetrics(sangam_pb2.ExportMetricsRequest())
            if resp.success:
                logger.info(f"Scheduler metrics exported to {resp.output_dir}")
            else:
                logger.warning("Scheduler metrics export returned success=False")
        except grpc.RpcError as e:
            logger.warning(f"Failed to export scheduler metrics: {e}")

    def submit_and_poll(self, request: Request, pbar: tqdm) -> RequestResult:
        submit_time = time.time()

        try:
            resp = self._stub.Generate(self._build_grpc_request(request))
        except grpc.RpcError as e:
            return RequestResult(
                request_id=request.external_id or "",
                prompt_len=request.prompt_len,
                gen_len=request.gen_len,
                arrived_at=request.arrived_at,
                submit_time=submit_time,
                complete_time=time.time(),
                latency=0,
                num_forward_evals=0,
                gen_tokens=0,
                tokens_per_sec=0,
                rendered_prompt=self._rendered_prompt_by_request_id.get(request.id),
                error=str(e),
            )

        request_id = request.external_id or resp.request_id

        if resp.status == "COMPLETED":
            complete_time = time.time()
            latency = complete_time - submit_time
            output_ids = list(resp.output_token_ids)
            gen_tokens = max(len(output_ids) - request.prompt_len, 0)
            generated_text = None
            if request.messages is not None and self._tokenizer is not None:
                generated_text = self._tokenizer.decode(
                    output_ids[request.prompt_len :],
                    skip_special_tokens=True,
                )

            pbar.update(1)
            return RequestResult(
                request_id=request_id,
                prompt_len=request.prompt_len,
                gen_len=request.gen_len,
                arrived_at=request.arrived_at,
                submit_time=submit_time,
                complete_time=complete_time,
                latency=latency,
                num_forward_evals=resp.num_forward_evals,
                gen_tokens=gen_tokens,
                tokens_per_sec=gen_tokens / latency if latency > 0 else 0,
                rendered_prompt=self._rendered_prompt_by_request_id.get(request.id),
                generated_text=generated_text,
                server_arrival_time=(
                    resp.server_arrival_time
                    if resp.HasField("server_arrival_time")
                    else None
                ),
                server_start_time=(
                    resp.server_start_time
                    if resp.HasField("server_start_time")
                    else None
                ),
                server_complete_time=(
                    resp.server_complete_time
                    if resp.HasField("server_complete_time")
                    else None
                ),
            )

        pbar.update(1)
        return RequestResult(
            request_id=request_id,
            prompt_len=request.prompt_len,
            gen_len=request.gen_len,
            arrived_at=request.arrived_at,
            submit_time=submit_time,
            complete_time=time.time(),
            latency=0,
            num_forward_evals=0,
            gen_tokens=0,
            tokens_per_sec=0,
            rendered_prompt=self._rendered_prompt_by_request_id.get(request.id),
            error=resp.error_message or f"unexpected status {resp.status!r}",
        )

    def _build_grpc_request(self, request: Request) -> sangam_pb2.GenerateRequest:
        prompt_token_ids = [1] * request.prompt_len
        if request.messages is not None:
            if self._tokenizer is None:
                raise ValueError(
                    "SangamBackend tokenizer was not initialized for messages requests."
                )
            prompt_token_ids = self._prompt_token_ids_by_request_id[request.id]

        gen_req = sangam_pb2.GenerateRequest(
            prompt_token_ids=prompt_token_ids,
            gen_length=request.gen_len,
            request_seed=request.request_seed,
        )
        gen_req.sampling_parameters.CopyFrom(
            self._sampling_parameters_for_request(request).to_proto()
        )
        return gen_req

    def _sampling_parameters_for_request(self, request: Request) -> SamplingParameters:
        if not self._should_default_to_conf_threshold(request):
            return self.config.sampling_parameters

        if not self._warned_prompt_text_sampling_default:
            logger.warning(
                "Trace requests include messages; defaulting benchmark sampling "
                "to unmasking_strategy='conf_threshold' and confidence_threshold=%.1f. "
                "Set benchmark sampling flags explicitly to override this.",
                PROMPT_TEXT_DEFAULT_CONFIDENCE_THRESHOLD,
            )
            self._warned_prompt_text_sampling_default = True

        return SamplingParameters(
            temperature=self.config.temperature,
            unmasking_strategy="conf_threshold",
            confidence_threshold=PROMPT_TEXT_DEFAULT_CONFIDENCE_THRESHOLD,
            fixed_unmask_quota=self.config.fixed_unmask_quota,
            dynamic_unmask_factor=self.config.dynamic_unmask_factor,
        )

    def _should_default_to_conf_threshold(self, request: Request) -> bool:
        return (
            request.messages is not None
            and self.config.unmasking_strategy == "random"
            and self.config.confidence_threshold is None
        )


def _coerce_worker_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
