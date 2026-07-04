"""Backend that talks to fastdllm_serve's SchedulerService.

fastdllm_serve reuses sangam's `sangam.proto` wire format and
implements `Submit` / `Poll` / `Generate`. This backend uses the blocking
`Generate` RPC, which holds the call open until the request reaches a
terminal status, so the client does not need to poll. Everything else
(status, metrics, worker registration) is `UNIMPLEMENTED`, so warmup
falls back to `max_batch_size` and metrics RPCs are no-ops.
"""

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

logger = init_logger(__name__)


class FastDllmBackend(BenchmarkBackend):
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._stub: sangam_pb2_grpc.SchedulerServiceStub | None = None
        self._tokenizer = None
        self._prompt_token_ids_by_request_id: dict[int, list[int]] = {}
        self._rendered_prompt_by_request_id: dict[int, str] = {}

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
        return 1

    def reset_metrics(self) -> None:
        return

    def export_metrics(self) -> None:
        return

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
                    "FastDllmBackend tokenizer was not initialized for "
                    "messages requests."
                )
            prompt_token_ids = self._prompt_token_ids_by_request_id[request.id]

        gen_req = sangam_pb2.GenerateRequest(
            prompt_token_ids=prompt_token_ids,
            gen_length=request.gen_len,
            request_seed=request.request_seed,
        )
        # fastdllm_serve only supports unmasking_strategy='conf_threshold';
        # build the proto directly so we (a) always send conf_threshold and
        # (b) omit confidence_threshold when the user didn't set one, letting
        # the server fall back to its --default-threshold.
        gen_req.sampling_parameters.temperature = self.config.temperature
        gen_req.sampling_parameters.unmasking_strategy = "conf_threshold"
        if self.config.confidence_threshold is not None:
            gen_req.sampling_parameters.confidence_threshold = (
                self.config.confidence_threshold
            )
        return gen_req
