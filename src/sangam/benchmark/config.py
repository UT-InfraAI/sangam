"""Benchmark configuration.

Flat BenchmarkConfig with argparse builder. Constructs nested generator configs
that the sarathi-derived request generators expect.
"""

from __future__ import annotations

import argparse
import enum
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields

from sangam.config_utils import scheduler_address_from_port
from sangam.engine.launch_config import (
    EngineLaunchConfig,
    add_engine_launch_args,
    engine_launch_config_from_namespace,
)
from sangam.sampling_parameters import SamplingParameters


_BENCHMARK_ENGINE_LAUNCH_FIELD_EXCLUSIONS = frozenset(
    {
        "metrics_output_dir",
        "max_grpc_message_length",
        "kv_dtype",
        "kv_transfer_timeout_s",
        "streaming_layer_ready_timeout_s",
        "streaming_recv_join_timeout_s",
        "term_timeout_seconds",
        "kill_timeout_seconds",
        "poll_interval",
    }
)


def benchmark_engine_launch_field_names() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclass_fields(EngineLaunchConfig)
        if field.name not in _BENCHMARK_ENGINE_LAUNCH_FIELD_EXCLUSIONS
    )


# ---------------------------------------------------------------------------
# Generator type enums
# ---------------------------------------------------------------------------


class RequestIntervalGeneratorType(enum.Enum):
    POISSON = "poisson"
    GAMMA = "gamma"
    STATIC = "static"
    TRACE = "trace"


class RequestLengthGeneratorType(enum.Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    ZIPF = "zipf"
    TRACE = "trace"


class RequestGeneratorType(enum.Enum):
    SYNTHETIC = "synthetic"
    TRACE = "trace"


# ---------------------------------------------------------------------------
# Generator configs (interfaces expected by the sarathi-derived generators)
# ---------------------------------------------------------------------------


@dataclass
class BaseRequestIntervalGeneratorConfig:
    seed: int = 42


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = 1.0

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = 1.0
    cv: float = 0.5

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = ""
    start_time: str = "1970-01-04 12:00:00"
    end_time: str = "1970-01-04 15:00:00"
    time_scale_factor: float = 0.3

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class BaseRequestLengthGeneratorConfig:
    seed: int = 42


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = 512
    decode_tokens: int = 128

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = 128
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 4.0

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = 0.6
    scramble: bool = False
    min_tokens: int = 1024
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 20.0

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = ""
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0
    max_tokens: int | None = None
    model: str | None = None
    shuffle: bool = True

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class BaseRequestGeneratorConfig:
    seed: int = 42


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: int | None = 64
    duration: float | None = None

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = ""
    date: str = ""
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0
    time_scale_factor: float = 1.0
    max_tokens: int = 4096

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE


# ---------------------------------------------------------------------------
# Top-level benchmark config
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig(EngineLaunchConfig):
    # Engine launch fields are inherited from EngineLaunchConfig so benchmark,
    # capacity search, and server launch stay on the same parameter surface.
    metrics_output_dir: str = field(
        init=False,
        default=EngineLaunchConfig.metrics_output_dir,
    )

    # Server lifecycle
    backend: str = "sangam"  # "sangam" | "fast-dllm"
    launch_server: bool = True
    server_startup_timeout: float = 300.0
    benchmark_timeout: float | None = 1800.0

    # Connection
    scheduler_address: str = scheduler_address_from_port(
        EngineLaunchConfig.scheduler_port, "localhost"
    )

    # Generation params (fixed for the whole run)
    gen_length: int | None = None
    temperature: float = 0.0
    unmasking_strategy: str = "random"
    confidence_threshold: float | None = None
    fixed_unmask_quota: int | None = 2
    dynamic_unmask_factor: float | None = None

    # Workload
    request_generator_type: str = "synthetic"  # "synthetic" | "trace"
    num_requests: int | None = 100
    duration: float | None = None
    seed: int = 42

    # Interval generator
    interval_type: str = "poisson"  # "poisson" | "gamma" | "static"
    qps: float = 1.0
    cv: float = 0.5  # gamma only

    # Length generator
    length_type: str = "fixed"  # "fixed" | "uniform" | "zipf"
    prefill_tokens: int = 512
    decode_tokens: int = 128
    min_tokens: int = 128
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 4.0
    zipf_theta: float = 0.6
    zipf_scramble: bool = False

    # Trace generator (for request_generator_type=trace)
    trace_file: str | None = None
    trace_date: str | None = None
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0
    time_scale_factor: float = 1.0

    # Trace interval/length generators (for interval_type=trace or length_type=trace)
    interval_trace_file: str | None = None
    interval_trace_start_time: str = "1970-01-04 12:00:00"
    interval_trace_end_time: str = "1970-01-04 15:00:00"
    interval_trace_time_scale_factor: float = 0.3
    length_trace_file: str | None = None
    length_trace_prefill_scale_factor: float = 1.0
    length_trace_decode_scale_factor: float = 1.0
    length_trace_max_tokens: int | None = None
    length_trace_shuffle: bool = True

    # Warmup
    num_warmup_requests: int | None = None  # None = auto (total worker count)

    # Output
    output_dir: str = "benchmark_output"
    poll_interval: float = 0.05

    # Cross-backend server log level. Forwarded only to backends whose CLI
    # accepts it (fastdllm_serve's --log-level today); silently ignored for
    # sangam.
    log_level: str | None = None

    # fast-dllm-specific (used only when backend == "fast-dllm")
    fast_dllm_default_threshold: float | None = None
    fast_dllm_grpc_workers: int | None = None
    fast_dllm_max_pending: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("sangam", "fast-dllm"):
            raise ValueError(
                f"Unknown backend '{self.backend}'. Use 'sangam' or 'fast-dllm'."
            )
        if (
            self.backend == "fast-dllm"
            and self.unmasking_strategy != "random"
            and self.unmasking_strategy != "conf_threshold"
        ):
            raise ValueError(
                "backend='fast-dllm' only supports unmasking_strategy="
                f"'conf_threshold'; got {self.unmasking_strategy!r}."
            )

    def build_engine_launch_config(self, metrics_output_dir: str) -> EngineLaunchConfig:
        values = {
            name: getattr(self, name) for name in benchmark_engine_launch_field_names()
        }
        values["metrics_output_dir"] = metrics_output_dir
        return EngineLaunchConfig(**values)

    def fast_dllm_gpu_csv(self) -> str:
        """GPU CSV to expose to fastdllm_serve via CUDA_VISIBLE_DEVICES."""

        def _clean(csv: str) -> str:
            return ",".join(gpu.strip() for gpu in csv.split(",") if gpu.strip())

        if self.mode == "colocated":
            return _clean(self.gpus)
        parts = [_clean(self.prefill_gpus), _clean(self.hybrid_colocated_gpus)]
        return ",".join(part for part in parts if part)

    def build_fast_dllm_launch_args(self) -> list[str]:
        """CLI args for `python -m fastdllm_serve`.

        The server infers its model backend (llada vs dream) from `--model`
        itself, so we do not pass `--backend`.
        """

        args = [
            "--model",
            self.model,
            "--port",
            str(self.scheduler_port),
            "--max-batch-size",
            str(self.max_batch_size),
            "--block-length",
            str(self.block_length),
        ]
        if self.fast_dllm_default_threshold is not None:
            args += ["--default-threshold", str(self.fast_dllm_default_threshold)]
        if self.fast_dllm_grpc_workers is not None:
            args += ["--grpc-workers", str(self.fast_dllm_grpc_workers)]
        if self.fast_dllm_max_pending is not None:
            args += ["--max-pending", str(self.fast_dllm_max_pending)]
        if self.max_gen_len is not None:
            args += ["--max-gen-len", str(self.max_gen_len)]
        if self.log_level is not None:
            args += ["--log-level", self.log_level]
        return args

    @property
    def sampling_parameters(self) -> SamplingParameters:
        return SamplingParameters(
            temperature=self.temperature,
            unmasking_strategy=self.unmasking_strategy,
            confidence_threshold=self.confidence_threshold,
            fixed_unmask_quota=self.fixed_unmask_quota,
            dynamic_unmask_factor=self.dynamic_unmask_factor,
        )

    def build_request_generator_config(
        self,
    ) -> SyntheticRequestGeneratorConfig | TraceRequestGeneratorConfig:
        if self.request_generator_type == "trace":
            return TraceRequestGeneratorConfig(
                seed=self.seed,
                trace_file=self.trace_file or "",
                date=self.trace_date or "",
                prefill_scale_factor=self.prefill_scale_factor,
                decode_scale_factor=self.decode_scale_factor,
                time_scale_factor=self.time_scale_factor,
                max_tokens=self.max_tokens,
            )

        return SyntheticRequestGeneratorConfig(
            seed=self.seed,
            num_requests=self.num_requests,
            duration=self.duration,
            length_generator_config=self._build_length_config(),
            interval_generator_config=self._build_interval_config(),
        )

    def _build_interval_config(self) -> BaseRequestIntervalGeneratorConfig:
        if self.interval_type == "poisson":
            return PoissonRequestIntervalGeneratorConfig(seed=self.seed, qps=self.qps)
        elif self.interval_type == "gamma":
            return GammaRequestIntervalGeneratorConfig(
                seed=self.seed, qps=self.qps, cv=self.cv
            )
        elif self.interval_type == "static":
            return StaticRequestIntervalGeneratorConfig(seed=self.seed)
        elif self.interval_type == "trace":
            return TraceRequestIntervalGeneratorConfig(
                seed=self.seed,
                trace_file=self.interval_trace_file or "",
                start_time=self.interval_trace_start_time,
                end_time=self.interval_trace_end_time,
                time_scale_factor=self.interval_trace_time_scale_factor,
            )
        else:
            raise ValueError(f"Unknown interval type: {self.interval_type}")

    def _build_length_config(self) -> BaseRequestLengthGeneratorConfig:
        if self.length_type == "fixed":
            return FixedRequestLengthGeneratorConfig(
                seed=self.seed,
                prefill_tokens=self.prefill_tokens,
                decode_tokens=self.decode_tokens,
            )
        elif self.length_type == "uniform":
            return UniformRequestLengthGeneratorConfig(
                seed=self.seed,
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                prefill_to_decode_ratio=self.prefill_to_decode_ratio,
            )
        elif self.length_type == "zipf":
            return ZipfRequestLengthGeneratorConfig(
                seed=self.seed,
                theta=self.zipf_theta,
                scramble=self.zipf_scramble,
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                prefill_to_decode_ratio=self.prefill_to_decode_ratio,
            )
        elif self.length_type == "trace":
            return TraceRequestLengthGeneratorConfig(
                seed=self.seed,
                trace_file=self.length_trace_file or "",
                prefill_scale_factor=self.length_trace_prefill_scale_factor,
                decode_scale_factor=self.length_trace_decode_scale_factor,
                max_tokens=self.length_trace_max_tokens,
                model=self.model,
                shuffle=self.length_trace_shuffle,
            )
        else:
            raise ValueError(f"Unknown length type: {self.length_type}")

    def to_dict(self) -> dict:
        from dataclasses import asdict

        data = asdict(self)
        if "kv_dtype" in data:
            data["kv_dtype"] = str(data["kv_dtype"])
        return data

    @staticmethod
    def create_from_cli_args() -> BenchmarkConfig:
        parser = argparse.ArgumentParser(description="Benchmark harness for sangam")

        # Server lifecycle
        parser.add_argument(
            "--backend",
            type=str,
            default="sangam",
            choices=["sangam", "fast-dllm"],
            help="Inference server to target (default: sangam)",
        )
        parser.add_argument(
            "--fast-dllm-default-threshold",
            type=float,
            default=None,
            help="fastdllm_serve --default-threshold (only when --backend=fast-dllm)",
        )
        parser.add_argument(
            "--fast-dllm-grpc-workers",
            type=int,
            default=None,
            help="fastdllm_serve --grpc-workers (only when --backend=fast-dllm)",
        )
        parser.add_argument(
            "--fast-dllm-max-pending",
            type=int,
            default=None,
            help="fastdllm_serve --max-pending (only when --backend=fast-dllm)",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=None,
            help="Server log level forwarded to backends that accept it "
            "(fastdllm_serve --log-level today; ignored for sangam).",
        )
        parser.add_argument(
            "--launch-server",
            action="store_true",
            default=True,
            help="Auto-start and stop the server (default: True)",
        )
        parser.add_argument(
            "--no-launch-server",
            action="store_false",
            dest="launch_server",
            help="Connect to an existing server instead of launching one",
        )
        add_engine_launch_args(parser, include_metrics_output_dir=False)
        parser.add_argument(
            "--server-startup-timeout",
            type=float,
            default=300.0,
        )
        parser.add_argument(
            "--benchmark-timeout",
            type=float,
            default=1800.0,
            help="Wall-clock cap (seconds) for the whole benchmark run "
            "(server startup + warmup + main loop + teardown). "
            "On expiry the benchmark logs a warning, tears down the server, "
            "and exits with code 2; pair with --export-partial-metrics to "
            "flush metrics collected so far. Default: 1800 (30 min). "
            "Set to 0 or negative to disable.",
        )

        # Connection
        parser.add_argument(
            "--scheduler-address",
            type=str,
            default=scheduler_address_from_port(
                EngineLaunchConfig.scheduler_port, "localhost"
            ),
        )

        # Generation params
        parser.add_argument("--gen-length", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument(
            "--unmasking_strategy",
            type=str,
            default="random",
            choices=["random", "conf_threshold", "conf_quota", "conf_dynamic"],
        )
        parser.add_argument("--confidence_threshold", type=float, default=None)
        parser.add_argument("--fixed_unmask_quota", type=int, default=2)
        parser.add_argument("--dynamic_unmask_factor", type=float, default=None)

        # Workload
        parser.add_argument(
            "--request-generator-type",
            type=str,
            default="synthetic",
            choices=["synthetic", "trace"],
        )
        parser.add_argument("--num-requests", type=int, default=100)
        parser.add_argument("--duration", type=float, default=None)
        parser.add_argument("--seed", type=int, default=42)

        # Interval generator
        parser.add_argument(
            "--interval-type",
            type=str,
            default="poisson",
            choices=["poisson", "gamma", "static", "trace"],
        )
        parser.add_argument("--qps", type=float, default=1.0)
        parser.add_argument("--cv", type=float, default=0.5)

        # Length generator
        parser.add_argument(
            "--length-type",
            type=str,
            default="fixed",
            choices=["fixed", "uniform", "zipf", "trace"],
        )
        parser.add_argument("--prefill-tokens", type=int, default=512)
        parser.add_argument("--decode-tokens", type=int, default=128)
        parser.add_argument("--min-tokens", type=int, default=128)
        parser.add_argument("--max-tokens", type=int, default=4096)
        parser.add_argument("--prefill-to-decode-ratio", type=float, default=4.0)
        parser.add_argument("--zipf-theta", type=float, default=0.6)
        parser.add_argument("--zipf-scramble", action="store_true")

        # Trace request generator
        parser.add_argument("--trace-file", type=str, default=None)
        parser.add_argument("--trace-date", type=str, default=None)
        parser.add_argument("--prefill-scale-factor", type=float, default=1.0)
        parser.add_argument("--decode-scale-factor", type=float, default=1.0)
        parser.add_argument("--time-scale-factor", type=float, default=1.0)

        # Trace interval generator
        parser.add_argument("--interval-trace-file", type=str, default=None)
        parser.add_argument(
            "--interval-trace-start-time",
            type=str,
            default="1970-01-04 12:00:00",
        )
        parser.add_argument(
            "--interval-trace-end-time",
            type=str,
            default="1970-01-04 15:00:00",
        )
        parser.add_argument(
            "--interval-trace-time-scale-factor", type=float, default=0.3
        )

        # Trace length generator
        parser.add_argument("--length-trace-file", type=str, default=None)
        parser.add_argument(
            "--length-trace-prefill-scale-factor", type=float, default=1.0
        )
        parser.add_argument(
            "--length-trace-decode-scale-factor", type=float, default=1.0
        )
        parser.add_argument("--length-trace-max-tokens", type=int, default=None)
        parser.add_argument(
            "--length-trace-shuffle",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Shuffle the length trace CSV rows (default: True). "
            "Pass --no-length-trace-shuffle to preserve CSV order.",
        )

        # Warmup
        parser.add_argument(
            "--num-warmup-requests",
            type=int,
            default=None,
            help="Number of warmup requests before benchmarking "
            "(default: total worker count)",
        )

        # Output
        parser.add_argument("--output-dir", type=str, default="benchmark_output")
        parser.add_argument("--poll-interval", type=float, default=0.05)

        args = parser.parse_args()

        # Derive scheduler_address from port when launching server
        scheduler_address = args.scheduler_address
        if args.launch_server:
            scheduler_address = scheduler_address_from_port(
                args.scheduler_port, args.master_addr
            )
        launch_config = engine_launch_config_from_namespace(args)

        launch_values = {
            name: getattr(launch_config, name)
            for name in benchmark_engine_launch_field_names()
        }

        return BenchmarkConfig(
            **launch_values,
            backend=args.backend,
            fast_dllm_default_threshold=args.fast_dllm_default_threshold,
            fast_dllm_grpc_workers=args.fast_dllm_grpc_workers,
            fast_dllm_max_pending=args.fast_dllm_max_pending,
            log_level=args.log_level,
            launch_server=args.launch_server,
            server_startup_timeout=args.server_startup_timeout,
            benchmark_timeout=(
                args.benchmark_timeout
                if args.benchmark_timeout is not None and args.benchmark_timeout > 0
                else None
            ),
            scheduler_address=scheduler_address,
            gen_length=args.gen_length,
            temperature=args.temperature,
            unmasking_strategy=args.unmasking_strategy,
            confidence_threshold=args.confidence_threshold,
            fixed_unmask_quota=args.fixed_unmask_quota,
            dynamic_unmask_factor=args.dynamic_unmask_factor,
            request_generator_type=args.request_generator_type,
            num_requests=args.num_requests,
            duration=args.duration,
            seed=args.seed,
            interval_type=args.interval_type,
            qps=args.qps,
            cv=args.cv,
            length_type=args.length_type,
            prefill_tokens=args.prefill_tokens,
            decode_tokens=args.decode_tokens,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            prefill_to_decode_ratio=args.prefill_to_decode_ratio,
            zipf_theta=args.zipf_theta,
            zipf_scramble=args.zipf_scramble,
            trace_file=args.trace_file,
            trace_date=args.trace_date,
            prefill_scale_factor=args.prefill_scale_factor,
            decode_scale_factor=args.decode_scale_factor,
            time_scale_factor=args.time_scale_factor,
            interval_trace_file=args.interval_trace_file,
            interval_trace_start_time=args.interval_trace_start_time,
            interval_trace_end_time=args.interval_trace_end_time,
            interval_trace_time_scale_factor=args.interval_trace_time_scale_factor,
            length_trace_file=args.length_trace_file,
            length_trace_prefill_scale_factor=args.length_trace_prefill_scale_factor,
            length_trace_decode_scale_factor=args.length_trace_decode_scale_factor,
            length_trace_max_tokens=args.length_trace_max_tokens,
            length_trace_shuffle=args.length_trace_shuffle,
            num_warmup_requests=args.num_warmup_requests,
            output_dir=args.output_dir,
            poll_interval=args.poll_interval,
        )
