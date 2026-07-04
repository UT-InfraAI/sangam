"""Shared engine launch configuration and CLI helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.engine.scheduler_config import (
    ColocatedSchedulerConfig,
    HybridSchedulerConfig,
)
from sangam.types import DecodeSchedulerPolicy, PrefillSchedulerPolicy
from sangam.worker.worker_config import (
    ColocatedWorkerConfig,
    PrefillWorkerConfig,
)


DEFAULT_CUDA_GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
DEFAULT_CUDA_GRAPH_BATCH_SIZES_SPEC = ",".join(
    str(b) for b in DEFAULT_CUDA_GRAPH_BATCH_SIZES
)


@dataclass
class EngineLaunchConfig:
    mode: str = "colocated"
    model: str = "GSAI-ML/LLaDA-8B-Instruct"
    scheduler_port: int = 50051
    gpus: str = "0,1"
    prefill_gpus: str = "0,1"
    hybrid_colocated_gpus: str = "1,2,3"
    base_worker_port: int = 20100
    master_addr: str = "localhost"
    master_port: int = 29500
    max_batch_size: int = 128
    max_tokens_per_iteration: int = 4096
    max_prefill_tokens_per_batch: int = 4096
    kv_page_size: int = 16
    kv_max_pages: int | None = None
    metrics_output_dir: str = "benchmark_output"
    disable_metrics: bool = False
    enable_individual_batch_metrics: bool = True
    enable_operation_metrics: bool = False
    op_metrics_layer_id: int | None = None
    export_partial_metrics: bool = False
    prefill_scheduler_policy: str = "least_outstanding_prefill_tokens"
    decode_grouping_slack_ratio: float = 0.10
    prefill_queue_policy: str = "arrival_order"
    decode_scheduler_policy: str = "max_free_memory"
    kv_fast_pairs: str = "0-1,2-3,4-5,6-7"
    kv_topology_alpha: float = 0.0
    prefill_overload_threshold: int = 16384
    enable_hybrid_prefill_overflow: bool = False
    block_length: int = 32
    mask_id: int | None = None
    max_gen_len: int | None = None
    colocated_sticky_worker: bool = False
    enable_cuda_graphs: bool = True
    cuda_graph_batch_sizes: str = DEFAULT_CUDA_GRAPH_BATCH_SIZES_SPEC
    # Internal-only tunables. Not exposed via CLI; propagated to scheduler /
    # worker / process-lifecycle configs through the build_* factories below.
    max_grpc_message_length: int = DEFAULT_MAX_GRPC_MESSAGE_LENGTH
    kv_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)
    kv_transfer_timeout_s: float = 30.0
    streaming_layer_ready_timeout_s: float = 30.0
    streaming_recv_join_timeout_s: float = 30.0
    term_timeout_seconds: float = 10.0
    kill_timeout_seconds: float = 5.0
    poll_interval: float = 0.1

    def to_cli_args(self) -> list[str]:
        args = [
            "--mode",
            self.mode,
            "--model",
            self.model,
            "--scheduler-port",
            str(self.scheduler_port),
            "--base-worker-port",
            str(self.base_worker_port),
            "--master-addr",
            self.master_addr,
            "--master-port",
            str(self.master_port),
            "--max-batch-size",
            str(self.max_batch_size),
            "--max-tokens-per-iteration",
            str(self.max_tokens_per_iteration),
            "--max-prefill-tokens-per-batch",
            str(self.max_prefill_tokens_per_batch),
            "--kv-page-size",
            str(self.kv_page_size),
            "--metrics-output-dir",
            self.metrics_output_dir,
            "--prefill-scheduler-policy",
            self.prefill_scheduler_policy,
            "--decode-grouping-slack-ratio",
            str(self.decode_grouping_slack_ratio),
            "--prefill-queue-policy",
            self.prefill_queue_policy,
            "--decode-scheduler-policy",
            self.decode_scheduler_policy,
            "--kv-fast-pairs",
            self.kv_fast_pairs,
            "--kv-topology-alpha",
            str(self.kv_topology_alpha),
            "--prefill-overload-threshold",
            str(self.prefill_overload_threshold),
            "--block-length",
            str(self.block_length),
        ]
        if self.kv_max_pages is not None:
            args.extend(["--kv-max-pages", str(self.kv_max_pages)])
        if self.mask_id is not None:
            args.extend(["--mask-id", str(self.mask_id)])
        if self.max_gen_len is not None:
            args.extend(["--max-gen-len", str(self.max_gen_len)])
        if self.mode == "colocated":
            args.extend(["--gpus", self.gpus])
            if self.colocated_sticky_worker:
                args.append("--colocated-sticky-worker")
        elif self.mode == "hybrid":
            args.extend(["--prefill-gpus", self.prefill_gpus])
            args.extend(["--hybrid-colocated-gpus", self.hybrid_colocated_gpus])
        if self.disable_metrics:
            args.append("--disable-metrics")
        if self.enable_individual_batch_metrics:
            args.append("--enable-individual-batch-metrics")
        if self.enable_operation_metrics:
            args.append("--enable-operation-metrics")
        if self.op_metrics_layer_id is not None:
            args.extend(["--op-metrics-layer-id", str(self.op_metrics_layer_id)])
        if self.export_partial_metrics:
            args.append("--export-partial-metrics")
        args.append(
            "--enable-hybrid-prefill-overflow"
            if self.enable_hybrid_prefill_overflow
            else "--no-enable-hybrid-prefill-overflow"
        )
        args.append(
            "--enable-cuda-graphs"
            if self.enable_cuda_graphs
            else "--no-enable-cuda-graphs"
        )
        args.extend(["--cuda-graph-batch-sizes", self.cuda_graph_batch_sizes])
        return args


def add_engine_launch_args(
    parser: argparse.ArgumentParser,
    *,
    include_metrics_output_dir: bool,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--mode",
        type=str,
        choices=["colocated", "hybrid"],
        default="colocated",
        help="Serving mode: colocated (prefill+decode per worker) or hybrid",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--scheduler-port",
        type=int,
        default=50051,
        help="gRPC port for the scheduler",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs for colocated workers",
    )
    parser.add_argument(
        "--prefill-gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs for prefill workers",
    )
    parser.add_argument(
        "--hybrid-colocated-gpus",
        type=str,
        default="1",
        help="Comma-separated GPU IDs for colocated workers in hybrid mode",
    )
    parser.add_argument(
        "--base-worker-port",
        type=int,
        default=20100,
        help=(
            "Starting port for worker gRPC servers (incremented per worker). "
            "Must be below the OS ephemeral port range (32768-60999 on Linux) "
            "to avoid collisions with NCCL/gRPC outgoing connections."
        ),
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="localhost",
        help="Master address for torch.distributed",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for torch.distributed",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Max decode steps to batch into a single model forward pass",
    )
    parser.add_argument(
        "--max-tokens-per-iteration",
        type=int,
        default=1024,
        help=(
            "Colocated mode token budget per scheduler iteration "
            "(decode + optional prefill admission)"
        ),
    )
    parser.add_argument(
        "--max-prefill-tokens-per-batch",
        type=int,
        default=2048,
        help="Max total token count across all requests in a single prefill forward pass",
    )
    parser.add_argument(
        "--kv-page-size",
        type=int,
        default=16,
        help="Tokens per KV cache page (default 16)",
    )
    parser.add_argument(
        "--kv-max-pages",
        type=int,
        default=None,
        help=(
            "Max KV cache pages per decode worker. When omitted, auto-selected "
            "per model from its HF architecture (Dream 49152, LLaDA 5632)."
        ),
    )
    if include_metrics_output_dir:
        parser.add_argument(
            "--metrics-output-dir",
            type=str,
            default="benchmark_output",
            help="Directory for metrics CSV and plot output",
        )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable metrics collection",
    )
    parser.add_argument(
        "--enable-individual-batch-metrics",
        action="store_true",
        default=False,
        help="Enable individual batch row export to worker_batch_metrics.csv",
    )
    parser.add_argument(
        "--enable-operation-metrics",
        action="store_true",
        default=False,
        help=(
            "Enable runtime model operation metrics (single sampled layer, "
            "scaled by num_layers)"
        ),
    )
    parser.add_argument(
        "--op-metrics-layer-id",
        type=int,
        default=None,
        help=(
            "Layer index to sample for operation metrics. "
            "Defaults to middle layer when operation metrics are enabled."
        ),
    )
    parser.add_argument(
        "--export-partial-metrics",
        action="store_true",
        help=(
            "Write scheduler metrics CSVs/plots when the server is terminated "
            "before an explicit ExportMetrics RPC. Off by default; the explicit "
            "RPC path is unaffected."
        ),
    )
    parser.add_argument(
        "--prefill-scheduler-policy",
        type=str,
        choices=[
            "round_robin",
            "least_outstanding_prefill_tokens",
            "least_outstanding_requests",
            "least_request_length_sum",
            "balanced_length_clustering",
        ],
        default="least_outstanding_prefill_tokens",
        help="Policy for assigning prefill work across eligible workers",
    )
    parser.add_argument(
        "--decode-grouping-slack-ratio",
        type=float,
        default=0.10,
        help=(
            "Slack ratio for decode balanced_length_clustering candidate filtering "
            "(default 0.10)"
        ),
    )
    parser.add_argument(
        "--prefill-queue-policy",
        type=str,
        choices=["arrival_order", "fewest_remaining_blocks"],
        default="arrival_order",
        help="Policy for ordering queued prefill work within each worker",
    )
    parser.add_argument(
        "--decode-scheduler-policy",
        type=str,
        choices=[
            "round_robin",
            "max_free_memory",
            "topology_guarded_memory",
            "balanced_length_clustering",
        ],
        default="max_free_memory",
        help="Policy for assigning decode work across eligible workers",
    )
    parser.add_argument(
        "--kv-fast-pairs",
        type=str,
        default="0-1,2-3,4-5,6-7",
        help=(
            "Undirected fast GPU pair list for topology-aware hybrid decode "
            'routing, for example "0-1,2-3"'
        ),
    )
    parser.add_argument(
        "--kv-topology-alpha",
        type=float,
        default=0.0,
        help=(
            "Guard factor for topology-aware decode routing; prefer a fast-link "
            "worker only when its free pages are at least alpha times the "
            "best eligible worker"
        ),
    )
    parser.add_argument(
        "--enable-hybrid-prefill-overflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Hybrid mode: when prefill workers are overloaded or out of memory, "
            "fall through to colocated workers for local prefill+decode. On by "
            "default in hybrid mode; pass --no-enable-hybrid-prefill-overflow to "
            "queue such requests as pending instead."
        ),
    )
    parser.add_argument(
        "--prefill-overload-threshold",
        dest="prefill_overload_threshold",
        type=int,
        default=8192,
        help=(
            "Hybrid mode: outstanding prefill tokens per worker above which "
            "requests overflow to colocated workers (when overflow is enabled)"
        ),
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Tokens per generation block for all requests on this server",
    )
    parser.add_argument(
        "--mask-id",
        type=int,
        default=None,
        help=(
            "Token ID used for masked positions. When omitted, sangam reads "
            "`mask_token_id` from the model's HF config.json. Subprocesses spawned "
            "by launch.py always receive the resolved value via this flag."
        ),
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=None,
        help=(
            "When set, every request is initialized with prompt + this many mask "
            "tokens regardless of its requested gen_length. The original requested "
            "gen_length (e.g. from a trace's gen_len column) is reinterpreted as "
            "the per-request stop point in fully-unmasked blocks (rounded up). "
            "Must be a positive multiple of --block-length."
        ),
    )
    parser.add_argument(
        "--colocated-sticky-worker",
        dest="colocated_sticky_worker",
        action="store_true",
        default=False,
        help=(
            "Colocated mode: pin each request to its first-assigned worker for "
            "all subsequent blocks, and hold full-request prefill accounting "
            "upfront rather than per-block."
        ),
    )
    parser.add_argument(
        "--enable-cuda-graphs",
        dest="enable_cuda_graphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Colocated mode: capture and replay CUDA graphs for decode-only "
            "batches to remove per-kernel launch overhead. Falls back to the "
            "eager path for prefill, mixed, and oversized batches. On by "
            "default; pass --no-enable-cuda-graphs to disable."
        ),
    )
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        dest="cuda_graph_batch_sizes",
        type=str,
        default=DEFAULT_CUDA_GRAPH_BATCH_SIZES_SPEC,
        help=(
            "Comma-separated decode batch sizes to capture CUDA graphs for "
            f"(e.g. '1,2,4,8,16'). Defaults to '{DEFAULT_CUDA_GRAPH_BATCH_SIZES_SPEC}' "
            "capped at --max-batch-size. Only used when --enable-cuda-graphs is set."
        ),
    )
    return parser


def engine_launch_config_from_namespace(
    args: argparse.Namespace,
    *,
    metrics_output_dir: str | None = None,
) -> EngineLaunchConfig:
    values = {
        "mode": args.mode,
        "model": args.model,
        "scheduler_port": args.scheduler_port,
        "gpus": args.gpus,
        "prefill_gpus": args.prefill_gpus,
        "hybrid_colocated_gpus": args.hybrid_colocated_gpus,
        "base_worker_port": args.base_worker_port,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "max_batch_size": args.max_batch_size,
        "max_tokens_per_iteration": args.max_tokens_per_iteration,
        "max_prefill_tokens_per_batch": args.max_prefill_tokens_per_batch,
        "kv_page_size": args.kv_page_size,
        "kv_max_pages": args.kv_max_pages,
        "disable_metrics": args.disable_metrics,
        "enable_individual_batch_metrics": args.enable_individual_batch_metrics,
        "enable_operation_metrics": args.enable_operation_metrics,
        "op_metrics_layer_id": args.op_metrics_layer_id,
        "export_partial_metrics": args.export_partial_metrics,
        "prefill_scheduler_policy": args.prefill_scheduler_policy,
        "decode_grouping_slack_ratio": args.decode_grouping_slack_ratio,
        "prefill_queue_policy": args.prefill_queue_policy,
        "decode_scheduler_policy": args.decode_scheduler_policy,
        "kv_fast_pairs": args.kv_fast_pairs,
        "kv_topology_alpha": args.kv_topology_alpha,
        "prefill_overload_threshold": args.prefill_overload_threshold,
        "enable_hybrid_prefill_overflow": args.enable_hybrid_prefill_overflow,
        "block_length": args.block_length,
        "mask_id": args.mask_id,
        "max_gen_len": args.max_gen_len,
        "colocated_sticky_worker": args.colocated_sticky_worker,
        "enable_cuda_graphs": (
            True if args.enable_cuda_graphs is None else args.enable_cuda_graphs
        ),
        "cuda_graph_batch_sizes": args.cuda_graph_batch_sizes,
    }
    if args.enable_hybrid_prefill_overflow is None:
        values["enable_hybrid_prefill_overflow"] = args.mode == "hybrid"
    if values["mask_id"] is None:
        from sangam.model.model_loader import read_mask_token_id

        values["mask_id"] = read_mask_token_id(args.model)
    if values["kv_max_pages"] is None:
        from sangam.model.model_loader import read_default_kv_max_pages

        values["kv_max_pages"] = read_default_kv_max_pages(args.model)
    if metrics_output_dir is None:
        if hasattr(args, "metrics_output_dir"):
            values["metrics_output_dir"] = args.metrics_output_dir
    else:
        values["metrics_output_dir"] = metrics_output_dir
    config = EngineLaunchConfig(**values)
    validate_engine_launch_config(config)
    return config


def validate_engine_launch_config(config: EngineLaunchConfig) -> None:
    if (
        config.prefill_scheduler_policy
        == PrefillSchedulerPolicy.LEAST_OUTSTANDING_REQUESTS.value
        and config.mode != "colocated"
    ):
        raise ValueError(
            "--prefill-scheduler-policy=least_outstanding_requests is supported "
            "only in colocated mode"
        )
    if (
        config.prefill_scheduler_policy
        == PrefillSchedulerPolicy.LEAST_REQUEST_LENGTH_SUM.value
        and config.mode != "colocated"
    ):
        raise ValueError(
            "--prefill-scheduler-policy=least_request_length_sum is supported "
            "only in colocated mode"
        )
    if (
        config.prefill_scheduler_policy
        == PrefillSchedulerPolicy.BALANCED_LENGTH_CLUSTERING.value
        and config.mode != "colocated"
    ):
        raise ValueError(
            "--prefill-scheduler-policy=balanced_length_clustering is supported "
            "only in colocated mode"
        )
    if config.decode_grouping_slack_ratio < 0:
        raise ValueError("--decode-grouping-slack-ratio must be non-negative")
    if (
        config.decode_scheduler_policy
        == DecodeSchedulerPolicy.TOPOLOGY_GUARDED_MEMORY.value
        and config.mode != "colocated"
        and not config.kv_fast_pairs.strip()
    ):
        raise ValueError(
            "--kv-fast-pairs is required when "
            "--decode-scheduler-policy=topology_guarded_memory"
        )
    if config.kv_topology_alpha < 0:
        raise ValueError("--kv-topology-alpha must be non-negative")
    if config.enable_hybrid_prefill_overflow and config.mode != "hybrid":
        raise ValueError(
            "--enable-hybrid-prefill-overflow is supported only in hybrid mode"
        )
    if config.max_gen_len is not None:
        if config.max_gen_len <= 0:
            raise ValueError("--max-gen-len must be a positive integer")
        if config.max_gen_len % config.block_length != 0:
            raise ValueError(
                f"--max-gen-len ({config.max_gen_len}) must be divisible by "
                f"--block-length ({config.block_length})"
            )


def parse_cuda_graph_batch_sizes(
    spec: str | None, *, max_batch_size: int
) -> tuple[int, ...]:
    """Resolve the decode CUDA-graph bucket sizes.

    Returns the sorted, de-duplicated buckets that fit within
    ``max_batch_size``. ``spec`` is a comma-separated list; when ``None`` the
    default ``DEFAULT_CUDA_GRAPH_BATCH_SIZES`` is used.
    """
    if spec is None:
        candidates: tuple[int, ...] = DEFAULT_CUDA_GRAPH_BATCH_SIZES
    else:
        candidates = tuple(int(part) for part in spec.split(",") if part.strip())
        if any(b <= 0 for b in candidates):
            raise ValueError("--cuda-graph-batch-sizes must be positive integers")
    buckets = tuple(sorted({b for b in candidates if b <= max_batch_size}))
    if not buckets:
        raise ValueError(
            "--cuda-graph-batch-sizes resolved to an empty set "
            f"(max_batch_size={max_batch_size})"
        )
    return buckets


# ---------- Per-class config factories ----------
#
# These map a validated `EngineLaunchConfig` into the per-class config
# dataclass each scheduler/worker constructor accepts. Keeping the
# mapping in this module (rather than scattering field lists across
# `_run_*` call sites in `entrypoints/launch.py`) means new fields only
# need to be wired through in one place.


def build_colocated_scheduler_config(
    cfg: EngineLaunchConfig,
) -> ColocatedSchedulerConfig:
    if cfg.mask_id is None:
        raise ValueError("mask_id must be resolved before building scheduler configs")
    return ColocatedSchedulerConfig(
        metrics_output_dir=cfg.metrics_output_dir,
        enable_metrics=not cfg.disable_metrics,
        enable_individual_batch_metrics=cfg.enable_individual_batch_metrics,
        export_partial_metrics=cfg.export_partial_metrics,
        block_length=cfg.block_length,
        mask_id=cfg.mask_id,
        max_gen_len=cfg.max_gen_len,
        max_grpc_message_length=cfg.max_grpc_message_length,
        prefill_scheduler_policy=cfg.prefill_scheduler_policy,
        decode_grouping_slack_ratio=cfg.decode_grouping_slack_ratio,
        colocated_sticky_worker=cfg.colocated_sticky_worker,
    )


def build_hybrid_scheduler_config(
    cfg: EngineLaunchConfig,
) -> HybridSchedulerConfig:
    if cfg.mask_id is None:
        raise ValueError("mask_id must be resolved before building scheduler configs")
    return HybridSchedulerConfig(
        metrics_output_dir=cfg.metrics_output_dir,
        enable_metrics=not cfg.disable_metrics,
        enable_individual_batch_metrics=cfg.enable_individual_batch_metrics,
        export_partial_metrics=cfg.export_partial_metrics,
        block_length=cfg.block_length,
        mask_id=cfg.mask_id,
        max_gen_len=cfg.max_gen_len,
        max_grpc_message_length=cfg.max_grpc_message_length,
        prefill_scheduler_policy=cfg.prefill_scheduler_policy,
        decode_grouping_slack_ratio=cfg.decode_grouping_slack_ratio,
        decode_scheduler_policy=cfg.decode_scheduler_policy,
        kv_fast_pairs=cfg.kv_fast_pairs,
        kv_topology_alpha=cfg.kv_topology_alpha,
        prefill_overload_threshold=cfg.prefill_overload_threshold,
        enable_prefill_overflow=cfg.enable_hybrid_prefill_overflow,
    )


def build_prefill_worker_config(
    cfg: EngineLaunchConfig,
    *,
    worker_id: str,
    gpu_id: int,
    dist_rank: int,
    world_size: int,
    port: int,
    scheduler_address: str,
) -> PrefillWorkerConfig:
    return PrefillWorkerConfig(
        worker_id=worker_id,
        gpu_id=gpu_id,
        dist_rank=dist_rank,
        world_size=world_size,
        port=port,
        model_name=cfg.model,
        scheduler_address=scheduler_address,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        enable_metrics=not cfg.disable_metrics,
        enable_operation_metrics=cfg.enable_operation_metrics,
        op_metrics_layer_id=cfg.op_metrics_layer_id,
        kv_page_size=cfg.kv_page_size,
        kv_max_pages=cfg.kv_max_pages,
        kv_dtype=cfg.kv_dtype,
        max_grpc_message_length=cfg.max_grpc_message_length,
        poll_interval=cfg.poll_interval,
        max_prefill_tokens_per_batch=cfg.max_prefill_tokens_per_batch,
        prefill_queue_policy=cfg.prefill_queue_policy,
        kv_transfer_timeout_s=cfg.kv_transfer_timeout_s,
        streaming_layer_ready_timeout_s=cfg.streaming_layer_ready_timeout_s,
        streaming_recv_join_timeout_s=cfg.streaming_recv_join_timeout_s,
    )


def build_colocated_worker_config(
    cfg: EngineLaunchConfig,
    *,
    worker_id: str,
    gpu_id: int,
    dist_rank: int,
    world_size: int,
    port: int,
    scheduler_address: str,
    enable_kv_receive: bool,
) -> ColocatedWorkerConfig:
    return ColocatedWorkerConfig(
        worker_id=worker_id,
        gpu_id=gpu_id,
        dist_rank=dist_rank,
        world_size=world_size,
        port=port,
        model_name=cfg.model,
        scheduler_address=scheduler_address,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        enable_metrics=not cfg.disable_metrics,
        enable_operation_metrics=cfg.enable_operation_metrics,
        op_metrics_layer_id=cfg.op_metrics_layer_id,
        kv_page_size=cfg.kv_page_size,
        kv_max_pages=cfg.kv_max_pages,
        kv_dtype=cfg.kv_dtype,
        max_grpc_message_length=cfg.max_grpc_message_length,
        poll_interval=cfg.poll_interval,
        max_batch_size=cfg.max_batch_size,
        max_tokens_per_iteration=cfg.max_tokens_per_iteration,
        prefill_queue_policy=cfg.prefill_queue_policy,
        enable_kv_receive=enable_kv_receive,
        block_length=cfg.block_length,
        enable_cuda_graphs=cfg.enable_cuda_graphs,
        cuda_graph_batch_sizes=parse_cuda_graph_batch_sizes(
            cfg.cuda_graph_batch_sizes, max_batch_size=cfg.max_batch_size
        ),
    )
