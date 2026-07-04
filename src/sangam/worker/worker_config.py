"""Per-class config dataclasses for worker implementations.

The hierarchy mirrors the worker class layout: `BaseWorkerConfig` holds
the fields consumed by `BaseWorker` (process identity, ports, KV pool
sizing, metrics knobs); each concrete worker type adds servicer-specific
fields. One config per worker type is shared by both the `BaseWorker`
subclass and its paired `*Servicer`. Configs are built from
`EngineLaunchConfig` via the `build_*_worker_config` factories in
`engine.launch_config` and threaded end-to-end across `mp.Process`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sangam.types import PrefillQueuePolicy


@dataclass
class BaseWorkerConfig:
    worker_id: str
    gpu_id: int
    dist_rank: int
    world_size: int
    port: int
    model_name: str
    scheduler_address: str
    master_addr: str
    master_port: int
    enable_metrics: bool
    enable_operation_metrics: bool
    op_metrics_layer_id: int | None
    kv_page_size: int
    kv_max_pages: int
    kv_dtype: torch.dtype
    max_grpc_message_length: int
    poll_interval: float

    def __post_init__(self) -> None:
        if self.gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
        if self.dist_rank < 0:
            raise ValueError("dist_rank must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if self.dist_rank >= self.world_size:
            raise ValueError(
                f"dist_rank ({self.dist_rank}) must be < world_size ({self.world_size})"
            )
        if self.port <= 0:
            raise ValueError("port must be a positive integer")
        if self.master_port <= 0:
            raise ValueError("master_port must be a positive integer")
        if self.kv_page_size <= 0:
            raise ValueError("kv_page_size must be a positive integer")
        if self.kv_max_pages <= 0:
            raise ValueError("kv_max_pages must be a positive integer")


@dataclass
class PrefillWorkerConfig(BaseWorkerConfig):
    max_prefill_tokens_per_batch: int
    prefill_queue_policy: str
    kv_transfer_timeout_s: float
    streaming_layer_ready_timeout_s: float
    streaming_recv_join_timeout_s: float

    def __post_init__(self) -> None:
        super().__post_init__()
        PrefillQueuePolicy(self.prefill_queue_policy)
        if self.max_prefill_tokens_per_batch <= 0:
            raise ValueError("max_prefill_tokens_per_batch must be a positive integer")


@dataclass
class ColocatedWorkerConfig(BaseWorkerConfig):
    max_batch_size: int
    max_tokens_per_iteration: int
    prefill_queue_policy: str
    enable_kv_receive: bool
    block_length: int
    enable_cuda_graphs: bool
    cuda_graph_batch_sizes: tuple[int, ...] | None

    def __post_init__(self) -> None:
        super().__post_init__()
        PrefillQueuePolicy(self.prefill_queue_policy)
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if self.max_tokens_per_iteration <= 0:
            raise ValueError("max_tokens_per_iteration must be a positive integer")
        if self.block_length <= 0:
            raise ValueError("block_length must be a positive integer")
        if self.enable_cuda_graphs and not self.cuda_graph_batch_sizes:
            raise ValueError(
                "cuda_graph_batch_sizes must be non-empty when "
                "enable_cuda_graphs is set"
            )
