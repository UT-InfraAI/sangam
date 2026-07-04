"""Per-class config dataclasses for scheduler implementations.

The hierarchy mirrors the scheduler classes: `BaseSchedulerConfig` holds
the fields consumed by `BaseScheduler`; each concrete scheduler has its
own subclass that adds policy-specific fields. Configs are constructed
from `EngineLaunchConfig` via the `build_*_scheduler_config` factories
in `engine.launch_config` and threaded end-to-end across the
`mp.Process` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from sangam.types import DecodeSchedulerPolicy, PrefillSchedulerPolicy


@dataclass
class BaseSchedulerConfig:
    metrics_output_dir: str
    enable_metrics: bool
    enable_individual_batch_metrics: bool
    export_partial_metrics: bool
    block_length: int
    mask_id: int
    max_gen_len: int | None
    max_grpc_message_length: int

    def __post_init__(self) -> None:
        if self.block_length <= 0:
            raise ValueError("block_length must be a positive integer")
        if self.max_gen_len is not None:
            if self.max_gen_len <= 0:
                raise ValueError("max_gen_len must be a positive integer")
            if self.max_gen_len % self.block_length != 0:
                raise ValueError(
                    f"max_gen_len ({self.max_gen_len}) must be divisible by "
                    f"block_length ({self.block_length})"
                )


@dataclass
class ColocatedSchedulerConfig(BaseSchedulerConfig):
    prefill_scheduler_policy: str
    decode_grouping_slack_ratio: float
    colocated_sticky_worker: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        # Raises ValueError on unknown policy.
        PrefillSchedulerPolicy(self.prefill_scheduler_policy)
        if self.decode_grouping_slack_ratio < 0:
            raise ValueError("decode_grouping_slack_ratio must be non-negative")


@dataclass
class HybridSchedulerConfig(BaseSchedulerConfig):
    prefill_scheduler_policy: str
    decode_grouping_slack_ratio: float
    decode_scheduler_policy: str
    kv_fast_pairs: str
    kv_topology_alpha: float
    prefill_overload_threshold: int
    enable_prefill_overflow: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        PrefillSchedulerPolicy(self.prefill_scheduler_policy)
        DecodeSchedulerPolicy(self.decode_scheduler_policy)
        if self.decode_grouping_slack_ratio < 0:
            raise ValueError("decode_grouping_slack_ratio must be non-negative")
        if self.kv_topology_alpha < 0:
            raise ValueError("kv_topology_alpha must be non-negative")
