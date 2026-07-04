"""Tests for the per-class config factories and __post_init__ validators."""

import pytest

from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.engine.launch_config import (
    EngineLaunchConfig,
    build_colocated_scheduler_config,
    build_colocated_worker_config,
    build_hybrid_scheduler_config,
    build_prefill_worker_config,
)
from sangam.engine.scheduler_config import (
    BaseSchedulerConfig,
    ColocatedSchedulerConfig,
    HybridSchedulerConfig,
)
from sangam.worker.worker_config import (
    BaseWorkerConfig,
    ColocatedWorkerConfig,
    PrefillWorkerConfig,
)


def _launch_config(**overrides) -> EngineLaunchConfig:
    cfg = EngineLaunchConfig(
        mode="colocated",
        model="GSAI-ML/LLaDA-8B-Instruct",
        scheduler_port=50051,
        gpus="0",
        prefill_gpus="0",
        hybrid_colocated_gpus="1",
        base_worker_port=20100,
        master_addr="localhost",
        master_port=29500,
        max_batch_size=8,
        max_tokens_per_iteration=1024,
        max_prefill_tokens_per_batch=4096,
        kv_page_size=16,
        kv_max_pages=2048,
        metrics_output_dir="./metrics_output",
        disable_metrics=False,
        enable_individual_batch_metrics=False,
        enable_operation_metrics=False,
        op_metrics_layer_id=None,
        export_partial_metrics=False,
        prefill_scheduler_policy="round_robin",
        decode_grouping_slack_ratio=0.10,
        prefill_queue_policy="arrival_order",
        decode_scheduler_policy="max_free_memory",
        kv_fast_pairs="",
        kv_topology_alpha=0.7,
        prefill_overload_threshold=4,
        enable_hybrid_prefill_overflow=False,
        block_length=32,
        mask_id=126336,
        max_gen_len=None,
        colocated_sticky_worker=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ---------- Factory mapping smoke tests (one per family) ----------


def test_build_hybrid_scheduler_config_maps_overload_threshold() -> None:
    cfg = _launch_config(
        mode="hybrid",
        prefill_overload_threshold=6,
        enable_hybrid_prefill_overflow=True,
    )
    sched = build_hybrid_scheduler_config(cfg)
    assert isinstance(sched, HybridSchedulerConfig)
    assert sched.prefill_overload_threshold == 6
    assert sched.enable_prefill_overflow is True


def test_build_colocated_scheduler_config_includes_sticky_worker_flag() -> None:
    cfg = _launch_config(mode="colocated", colocated_sticky_worker=True)
    sched = build_colocated_scheduler_config(cfg)
    assert isinstance(sched, ColocatedSchedulerConfig)
    assert sched.colocated_sticky_worker is True


def test_build_prefill_worker_config_threads_identity_kwargs() -> None:
    cfg = _launch_config()
    worker = build_prefill_worker_config(
        cfg,
        worker_id="prefill-7",
        gpu_id=3,
        dist_rank=2,
        world_size=8,
        port=20107,
        scheduler_address="scheduler:50051",
    )
    assert isinstance(worker, PrefillWorkerConfig)
    assert worker.worker_id == "prefill-7"
    assert worker.gpu_id == 3
    assert worker.dist_rank == 2
    assert worker.world_size == 8
    assert worker.port == 20107
    assert worker.scheduler_address == "scheduler:50051"
    assert worker.kv_page_size == cfg.kv_page_size
    assert worker.max_prefill_tokens_per_batch == cfg.max_prefill_tokens_per_batch


def test_build_colocated_worker_config_propagates_kv_receive_flag() -> None:
    cfg = _launch_config(mode="hybrid")
    worker = build_colocated_worker_config(
        cfg,
        worker_id="colocated-0",
        gpu_id=2,
        dist_rank=0,
        world_size=4,
        port=20100,
        scheduler_address="scheduler:50051",
        enable_kv_receive=True,
    )
    assert isinstance(worker, ColocatedWorkerConfig)
    assert worker.enable_kv_receive is True


# ---------- __post_init__ validator coverage (one positive + one negative per config) ----------


def _base_scheduler_kwargs() -> dict:
    return dict(
        metrics_output_dir="./out",
        enable_metrics=False,
        enable_individual_batch_metrics=False,
        export_partial_metrics=False,
        block_length=32,
        mask_id=126336,
        max_gen_len=None,
        max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
    )


def test_base_scheduler_config_accepts_valid_max_gen_len() -> None:
    cfg = BaseSchedulerConfig(**{**_base_scheduler_kwargs(), "max_gen_len": 64})
    assert cfg.max_gen_len == 64


def test_base_scheduler_config_rejects_max_gen_len_not_divisible_by_block_length() -> (
    None
):
    with pytest.raises(ValueError, match="must be divisible by"):
        BaseSchedulerConfig(**{**_base_scheduler_kwargs(), "max_gen_len": 60})


def test_hybrid_scheduler_config_rejects_negative_kv_topology_alpha() -> None:
    kwargs = {
        **_base_scheduler_kwargs(),
        "prefill_scheduler_policy": "round_robin",
        "decode_grouping_slack_ratio": 0.10,
        "decode_scheduler_policy": "round_robin",
        "kv_fast_pairs": "",
        "kv_topology_alpha": -0.1,
        "prefill_overload_threshold": 4,
        "enable_prefill_overflow": False,
    }
    with pytest.raises(ValueError, match="kv_topology_alpha"):
        HybridSchedulerConfig(**kwargs)


def test_colocated_scheduler_config_rejects_unknown_policy() -> None:
    kwargs = {
        **_base_scheduler_kwargs(),
        "prefill_scheduler_policy": "not_a_real_policy",
        "decode_grouping_slack_ratio": 0.10,
        "colocated_sticky_worker": False,
    }
    with pytest.raises(ValueError):
        ColocatedSchedulerConfig(**kwargs)


def _base_worker_kwargs() -> dict:
    import torch

    return dict(
        worker_id="pw-0",
        gpu_id=0,
        dist_rank=0,
        world_size=1,
        port=20100,
        model_name="dummy",
        scheduler_address="localhost:50051",
        master_addr="localhost",
        master_port=29500,
        enable_metrics=False,
        enable_operation_metrics=False,
        op_metrics_layer_id=None,
        kv_page_size=16,
        kv_max_pages=2048,
        kv_dtype=torch.bfloat16,
        max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
        poll_interval=0.1,
    )


def test_base_worker_config_rejects_dist_rank_exceeding_world_size() -> None:
    kwargs = {**_base_worker_kwargs(), "dist_rank": 2, "world_size": 2}
    with pytest.raises(ValueError, match="dist_rank"):
        BaseWorkerConfig(**kwargs)


def test_prefill_worker_config_rejects_invalid_queue_policy() -> None:
    kwargs = {
        **_base_worker_kwargs(),
        "max_prefill_tokens_per_batch": 4096,
        "prefill_queue_policy": "not_a_real_policy",
        "kv_transfer_timeout_s": 30.0,
        "streaming_layer_ready_timeout_s": 30.0,
        "streaming_recv_join_timeout_s": 30.0,
    }
    with pytest.raises(ValueError):
        PrefillWorkerConfig(**kwargs)
