"""Helpers for runtime model operation metrics."""

from __future__ import annotations

from sangam.model.model_runner import ModelOperationMetricsContext


def resolve_profile_layer_id(
    *,
    enabled: bool,
    requested_layer_id: int | None,
    num_layers: int,
) -> int | None:
    if not enabled:
        return None
    profile_layer_id = requested_layer_id
    if profile_layer_id is None:
        profile_layer_id = num_layers // 2
    if profile_layer_id < 0 or profile_layer_id >= num_layers:
        raise ValueError(
            "op_metrics_layer_id out of range: "
            f"got {profile_layer_id}, expected [0, {num_layers})"
        )
    return profile_layer_id


def create_operation_metrics_context(
    *,
    enabled: bool,
    profile_layer_id: int | None,
    num_layers: int,
) -> ModelOperationMetricsContext | None:
    if not enabled or profile_layer_id is None:
        return None
    return ModelOperationMetricsContext.create(
        enabled=True,
        profile_layer_id=profile_layer_id,
        num_layers=num_layers,
    )


def batch_operation_metric_values(
    *,
    enabled: bool,
    operation_metrics_seconds: dict[str, float] | None,
) -> tuple[float | None, float | None, float | None]:
    if not enabled:
        return None, None, None
    attn_time = 0.0
    mlp_time = 0.0
    qkv_time = 0.0
    if operation_metrics_seconds:
        sampled_attn = operation_metrics_seconds.get("attn")
        sampled_mlp = operation_metrics_seconds.get("mlp")
        sampled_qkv = operation_metrics_seconds.get("qkv")
        if sampled_attn is not None and sampled_attn > 0.0:
            attn_time = sampled_attn
        if sampled_mlp is not None and sampled_mlp > 0.0:
            mlp_time = sampled_mlp
        if sampled_qkv is not None and sampled_qkv > 0.0:
            qkv_time = sampled_qkv
    return attn_time, mlp_time, qkv_time
