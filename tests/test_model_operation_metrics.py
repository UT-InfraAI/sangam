import types

import pytest
import torch

from sangam.model.model_runner import (
    _iter_op_metrics_blocks,
    measure_operation_time,
)
from sangam.worker.model_operation_metrics import (
    batch_operation_metric_values,
    create_operation_metrics_context,
    resolve_profile_layer_id,
)


def test_resolve_profile_layer_id_disabled_returns_none() -> None:
    assert (
        resolve_profile_layer_id(enabled=False, requested_layer_id=None, num_layers=32)
        is None
    )


def test_resolve_profile_layer_id_defaults_to_middle_layer() -> None:
    assert (
        resolve_profile_layer_id(enabled=True, requested_layer_id=None, num_layers=32)
        == 16
    )


def test_resolve_profile_layer_id_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="op_metrics_layer_id out of range"):
        resolve_profile_layer_id(enabled=True, requested_layer_id=99, num_layers=32)


def test_operation_context_scales_by_num_layers() -> None:
    context = create_operation_metrics_context(
        enabled=True,
        profile_layer_id=4,
        num_layers=24,
    )
    assert context is not None
    context.record_sample("qkv", layer_id=4, elapsed_seconds=0.03)
    context.record_sample("attn", layer_id=4, elapsed_seconds=0.01)
    context.record_sample("mlp", layer_id=4, elapsed_seconds=0.02)
    assert context.scaled_seconds() == {"qkv": 0.72, "attn": 0.24, "mlp": 0.48}


def test_batch_operation_metric_values_disabled_returns_none() -> None:
    assert batch_operation_metric_values(
        enabled=False,
        operation_metrics_seconds={"attn": 1.0, "mlp": 2.0, "qkv": 3.0},
    ) == (None, None, None)


def test_batch_operation_metric_values_enabled_defaults_to_zero() -> None:
    assert batch_operation_metric_values(
        enabled=True,
        operation_metrics_seconds=None,
    ) == (0.0, 0.0, 0.0)


def test_batch_operation_metric_values_enabled_uses_positive_samples() -> None:
    assert batch_operation_metric_values(
        enabled=True,
        operation_metrics_seconds={"attn": 0.12, "mlp": 0.34, "qkv": 0.56},
    ) == (0.12, 0.34, 0.56)


def test_measure_operation_time_records_only_profiled_layer() -> None:
    context = create_operation_metrics_context(
        enabled=True, profile_layer_id=4, num_layers=24
    )
    assert context is not None
    cpu = torch.device("cpu")

    # Non-profiled layer: runs fn but records nothing.
    out = measure_operation_time(context, 3, "mlp", lambda: torch.ones(2), cpu)
    assert torch.equal(out, torch.ones(2))
    assert context.scaled_seconds() == {}

    # Profiled layer: records a positive sample for the op.
    measure_operation_time(context, 4, "mlp", lambda: torch.ones(2), cpu)
    scaled = context.scaled_seconds()
    assert set(scaled) == {"mlp"}
    assert scaled["mlp"] > 0.0


def test_measure_operation_time_noop_without_context() -> None:
    # No context installed -> fn runs verbatim, nothing recorded, no error.
    out = measure_operation_time(
        None, 0, "mlp", lambda: torch.ones(3), torch.device("cpu")
    )
    assert torch.equal(out, torch.ones(3))


def _llada_shaped_model(num_blocks: int) -> object:
    blocks = [types.SimpleNamespace(name=f"block-{i}") for i in range(num_blocks)]
    transformer = types.SimpleNamespace(blocks=blocks)
    return types.SimpleNamespace(model=types.SimpleNamespace(transformer=transformer))


def _dream_shaped_model(num_layers: int) -> object:
    layers = [
        types.SimpleNamespace(
            self_attn=types.SimpleNamespace(), mlp=types.SimpleNamespace()
        )
        for _ in range(num_layers)
    ]
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


def test_iter_op_metrics_blocks_llada_yields_blocks() -> None:
    model = _llada_shaped_model(3)
    blocks = list(_iter_op_metrics_blocks(model))
    assert blocks == model.model.transformer.blocks


def test_iter_op_metrics_blocks_dream_yields_layer_and_self_attn() -> None:
    model = _dream_shaped_model(2)
    blocks = list(_iter_op_metrics_blocks(model))
    # Each decoder layer contributes the layer (carries the mlp) and its self_attn.
    assert len(blocks) == 4
    for layer in model.model.layers:
        assert layer in blocks
        assert layer.self_attn in blocks
