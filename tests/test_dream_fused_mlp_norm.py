"""Numerical-equivalence tests for the Dream fused norm/residual + SwiGLU MLP path.

The fused `DreamDecoderLayer._forward_fused` and `DreamMLP` fused path use
FlashInfer's `rmsnorm`, `fused_add_rmsnorm`, and `silu_and_mul` kernels, which
are CUDA-only, so the equivalence test is skipped without a GPU. It checks that,
for the served Dream-7B config (weight-only RMSNorm, plain SiLU), the fused layer
matches the unfused reference on identical weights.
"""

import pytest
import torch

from sangam.model.dream.configuration_dream import DreamConfig
from sangam.model.dream.modeling_dream import DreamModel

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kernels require CUDA/FlashInfer"
)


def _make_config(hidden_act: str = "silu") -> DreamConfig:
    return DreamConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act=hidden_act,
        attention_dropout=0.0,
        mask_token_id=99,
        pad_token_id=0,
        use_cache=False,
    )


class _Attn(torch.nn.Module):
    """Deterministic stand-in for DreamAttention: returns (o_proj(x), None, None)
    so the layer is a pure function of inputs/weights (no KV cache / paged state)
    and both the fused and unfused paths exercise identical attention."""

    def __init__(self, real):
        super().__init__()
        self.o_proj = real.o_proj

    def forward(self, hidden_states, **kwargs):
        return self.o_proj(hidden_states), None, None


@CUDA
def test_fused_layer_matches_unfused(monkeypatch) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = DreamModel(_make_config()).to(device=device, dtype=dtype).eval()
    layer = model.model.layers[0]
    assert layer._fused_mlp_norm

    monkeypatch.setattr(layer, "self_attn", _Attn(layer.self_attn))

    x = torch.randn(2, 5, model.config.hidden_size, device=device, dtype=dtype)

    with torch.inference_mode():
        # Unfused reference: fuse_ff_gate_up() not yet called.
        assert not layer.mlp._ff_fused
        ref = layer(x.clone())[0]

        # Fused path.
        model.fuse_ff_gate_up()
        assert layer.mlp._ff_fused
        out = layer(x.clone())[0]

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_guard_disabled_for_non_silu_activation() -> None:
    model = DreamModel(_make_config(hidden_act="gelu"))
    layer = model.model.layers[0]
    assert not layer._fused_mlp_norm
