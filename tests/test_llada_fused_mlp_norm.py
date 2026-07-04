"""Numerical-equivalence tests for the fused norm/residual + SwiGLU MLP path.

The fused `LLaDALlamaBlock._forward_fused` uses FlashInfer's `rmsnorm`,
`fused_add_rmsnorm`, and `silu_and_mul` kernels, which are CUDA-only, so these
tests are skipped without a GPU. They check that, for the served LLaDA-8B config
(weight-only RMSNorm, no QK-norm, plain SiLU), the fused block produces the same
output as the unfused reference block on identical weights.
"""

import pytest
import torch

from sangam.model.llada.configuration_llada import (
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
)
from sangam.model.llada.modeling_llada import LLaDAModel, RMSLayerNorm, SiLU

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kernels require CUDA/FlashInfer"
)


def _make_config() -> ModelConfig:
    # Mirrors GSAI-ML/LLaDA-8B-Instruct: rms norm (weight-only), plain silu, no
    # QK-norm, no biases.
    return ModelConfig(
        vocab_size=128,
        max_sequence_length=64,
        d_model=64,
        mlp_hidden_size=192,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        block_type=BlockType.llama,
        block_group_size=1,
        activation_type=ActivationType.silu,
        layer_norm_type=LayerNormType.rms,
        layer_norm_with_affine=True,
        include_bias=False,
        bias_for_layer_norm=False,
        attention_layer_norm=False,
        rope=True,
        alibi=False,
        init_device="cpu",
        flash_attention=False,
    )


def _deterministic_attention(block):
    """Replace paged attention with attn_out(W @ stacked qkv) so the block is a
    pure function of its inputs and weights (no KV cache needed)."""

    def _fake(q, k, v, B, T, C, state):
        return block.attn_out((q + k + v).reshape(B, T, C))

    return _fake


@CUDA
def test_fused_block_matches_unfused(monkeypatch) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = LLaDAModel(_make_config(), init_params=True)
    model.fuse_qkv()
    model = model.to(device=device, dtype=dtype).eval()

    block = model.transformer.blocks[0]
    assert isinstance(block.attn_norm, RMSLayerNorm)
    assert isinstance(block.act, SiLU)
    assert block._fused_mlp_norm

    monkeypatch.setattr(block, "_paged_attention", _deterministic_attention(block))
    block._paged_attn_state = object()

    x = torch.randn(2, 5, model.config.d_model, device=device, dtype=dtype)

    with torch.inference_mode():
        # Unfused reference: fuse_ff_gate_up() not yet called, so forward() takes
        # the original path.
        assert not block._ff_fused
        ref = block(x.clone())

        # Fused path.
        model.fuse_ff_gate_up()
        assert block._ff_fused
        out = block(x.clone())

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_guard_disabled_for_non_rms_norm() -> None:
    # LayerNorm (not RMS) must fall through to the unfused path.
    config = _make_config()
    config.layer_norm_type = LayerNormType.default
    model = LLaDAModel(config, init_params=False)
    block = model.transformer.blocks[0]
    assert not block._fused_mlp_norm
