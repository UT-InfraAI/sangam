"""Numerical-equivalence test for the Dream fused QKV projection.

`DreamAttention.fuse_qkv()` concatenates the separate q/k/v projections (all
bias-carrying, with GQA-shaped k/v) into a single wide `nn.Linear`. The fused
forward issues one GEMM then splits, which must match the three separate GEMMs.
The QKV GEMM + split is plain linear algebra, so this runs on CPU without
CUDA/FlashInfer.
"""

import torch

from sangam.model.dream.configuration_dream import DreamConfig
from sangam.model.dream.modeling_dream import DreamAttention


def _make_config() -> DreamConfig:
    return DreamConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: k/v narrower than q
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_dropout=0.0,
        mask_token_id=99,
        pad_token_id=0,
        use_cache=False,
    )


def test_fused_qkv_matches_separate_projections() -> None:
    torch.manual_seed(0)
    config = _make_config()
    attn = DreamAttention(config, layer_idx=0).eval()

    x = torch.randn(2, 5, config.hidden_size)

    with torch.inference_mode():
        ref_q = attn.q_proj(x)
        ref_k = attn.k_proj(x)
        ref_v = attn.v_proj(x)

        assert not attn._qkv_fused
        attn.fuse_qkv()
        assert attn._qkv_fused

        q, k, v = attn.qkv_proj(x).split(attn.fused_dims, dim=-1)

    torch.testing.assert_close(q, ref_q)
    torch.testing.assert_close(k, ref_k)
    torch.testing.assert_close(v, ref_v)
