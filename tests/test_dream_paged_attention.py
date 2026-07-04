import pytest
import torch

from sangam.model.dream.configuration_dream import DreamConfig
from sangam.model.dream.modeling_dream import (
    DreamModel,
    DreamRotaryEmbedding,
    apply_rotary_pos_emb,
)
from sangam.model.registry import resolve_model_classes


def _make_config() -> DreamConfig:
    return DreamConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        mask_token_id=99,
        pad_token_id=0,
        use_cache=False,
    )


def test_registry_resolves_dream_architecture() -> None:
    config_cls, model_cls = resolve_model_classes(["DreamModel"])
    assert config_cls is DreamConfig
    assert model_cls is DreamModel


def test_forward_requires_paged_attention_state() -> None:
    model = DreamModel(_make_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long)

    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="paged attention state"):
            model(input_ids=input_ids)


def test_cached_rope_matches_legacy_per_layer_recompute() -> None:
    # The cached/fused RoPE path must produce identical numerics to the legacy
    # `forward` + `apply_rotary_pos_emb` recompute it replaced, and the per-forward
    # sin/cos table must be built once and reused across calls.
    cfg = _make_config()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    rope = DreamRotaryEmbedding(config=cfg)

    torch.manual_seed(0)
    seq_len = 17
    q = torch.randn(seq_len, cfg.num_attention_heads, head_dim)
    k = torch.randn(seq_len, cfg.num_key_value_heads, head_dim)
    positions = torch.randperm(cfg.max_position_embeddings)[:seq_len].sort().values

    # Legacy path: recompute cos/sin per call, apply via rotate_half.
    cos, sin = rope(q, positions.unsqueeze(0))
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    q_legacy, k_legacy = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # New path: cached table + position gather + fused apply.
    pos_sin, pos_cos = rope.get_rotary_embedding(int(positions.max()) + 1, q.device)
    table_obj = pos_sin
    sel_sin = pos_sin[0, 0, positions].unsqueeze(1)
    sel_cos = pos_cos[0, 0, positions].unsqueeze(1)
    q_new = rope.apply_rotary_pos_emb(sel_sin, sel_cos, q)
    k_new = rope.apply_rotary_pos_emb(sel_sin, sel_cos, k)

    assert torch.allclose(q_legacy, q_new, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_legacy, k_new, atol=1e-5, rtol=1e-5)

    # A second request for a shorter span reuses the same cached tensor.
    reused, _ = rope.get_rotary_embedding(seq_len, q.device)
    assert reused.data_ptr() == table_obj.data_ptr()


def test_dream_model_exposes_uniform_property_surface() -> None:
    model = DreamModel(_make_config()).eval()
    assert model.num_layers == 2
    assert model.num_q_heads == 4
    assert model.num_kv_heads == 2
    assert model.head_dim == 8
    model.fuse_qkv()
