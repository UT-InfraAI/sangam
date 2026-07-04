import pytest
import torch
from types import SimpleNamespace

from sangam.kv_cache.paged_kv_cache import (
    PagedAttentionState,
    PagedKVPool,
    RequestKVState,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashInfer fused RoPE kernel requires CUDA",
)


def test_paged_attention_plan_sets_q_and_kv_dtype(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyWrapper:
        def __init__(self, workspace, layout):
            captured["workspace"] = workspace
            captured["layout"] = layout

        def plan(self, **kwargs):
            captured["plan_kwargs"] = kwargs

        def run(self, q, kv):
            return q

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    pool = PagedKVPool(
        num_layers=1,
        max_pages=4,
        page_size=16,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        zero_init=True,
    )
    request = RequestKVState(page_ids=[0], seq_len=4, last_page_len=4)
    workspace = torch.empty(64, dtype=torch.uint8)

    PagedAttentionState(
        pool=pool,
        requests=[request],
        block_starts=[0],
        block_ends=[4],
        active_kv_lens=[4],
        num_q_heads=4,
        device=torch.device("cpu"),
        workspace=workspace,
    )

    plan_kwargs = captured["plan_kwargs"]
    assert plan_kwargs["q_data_type"] == torch.bfloat16
    assert plan_kwargs["kv_data_type"] == torch.bfloat16


def test_paged_attention_update_invokes_page_callback(monkeypatch) -> None:
    class DummyWrapper:
        def __init__(self, workspace, layout):
            self.workspace = workspace
            self.layout = layout

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

        def run(self, q, kv):
            return q

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    pool = PagedKVPool(
        num_layers=1,
        max_pages=4,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        zero_init=True,
    )
    request = RequestKVState(page_ids=[0], seq_len=4, last_page_len=4)
    captured: dict[str, object] = {}

    state = PagedAttentionState(
        pool=pool,
        requests=[request],
        block_starts=[0],
        block_ends=[4],
        active_kv_lens=[4],
        num_q_heads=4,
        device=torch.device("cpu"),
        workspace=torch.empty(64, dtype=torch.uint8),
        kv_page_callback=lambda layer_idx, layer_kv, requests: captured.update(
            layer_idx=layer_idx,
            layer_shape=tuple(layer_kv.shape),
            page_ids=requests[0].page_ids,
        ),
    )

    k = torch.ones((4, 2, 8), dtype=torch.bfloat16)
    v = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    state.update_kv_pages(0, k, v)

    assert captured == {
        "layer_idx": 0,
        "layer_shape": (4, 2, 4, 2, 8),
        "page_ids": [0],
    }


def test_paged_attention_rope_tensors_are_cached_by_dtype(monkeypatch) -> None:
    class DummyWrapper:
        def __init__(self, workspace, layout):
            self.workspace = workspace
            self.layout = layout

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

        def run(self, q, kv):
            return q

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    pool = PagedKVPool(
        num_layers=1,
        max_pages=4,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        zero_init=True,
    )
    state = PagedAttentionState(
        pool=pool,
        requests=[RequestKVState(page_ids=[0], seq_len=4, last_page_len=4)],
        block_starts=[0],
        block_ends=[4],
        active_kv_lens=[4],
        num_q_heads=4,
        device=torch.device("cpu"),
        workspace=torch.empty(64, dtype=torch.uint8),
    )

    calls = {"count": 0}

    def _get_rotary_embedding(max_pos, device):
        calls["count"] += 1
        base = torch.arange(max_pos * 8, dtype=torch.float32, device=device).reshape(
            1, 1, max_pos, 8
        )
        return base, base + 1

    rotary = SimpleNamespace(get_rotary_embedding=_get_rotary_embedding)

    sin_1, cos_1 = state.get_rope_tensors(
        rotary, torch.bfloat16, False, torch.device("cpu")
    )
    sin_2, cos_2 = state.get_rope_tensors(
        rotary, torch.bfloat16, False, torch.device("cpu")
    )

    assert calls["count"] == 1
    assert sin_1.dtype == torch.bfloat16
    assert cos_1.dtype == torch.bfloat16
    assert sin_1 is sin_2
    assert cos_1 is cos_2


def test_paged_attention_layer_order_guard(monkeypatch) -> None:
    class DummyWrapper:
        def __init__(self, workspace, layout):
            self.workspace = workspace
            self.layout = layout

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

        def run(self, q, kv):
            return q

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    pool = PagedKVPool(
        num_layers=1,
        max_pages=4,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        zero_init=True,
    )
    state = PagedAttentionState(
        pool=pool,
        requests=[RequestKVState(page_ids=[0], seq_len=4, last_page_len=4)],
        block_starts=[0],
        block_ends=[4],
        active_kv_lens=[4],
        num_q_heads=4,
        device=torch.device("cpu"),
        workspace=torch.empty(64, dtype=torch.uint8),
    )

    assert state.next_layer_idx(0) == 0
    state.finish_layer(0)
    assert state.next_layer_idx(1) == 1


def test_paged_kv_pool_zero_init_uses_zeroed_storage() -> None:
    pool = PagedKVPool(
        num_layers=1,
        max_pages=1,
        page_size=2,
        num_kv_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        zero_init=True,
    )

    assert torch.equal(pool.kv_data[0], torch.zeros_like(pool.kv_data[0]))


def test_paged_attention_uses_active_kv_len_for_page_plan(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyWrapper:
        def __init__(self, workspace, layout):
            self.workspace = workspace
            self.layout = layout

        def plan(self, **kwargs):
            captured["plan_kwargs"] = kwargs

        def run(self, q, kv):
            return q

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    pool = PagedKVPool(
        num_layers=1,
        max_pages=4,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        zero_init=True,
    )
    request = RequestKVState(page_ids=[0, 1, 2], seq_len=12, last_page_len=4)

    PagedAttentionState(
        pool=pool,
        requests=[request],
        block_starts=[4],
        block_ends=[5],
        active_kv_lens=[5],
        num_q_heads=4,
        device=torch.device("cpu"),
        workspace=torch.empty(64, dtype=torch.uint8),
    )

    plan_kwargs = captured["plan_kwargs"]
    assert plan_kwargs["paged_kv_indptr"].tolist() == [0, 2]
    assert plan_kwargs["paged_kv_indices"].tolist() == [0, 1]
    assert plan_kwargs["paged_kv_last_page_len"].tolist() == [1]


def test_paged_attention_scatter_and_rope_follow_absolute_query_positions(
    monkeypatch,
) -> None:
    class DummyWrapper:
        def __init__(self, workspace, layout):
            self.workspace = workspace
            self.layout = layout

        def plan(self, **kwargs):
            self.plan_kwargs = kwargs

        def run(self, q, kv):
            return q

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    pool = PagedKVPool(
        num_layers=1,
        max_pages=8,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        zero_init=True,
    )
    state = PagedAttentionState(
        pool=pool,
        requests=[
            RequestKVState(page_ids=[0, 1], seq_len=8, last_page_len=4),
            RequestKVState(page_ids=[2, 3], seq_len=8, last_page_len=4),
        ],
        block_starts=[1, 5],
        block_ends=[3, 7],
        active_kv_lens=[8, 8],
        num_q_heads=4,
        device=torch.device("cpu"),
        workspace=torch.empty(64, dtype=torch.uint8),
    )

    assert state.scatter_page_ids.tolist() == [0, 0, 3, 3]
    assert state.scatter_offsets.tolist() == [1, 2, 1, 2]
    assert state.rope_positions.tolist() == [1, 2, 5, 6]


@CUDA
def test_apply_rope_inplace_matches_unfused_reference() -> None:
    """Fused FlashInfer RoPE matches the legacy fp32 unfused path within bf16 tol.

    The reference is the exact code the fused path replaces in LLaDA's decode:
    select cos/sin at the batch positions and apply `apply_rotary_pos_emb` in
    fp32, then cast back to bf16. The fused kernel works on bf16 q/k with an fp32
    cos_sin cache, so we use the loose tolerance from the fused-MLP test.
    """
    from sangam.model.llada.modeling_llada import BufferCache, RotaryEmbedding

    device = torch.device("cuda")
    head_dim, n_q_heads, n_kv_heads = 64, 8, 2
    d_model = head_dim * n_q_heads
    rope_max_pos = 32

    config = SimpleNamespace(
        d_model=d_model,
        n_heads=n_q_heads,
        rope_theta=10000.0,
        max_sequence_length=rope_max_pos,
        init_device="cuda",
    )
    rope = RotaryEmbedding(config, BufferCache())

    rope_positions = torch.tensor(
        [0, 3, 7, 15, 16, 31], dtype=torch.long, device=device
    )
    state = PagedAttentionState.from_static_buffers(
        pool=SimpleNamespace(),
        wrapper=SimpleNamespace(),
        scatter_page_ids=torch.zeros_like(rope_positions),
        scatter_offsets=torch.zeros_like(rope_positions),
        rope_positions=rope_positions,
        rope_max_pos=rope_max_pos,
    )

    n_tok = rope_positions.numel()
    q = torch.randn(n_tok, n_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(n_tok, n_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Reference: the fp32 unfused path being replaced.
    sel_sin, sel_cos = state.get_rope_tensors(rope, q.dtype, True, device)
    q_ref = rope.apply_rotary_pos_emb(sel_sin.float(), sel_cos.float(), q.float()).to(
        q.dtype
    )
    k_ref = rope.apply_rotary_pos_emb(sel_sin.float(), sel_cos.float(), k.float()).to(
        k.dtype
    )

    # Fused in-place rotation.
    state.clear_rope_cache()
    state.apply_rope_inplace(rope, q, k, head_dim)

    torch.testing.assert_close(q, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k, k_ref, rtol=2e-2, atol=2e-2)

    # cos_sin cache is fp32, sized to the table, and reused across calls.
    assert state._cos_sin_cache.dtype == torch.float32
    assert state._cos_sin_cache.shape == (rope_max_pos, head_dim)
    ptr = state._cos_sin_cache.data_ptr()
    state.apply_rope_inplace(rope, q, k, head_dim)
    assert state._cos_sin_cache.data_ptr() == ptr


@CUDA
def test_apply_rope_inplace_matches_dream_unfused_reference() -> None:
    """Fused FlashInfer RoPE matches Dream's bf16 unfused path within tolerance.

    Dream already runs RoPE in bf16, so the reference selects cos/sin at the
    batch positions and applies `apply_rotary_pos_emb` directly on bf16 q/k (no
    fp32 upcast). This also exercises Dream's table layout, which bakes
    `attention_scaling` into the cached sin/cos.
    """
    from sangam.model.dream.configuration_dream import DreamConfig
    from sangam.model.dream.modeling_dream import DreamRotaryEmbedding

    device = torch.device("cuda")
    head_dim, n_q_heads, n_kv_heads = 64, 8, 2
    rope_max_pos = 32

    cfg = DreamConfig(
        vocab_size=128,
        hidden_size=head_dim * n_q_heads,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=n_q_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=rope_max_pos,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        mask_token_id=99,
        pad_token_id=0,
        use_cache=False,
    )
    rope = DreamRotaryEmbedding(config=cfg).to(device)

    rope_positions = torch.tensor(
        [0, 3, 7, 15, 16, 31], dtype=torch.long, device=device
    )
    state = PagedAttentionState.from_static_buffers(
        pool=SimpleNamespace(),
        wrapper=SimpleNamespace(),
        scatter_page_ids=torch.zeros_like(rope_positions),
        scatter_offsets=torch.zeros_like(rope_positions),
        rope_positions=rope_positions,
        rope_max_pos=rope_max_pos,
    )

    n_tok = rope_positions.numel()
    q = torch.randn(n_tok, n_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(n_tok, n_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Reference: Dream's bf16 unfused path being replaced.
    sel_sin, sel_cos = state.get_rope_tensors(rope, q.dtype, False, device)
    q_ref = rope.apply_rotary_pos_emb(sel_sin, sel_cos, q)
    k_ref = rope.apply_rotary_pos_emb(sel_sin, sel_cos, k)

    # Fused in-place rotation.
    state.clear_rope_cache()
    state.apply_rope_inplace(rope, q, k, head_dim)

    torch.testing.assert_close(q, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k, k_ref, rtol=2e-2, atol=2e-2)
