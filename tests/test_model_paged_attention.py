from types import SimpleNamespace

import pytest
import torch

from sangam.model.llada.configuration_llada import (
    ActivationType,
    BlockType,
    ModelConfig,
)
from sangam.model.model_runner import (
    MixedBatchItem,
    pack_mixed_batch,
    run_mixed_paged_forward,
)
from sangam.kv_cache.paged_kv_cache import PagedKVPool, RequestKVState
from sangam.model.llada.modeling_llada import LLaDAModel


def _make_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_sequence_length=64,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        block_type=BlockType.llama,
        block_group_size=1,
        activation_type=ActivationType.silu,
        rope=True,
        alibi=False,
        init_device="cpu",
        flash_attention=False,
    )


def test_forward_requires_paged_attention_state() -> None:
    model = LLaDAModel(_make_config(), init_params=False)
    model.fuse_qkv()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long)

    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="paged attention state"):
            model.forward(input_ids=input_ids)


def test_attention_passes_nhd_layout_to_paged_attention(monkeypatch) -> None:
    model = LLaDAModel(_make_config(), init_params=False)
    model.fuse_qkv()
    block = model.transformer.blocks[0]
    hidden_states = torch.randn(2, 8, model.config.d_model)
    captured: dict[str, tuple[int, ...]] = {}

    def _fake_paged_attention(q, k, v, B, T, C, state):
        captured["q_shape"] = tuple(q.shape)
        captured["k_shape"] = tuple(k.shape)
        captured["v_shape"] = tuple(v.shape)
        return torch.zeros(B, T, C, dtype=q.dtype)

    monkeypatch.setattr(block, "_paged_attention", _fake_paged_attention)
    block._paged_attn_state = object()

    with torch.inference_mode():
        block(hidden_states)

    assert captured == {
        "q_shape": (16, 4, 8),
        "k_shape": (16, 4, 8),
        "v_shape": (16, 4, 8),
    }


def test_run_mixed_paged_forward_returns_per_item_logits_for_heterogeneous_lengths(
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

    class DummyModel:
        num_q_heads = 2

        def __init__(self) -> None:
            self.model = SimpleNamespace(
                transformer=SimpleNamespace(
                    blocks=[SimpleNamespace(_paged_attn_state=None)]
                )
            )
            self.seen_input_ids = None

        def __call__(self, input_ids):
            self.seen_input_ids = input_ids
            total_tokens = input_ids.shape[1]
            logits = torch.arange(total_tokens * 6, dtype=torch.float32).reshape(
                1, total_tokens, 6
            )
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        "sangam.kv_cache.paged_kv_cache.flashinfer.BatchPrefillWithPagedKVCacheWrapper",
        DummyWrapper,
    )

    model = DummyModel()
    pool = PagedKVPool(
        num_layers=1,
        max_pages=8,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        zero_init=True,
    )
    packed_batch = pack_mixed_batch(
        items=[
            MixedBatchItem(
                request_id="prefill-1",
                token_ids=[1, 2, 3],
                query_start=0,
                query_end=3,
                active_kv_len=3,
                kv_state=RequestKVState(page_ids=[0], seq_len=3, last_page_len=3),
                phase="prefill",
            ),
            MixedBatchItem(
                request_id="decode-1",
                token_ids=[4, 5],
                query_start=5,
                query_end=7,
                active_kv_len=8,
                kv_state=RequestKVState(page_ids=[1, 2], seq_len=8, last_page_len=4),
                phase="decode",
            ),
        ],
        pool=pool,
        num_q_heads=model.num_q_heads,
        device=torch.device("cpu"),
        workspace=torch.empty(64, dtype=torch.uint8),
    )

    result = run_mixed_paged_forward(model, packed_batch)

    assert model.seen_input_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert result.query_offsets == [0, 3, 5]
    assert [tuple(logits.shape) for logits in result.item_logits] == [
        (1, 3, 6),
        (1, 2, 6),
    ]
    assert result.item_logits[0].tolist() == result.packed_logits[:, :3, :].tolist()
    assert result.item_logits[1].tolist() == result.packed_logits[:, 3:5, :].tolist()
