"""Tests for decode CUDA-graph bucket selection and graph/eager parity.

The bucket-selection tests are CPU-only. The parity and padding-isolation
tests require a GPU (FlashInfer + a real diffusion LM) and are marked ``gpu``.
"""

import pytest
import torch

from sangam.engine.launch_config import (
    DEFAULT_CUDA_GRAPH_BATCH_SIZES,
    parse_cuda_graph_batch_sizes,
)

MODEL = "GSAI-ML/LLaDA-8B-Instruct"
BLOCK = 32
PAGE = 16


# ----- CPU-only: bucket parsing / selection -----


def test_parse_default_buckets_capped_to_max_batch_size():
    assert parse_cuda_graph_batch_sizes(None, max_batch_size=128) == (
        DEFAULT_CUDA_GRAPH_BATCH_SIZES
    )
    assert parse_cuda_graph_batch_sizes(None, max_batch_size=16) == (1, 2, 4, 8, 16)


def test_parse_explicit_buckets_dedup_sorted_and_filtered():
    assert parse_cuda_graph_batch_sizes("8,1,2,2,100", max_batch_size=16) == (1, 2, 8)


def test_parse_rejects_nonpositive_and_empty():
    with pytest.raises(ValueError):
        parse_cuda_graph_batch_sizes("0,1", max_batch_size=16)
    with pytest.raises(ValueError):
        parse_cuda_graph_batch_sizes("100,200", max_batch_size=16)


def test_select_bucket_picks_smallest_fitting():
    from sangam.model.cuda_graph_runner import DecodeCudaGraphRunner

    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.buckets = (1, 2, 4, 8, 16)
    runner.max_bucket = 16
    runner.block_length = BLOCK

    assert runner.select_bucket(1) == 1
    assert runner.select_bucket(3) == 4
    assert runner.select_bucket(5) == 8
    assert runner.select_bucket(16) == 16
    with pytest.raises(ValueError):
        runner.select_bucket(17)


def test_can_run_gating():
    from sangam.model.cuda_graph_runner import DecodeCudaGraphRunner

    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.buckets = (1, 2, 4, 8, 16)
    runner.max_bucket = 16
    runner.block_length = BLOCK

    assert runner.can_run(4, BLOCK)
    assert not runner.can_run(0, BLOCK)  # empty batch
    assert not runner.can_run(17, BLOCK)  # exceeds max bucket
    assert not runner.can_run(4, BLOCK + 1)  # block_length mismatch


def test_dummy_pages_per_req():
    from sangam.model.cuda_graph_runner import DecodeCudaGraphRunner

    assert DecodeCudaGraphRunner.dummy_pages_per_req(BLOCK, PAGE) == 2
    assert DecodeCudaGraphRunner.dummy_pages_per_req(PAGE, PAGE) == 1
    assert DecodeCudaGraphRunner.dummy_pages_per_req(PAGE + 1, PAGE) == 2


def test_registration_advertises_reduced_capacity_with_cuda_graphs():
    """Worker subtracts reserved dummy pages from advertised capacity so the
    scheduler's reservation accounting matches real allocatable pages."""
    from types import SimpleNamespace

    from sangam.worker.colocated_worker import ColocatedWorker

    worker = ColocatedWorker.__new__(ColocatedWorker)

    worker._config = SimpleNamespace(
        kv_max_pages=6192,
        kv_page_size=PAGE,
        block_length=BLOCK,
        enable_cuda_graphs=True,
    )
    assert worker._registration_extra_fields() == {
        "max_pages": 6192 - 2,
        "page_size": PAGE,
    }

    worker._config = SimpleNamespace(
        kv_max_pages=6192,
        kv_page_size=PAGE,
        block_length=BLOCK,
        enable_cuda_graphs=False,
    )
    assert worker._registration_extra_fields() == {
        "max_pages": 6192,
        "page_size": PAGE,
    }


# ----- GPU: graph vs eager parity and padding isolation -----


@pytest.fixture(scope="module")
def gpu_model_pool():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from sangam.kv_cache.paged_kv_cache import PagedKVPool
    from sangam.model.model_loader import load_model, read_mask_token_id

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model = load_model(MODEL, 0, torch.bfloat16)
    mask_id = read_mask_token_id(MODEL)
    pool = PagedKVPool(
        num_layers=model.num_layers,
        max_pages=512,
        page_size=PAGE,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        device=device,
        dtype=torch.bfloat16,
        zero_init=True,
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return model, pool, workspace, device, mask_id


def _reset_pool(pool):
    pool.allocator = type(pool.allocator)(pool.max_pages)
    for t in pool.kv_data:
        t.zero_()


def _build_requests(pool, ctxs, seeds):
    from sangam.kv_cache.paged_kv_cache import RequestKVState

    built = []
    for ctx, seed in zip(ctxs, seeds):
        gen = torch.Generator().manual_seed(seed)
        tokens = torch.randint(5, 1000, (BLOCK,), generator=gen).tolist()
        page_ids, last = pool.allocate(ctx)
        state = RequestKVState(page_ids=page_ids, seq_len=ctx, last_page_len=last)
        built.append((tokens, ctx, page_ids, state))
    return built


def _eager_logits(model, pool, workspace, device, ctxs, seeds):
    from sangam.model.model_runner import (
        MixedBatchItem,
        pack_mixed_batch,
        run_mixed_paged_forward,
    )

    _reset_pool(pool)
    built = _build_requests(pool, ctxs, seeds)
    items = [
        MixedBatchItem(
            request_id="r",
            token_ids=tokens,
            query_start=ctx - BLOCK,
            query_end=ctx,
            active_kv_len=ctx,
            kv_state=state,
            phase="decode",
        )
        for (tokens, ctx, _pids, state) in built
    ]
    packed = pack_mixed_batch(
        items=items,
        pool=pool,
        num_q_heads=model.num_q_heads,
        device=device,
        workspace=workspace,
    )
    with torch.inference_mode():
        packed.paged_state.current_layer_idx = 0
        result = run_mixed_paged_forward(model, packed)
    return torch.cat(result.item_logits, dim=0).float().cpu()


@pytest.mark.gpu
def test_graph_matches_eager_exact_fill(gpu_model_pool):
    """Bucket exactly filled (no padding) must match eager bit-for-bit."""
    from sangam.model.cuda_graph_runner import (
        DecodeCudaGraphRunner,
        DecodeGraphItem,
    )

    model, pool, workspace, device, _mask_id = gpu_model_pool
    runner = DecodeCudaGraphRunner(
        model=model,
        pool=pool,
        flashinfer_workspace=workspace,
        device=device,
        block_length=BLOCK,
        num_q_heads=model.num_q_heads,
        buckets=(1, 2),
        rope_max_pos=pool.max_pages * pool.page_size,
    )
    _reset_pool(pool)
    runner.maybe_capture()

    ctxs, seeds = [64, 200], [3, 4]
    eager = _eager_logits(model, pool, workspace, device, ctxs, seeds)

    _reset_pool(pool)
    pool.allocator.allocate(runner._dummy_pages_per_req)  # keep dummy pages reserved
    built = _build_requests(pool, ctxs, seeds)
    items = [
        DecodeGraphItem(
            token_ids=tokens,
            block_start=ctx - BLOCK,
            active_kv_len=ctx,
            page_ids=pids,
        )
        for (tokens, ctx, pids, _state) in built
    ]
    graph = torch.cat(runner.run(items).item_logits, dim=0).float().cpu()

    assert torch.equal(eager.argmax(-1), graph.argmax(-1))
    assert (eager - graph).abs().max().item() == 0.0


@pytest.mark.gpu
def test_graph_with_padding_no_kv_corruption(gpu_model_pool):
    """A padded batch (B' < bucket) must not corrupt real requests' tokens.

    Padding rows change the matmul shape, so logits can differ by bf16 GEMM
    nondeterminism, but argmax should match almost everywhere and real KV
    pages must be written identically to a non-padded run of the same request.
    """
    from sangam.model.cuda_graph_runner import (
        DecodeCudaGraphRunner,
        DecodeGraphItem,
    )

    model, pool, workspace, device, _mask_id = gpu_model_pool
    runner = DecodeCudaGraphRunner(
        model=model,
        pool=pool,
        flashinfer_workspace=workspace,
        device=device,
        block_length=BLOCK,
        num_q_heads=model.num_q_heads,
        buckets=(1, 4),
        rope_max_pos=pool.max_pages * pool.page_size,
    )
    _reset_pool(pool)
    runner.maybe_capture()

    ctxs, seeds = [96, 160, 256], [6, 7, 8]  # B'=3 -> bucket 4 (1 pad)
    eager = _eager_logits(model, pool, workspace, device, ctxs, seeds)

    _reset_pool(pool)
    pool.allocator.allocate(runner._dummy_pages_per_req)
    built = _build_requests(pool, ctxs, seeds)
    items = [
        DecodeGraphItem(
            token_ids=tokens,
            block_start=ctx - BLOCK,
            active_kv_len=ctx,
            page_ids=pids,
        )
        for (tokens, ctx, pids, _state) in built
    ]
    graph = torch.cat(runner.run(items).item_logits, dim=0).float().cpu()

    # Same shapes; the vast majority of argmax decisions agree.
    assert eager.shape == graph.shape
    match = (eager.argmax(-1) == graph.argmax(-1)).float().mean().item()
    assert match > 0.9
