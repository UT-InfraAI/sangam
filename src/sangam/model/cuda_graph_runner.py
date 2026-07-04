"""CUDA graph capture/replay for colocated decode-only forward passes.

A decode-only batch in the colocated worker is a fixed-shape forward: every
request contributes exactly ``block_length`` query tokens, so a batch of ``B``
requests has ``B * block_length`` query tokens. The forward is dominated by
per-kernel launch overhead (hundreds of tiny kernels walking the HF-style
module code), not GPU compute, so replaying a captured CUDA graph instead of
launching eagerly removes that overhead.

This runner owns, per batch-size bucket ``B``:
  - persistent input/index/scatter/rope buffers (static addresses),
  - a FlashInfer paged-attention wrapper built in CUDA-graph mode and bound to
    those buffers,
  - the captured ``torch.cuda.CUDAGraph`` and its static output logits.

Each decode iteration copies live request data into the persistent buffers,
re-``plan()``s the wrapper (outside the graph), replays the graph, then reads
the real (un-padded) prefix of the static logits. Sampling stays outside the
graph and is unchanged.

Batches of ``B' < B`` are padded up to the bucket with dummy requests whose KV
reads/writes are confined to a reserved dummy page, so padding never corrupts
real KV cache and its output rows are simply discarded.

Capture is lazy (first eligible decode-only batch) because the pool is built
lazily in pure colocated mode. Capture and replay both run on the worker's
single GPU-queue thread, on a consistent stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sangam.kv_cache.paged_kv_cache import PagedAttentionState, PagedKVPool
from sangam.logger import init_logger
from sangam.model.model_runner import (
    MixedBatchForwardResult,
    _shift_logits_right_per_item,
    paged_attention_context,
)

logger = init_logger(__name__)


@dataclass
class DecodeGraphItem:
    """One real decode request's inputs for a graph forward.

    ``token_ids`` are the ``block_length`` query tokens; ``block_start`` is the
    absolute position of the block within the sequence; ``active_kv_len`` is the
    request's current KV length; ``page_ids`` are the request's allocated pages.
    """

    token_ids: list[int]
    block_start: int
    active_kv_len: int
    page_ids: list[int]


class _Bucket:
    """Per-batch-size static buffers, graph-mode wrapper, and captured graph."""

    def __init__(
        self,
        *,
        batch_size: int,
        block_length: int,
        max_indices: int,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.block_length = block_length
        q = batch_size * block_length
        self.total_q = q

        # Static input/scatter/rope buffers (persistent addresses).
        self.input_ids = torch.zeros(1, q, dtype=torch.long, device=device)
        self.rope_positions = torch.zeros(q, dtype=torch.long, device=device)
        self.scatter_page_ids = torch.zeros(q, dtype=torch.long, device=device)
        self.scatter_offsets = torch.zeros(q, dtype=torch.long, device=device)

        # FlashInfer persistent index buffers (graph-mode wrapper writes here).
        self.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.kv_indices = torch.zeros(max_indices, dtype=torch.int32, device=device)
        self.kv_last_page_len = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )

        # Pinned host staging buffers (allocated once). Each replay fills these
        # on the host, then issues async H2D copies into the graph-input device
        # buffers and hands the host index tensors to wrapper.plan(). Passing
        # host tensors makes plan()'s internal `.to("cpu")` calls no-ops (no
        # device->host sync) while it still copies them into its bound buffers.
        self.h_input_ids = torch.zeros(q, dtype=torch.long, pin_memory=True)
        self.h_rope_positions = torch.zeros(q, dtype=torch.long, pin_memory=True)
        self.h_scatter_page_ids = torch.zeros(q, dtype=torch.long, pin_memory=True)
        self.h_scatter_offsets = torch.zeros(q, dtype=torch.long, pin_memory=True)
        self.h_qo_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, pin_memory=True
        )
        self.h_kv_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, pin_memory=True
        )
        self.h_kv_indices = torch.zeros(max_indices, dtype=torch.int32, pin_memory=True)
        self.h_kv_last_page_len = torch.zeros(
            batch_size, dtype=torch.int32, pin_memory=True
        )

        self.wrapper = None
        self.state: PagedAttentionState | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_logits: torch.Tensor | None = None


class DecodeCudaGraphRunner:
    """Captures and replays CUDA graphs for colocated decode-only batches."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        pool: PagedKVPool,
        flashinfer_workspace: torch.Tensor,
        device: torch.device,
        block_length: int,
        num_q_heads: int,
        buckets: tuple[int, ...],
        rope_max_pos: int,
    ):
        self.model = model
        self.pool = pool
        self.workspace = flashinfer_workspace
        self.device = device
        self.block_length = block_length
        self.num_q_heads = num_q_heads
        self.buckets = tuple(sorted(set(buckets)))
        self.max_bucket = max(self.buckets)
        self.rope_max_pos = rope_max_pos

        self.page_size = pool.page_size
        # Dummy pages absorb all padding KV writes/reads; one block's worth.
        self._dummy_pages_per_req = self.dummy_pages_per_req(
            block_length, self.page_size
        )

        self._dummy_page_ids: list[int] | None = None
        self._captured = False
        # Shared across all bucket captures: one CUDA-graph memory pool so the
        # buckets reuse activation scratch instead of each holding a private
        # pool, and one logits buffer (owned by the largest bucket) that smaller
        # buckets write into the prefix of.
        self._graph_pool = None
        self._shared_logits: torch.Tensor | None = None

    @staticmethod
    def dummy_pages_per_req(block_length: int, page_size: int) -> int:
        """Pages reserved out of the KV pool to back captured-graph padding.

        Reserved once at capture and never freed, so callers (e.g. capacity
        advertised at worker registration) must account for them.
        """
        return math.ceil(block_length / page_size)

    # ----- eligibility -----

    def can_run(self, num_active: int, block_length: int) -> bool:
        return (
            num_active > 0
            and block_length == self.block_length
            and num_active <= self.max_bucket
        )

    def select_bucket(self, num_active: int) -> int:
        for b in self.buckets:
            if b >= num_active:
                return b
        raise ValueError(
            f"num_active={num_active} exceeds max bucket {self.max_bucket}"
        )

    # ----- capture -----

    def maybe_capture(self) -> None:
        """Capture one graph per bucket. Idempotent; runs on the GPU thread."""
        if self._captured:
            return
        self._reserve_dummy_pages()
        # One pool handle shared by every bucket's capture; reuses activation
        # scratch across buckets instead of allocating a private pool per graph.
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._buckets_by_size: dict[int, _Bucket] = {}
        # Capture largest-first so the first bucket allocates the shared logits
        # buffer at its full size and smaller buckets reuse its prefix.
        for b in sorted(self.buckets, reverse=True):
            # Upper bound on flat page indices for a full bucket: real requests'
            # KV may span the whole pool (<= max_pages), and each of the up-to-b
            # dummy padding items adds _dummy_pages_per_req indices on top.
            max_indices = self.pool.max_pages + b * self._dummy_pages_per_req
            bucket = _Bucket(
                batch_size=b,
                block_length=self.block_length,
                max_indices=max_indices,
                device=self.device,
            )
            self._capture_bucket(bucket)
            self._buckets_by_size[b] = bucket
            logger.debug(
                "Captured decode CUDA graph for batch_size=%d (q_tokens=%d)",
                b,
                bucket.total_q,
            )
        logger.info(
            "Captured %d decode CUDA graphs for batch sizes %s",
            len(self.buckets),
            list(self.buckets),
        )
        self._captured = True

    def _reserve_dummy_pages(self) -> None:
        # Reserve dummy pages once; never freed, never handed to real requests.
        self._dummy_page_ids = self.pool.allocator.allocate(self._dummy_pages_per_req)

    def _capture_bucket(self, bucket: _Bucket) -> None:
        import flashinfer

        bucket.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=bucket.qo_indptr,
            paged_kv_indptr_buf=bucket.kv_indptr,
            paged_kv_indices_buf=bucket.kv_indices,
            paged_kv_last_page_len_buf=bucket.kv_last_page_len,
        )
        bucket.state = PagedAttentionState.from_static_buffers(
            pool=self.pool,
            wrapper=bucket.wrapper,
            scatter_page_ids=bucket.scatter_page_ids,
            scatter_offsets=bucket.scatter_offsets,
            rope_positions=bucket.rope_positions,
            rope_max_pos=self.rope_max_pos,
        )

        # Fill buffers with an all-dummy batch so plan()/capture see valid data.
        dummy_items = [self._dummy_item() for _ in range(bucket.batch_size)]
        self._fill_buffers_and_plan(bucket, dummy_items)

        with torch.inference_mode():
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    bucket.state.reset_layers()
                    with paged_attention_context(self.model, bucket.state, None):
                        self.model(bucket.input_ids)
            torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
            torch.cuda.synchronize(self.device)

            # Redirect logits into the single shared buffer during capture only,
            # so the copy (for smaller buckets) is captured into the graph and
            # replayed. The largest bucket (captured first) donates its in-graph
            # logits as the shared buffer with no copy; every smaller bucket
            # copies its logits into the buffer's contiguous (1, q, vocab) prefix
            # and rebinds output.logits to that view, so only one logits tensor
            # is retained across all graphs.
            graph = torch.cuda.CUDAGraph()
            bucket.state.reset_layers()
            bucket.state.clear_rope_cache()
            handle = self.model.register_forward_hook(self._redirect_logits_hook)
            try:
                with paged_attention_context(self.model, bucket.state, None):
                    with torch.cuda.graph(graph, pool=self._graph_pool):
                        output = self.model(bucket.input_ids)
            finally:
                handle.remove()
            bucket.graph = graph
            bucket.static_logits = output.logits
            torch.cuda.synchronize(self.device)

    def _redirect_logits_hook(self, _module, _inputs, output):
        """Point output.logits at the shared logits buffer during capture.

        The first (largest) bucket donates its freshly allocated in-graph buffer;
        later (smaller) buckets copy into its prefix and rebind. The donated
        buffer is kept alive via ``self._shared_logits`` so the shared graph pool
        never reuses its memory for another bucket's activations.
        """
        if self._shared_logits is None:
            self._shared_logits = output.logits
            return output
        q = output.logits.shape[1]
        dst = self._shared_logits[:, :q, :]
        dst.copy_(output.logits)
        output.logits = dst
        return output

    def _dummy_item(self) -> DecodeGraphItem:
        # All-zero tokens, RoPE position 0, KV confined to the dummy page(s).
        active = self.block_length
        return DecodeGraphItem(
            token_ids=[0] * self.block_length,
            block_start=0,
            active_kv_len=active,
            page_ids=list(self._dummy_page_ids),
        )

    # ----- replay -----

    def run(self, items: list[DecodeGraphItem]) -> MixedBatchForwardResult:
        """Replay the graph for ``items`` and return per-item logits views.

        Pads up to the selected bucket with dummy requests; returns logits for
        the real items only, with the Dream logit shift applied post-replay.
        """
        b_real = len(items)
        bucket_size = self.select_bucket(b_real)
        bucket = self._buckets_by_size[bucket_size]

        padded = list(items) + [self._dummy_item() for _ in range(bucket_size - b_real)]
        with torch.inference_mode():
            self._fill_buffers_and_plan(bucket, padded)
            bucket.graph.replay()

        block_length = self.block_length
        real_tokens = b_real * block_length
        # Slice the real (un-padded) prefix, then apply logit shift if needed.
        packed_logits = bucket.static_logits[:, :real_tokens, :]
        query_offsets = [i * block_length for i in range(b_real + 1)]
        if getattr(self.model, "requires_logit_shift", False):
            packed_logits = _shift_logits_right_per_item(
                packed_logits.clone(), query_offsets
            )
        item_logits = [
            packed_logits[:, s:e, :]
            for s, e in zip(query_offsets[:-1], query_offsets[1:], strict=True)
        ]
        return MixedBatchForwardResult(
            packed_logits=packed_logits,
            query_offsets=query_offsets,
            item_logits=item_logits,
            operation_metrics_seconds=None,
        )

    def _fill_buffers_and_plan(
        self, bucket: _Bucket, items: list[DecodeGraphItem]
    ) -> None:
        """Write per-request indices into persistent buffers and re-plan.

        Mirrors ``PagedAttentionState.__init__`` index construction but targets
        the bucket's persistent buffers. Runs outside graph capture/replay.
        """
        block_length = self.block_length
        page_size = self.page_size

        token_ids: list[int] = []
        rope_positions: list[int] = []
        scatter_page_ids: list[int] = []
        scatter_offsets: list[int] = []
        q_lengths: list[int] = []
        page_counts: list[int] = []
        last_page_lengths: list[int] = []
        all_page_ids: list[int] = []

        for it in items:
            block_start = it.block_start
            block_end = block_start + block_length
            token_ids.extend(it.token_ids)
            q_lengths.append(block_length)
            for pos in range(block_start, block_end):
                rope_positions.append(pos)
                scatter_page_ids.append(it.page_ids[pos // page_size])
                scatter_offsets.append(pos % page_size)
            page_count = math.ceil(it.active_kv_len / page_size)
            page_counts.append(page_count)
            all_page_ids.extend(it.page_ids[:page_count])
            last_page_lengths.append(it.active_kv_len % page_size or page_size)

        if len(all_page_ids) > bucket.kv_indices.numel():
            raise RuntimeError(
                "paged_kv_indices exceeds captured buffer "
                f"({len(all_page_ids)} > {bucket.kv_indices.numel()})"
            )

        qo = [0]
        for ql in q_lengths:
            qo.append(qo[-1] + ql)
        kvip = [0]
        for pc in page_counts:
            kvip.append(kvip[-1] + pc)

        n_q = len(token_ids)
        n_idx = len(all_page_ids)

        # Fill the pinned host buffers (cheap host memcpy; no device alloc). The
        # token/scatter/rope arrays span the full bucket (n_q == total_q); only
        # kv_indices is variable-length per call.
        bucket.h_input_ids[:n_q].copy_(torch.tensor(token_ids, dtype=torch.long))
        bucket.h_rope_positions[:n_q].copy_(
            torch.tensor(rope_positions, dtype=torch.long)
        )
        bucket.h_scatter_page_ids[:n_q].copy_(
            torch.tensor(scatter_page_ids, dtype=torch.long)
        )
        bucket.h_scatter_offsets[:n_q].copy_(
            torch.tensor(scatter_offsets, dtype=torch.long)
        )
        bucket.h_qo_indptr.copy_(torch.tensor(qo, dtype=torch.int32))
        bucket.h_kv_indptr.copy_(torch.tensor(kvip, dtype=torch.int32))
        bucket.h_kv_indices[:n_idx].copy_(torch.tensor(all_page_ids, dtype=torch.int32))
        bucket.h_kv_last_page_len.copy_(
            torch.tensor(last_page_lengths, dtype=torch.int32)
        )

        # Async H2D into the graph-input device buffers from the pinned source.
        bucket.input_ids[0].copy_(bucket.h_input_ids[:n_q], non_blocking=True)
        bucket.rope_positions.copy_(bucket.h_rope_positions[:n_q], non_blocking=True)
        bucket.scatter_page_ids.copy_(
            bucket.h_scatter_page_ids[:n_q], non_blocking=True
        )
        bucket.scatter_offsets.copy_(bucket.h_scatter_offsets[:n_q], non_blocking=True)

        # plan() with HOST index tensors: its internal `.to("cpu")` becomes a
        # no-op (no device->host sync), while it still copies them H2D into the
        # bound persistent buffers. plan() still runs every replay.
        bucket.wrapper.plan(
            qo_indptr=bucket.h_qo_indptr,
            paged_kv_indptr=bucket.h_kv_indptr,
            paged_kv_indices=bucket.h_kv_indices[:n_idx],
            paged_kv_last_page_len=bucket.h_kv_last_page_len,
            num_qo_heads=self.num_q_heads,
            num_kv_heads=self.pool.num_kv_heads,
            head_dim_qk=self.pool.head_dim,
            head_dim_vo=self.pool.head_dim,
            page_size=page_size,
            causal=False,
            q_data_type=self.pool.dtype,
            kv_data_type=self.pool.dtype,
        )
