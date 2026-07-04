"""Paged KV cache pool and FlashInfer attention state for paged attention.

Eliminates per-request dense KV tensors, padding, and copying by storing
all KV cache entries in a shared page pool on GPU. FlashInfer's
BatchPrefillWithPagedKVCacheWrapper handles variable-length KV natively.

KV layout per layer: (max_pages, 2, page_size, num_kv_heads, head_dim)
  - dim 1: [K, V] interleaved
  - K values are stored WITH RoPE pre-applied
"""

import importlib
import math
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable

import torch

from sangam.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _load_flashinfer():
    return importlib.import_module("flashinfer")


class _LazyFlashinferAttr:
    def __init__(self, attr_name: str) -> None:
        self._attr_name = attr_name

    def __call__(self, *args, **kwargs):
        return getattr(_load_flashinfer(), self._attr_name)(*args, **kwargs)


flashinfer = SimpleNamespace(
    BatchPrefillWithPagedKVCacheWrapper=_LazyFlashinferAttr(
        "BatchPrefillWithPagedKVCacheWrapper"
    ),
    apply_rope_with_cos_sin_cache_inplace=_LazyFlashinferAttr(
        "apply_rope_with_cos_sin_cache_inplace"
    ),
)


class PageAllocator:
    """Simple free-list page allocator (CPU-side bookkeeping)."""

    def __init__(self, max_pages: int):
        self._free: list[int] = list(range(max_pages - 1, -1, -1))  # stack
        self._max_pages = max_pages

    def allocate(self, n: int) -> list[int]:
        if len(self._free) < n:
            raise RuntimeError(
                f"KV page OOM: need {n} pages, only {len(self._free)} free "
                f"out of {self._max_pages}"
            )
        pages = self._free[-n:]
        del self._free[-n:]
        return pages

    def free(self, pages: list[int]) -> None:
        self._free.extend(pages)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self._max_pages - len(self._free)


class PagedKVPool:
    """Pre-allocated GPU memory pool for paged KV cache.

    All decode requests on this worker share the same pool. Pages are
    allocated when KV cache is received from prefill and freed when the
    decode block completes.
    """

    def __init__(
        self,
        num_layers: int,
        max_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        zero_init: bool = True,
    ):
        self.num_layers = num_layers
        self.max_pages = max_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.zero_init = zero_init

        # Per-layer KV data in FlashInfer NHD page layout
        tensor_factory = torch.zeros if zero_init else torch.empty
        self.kv_data: list[torch.Tensor] = [
            tensor_factory(
                max_pages,
                2,
                page_size,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self.allocator = PageAllocator(max_pages)

        total_bytes = (
            num_layers
            * max_pages
            * 2
            * page_size
            * num_kv_heads
            * head_dim
            * dtype.itemsize
        )
        logger.info(
            f"PagedKVPool: {num_layers} layers, {max_pages} pages x {page_size} tokens, "
            f"{num_kv_heads} heads x {head_dim}d, "
            f"total {total_bytes / (1024**3):.2f} GB on {device}"
        )

    def allocate(self, seq_len: int) -> tuple[list[int], int]:
        """Allocate pages for a sequence. Returns (page_ids, last_page_len)."""
        num_pages = math.ceil(seq_len / self.page_size)
        page_ids = self.allocator.allocate(num_pages)
        last_page_len = seq_len % self.page_size or self.page_size
        return page_ids, last_page_len

    def free(self, page_ids: list[int]) -> None:
        self.allocator.free(page_ids)

    def write_dense_kv(
        self,
        layer_idx: int,
        page_ids: list[int],
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
    ) -> None:
        """Copy dense NHD tensors (seq_len, num_kv_heads, head_dim) into pages."""
        kv = self.kv_data[layer_idx]
        ps = self.page_size
        for i, pid in enumerate(page_ids):
            start = i * ps
            end = min(start + ps, seq_len)
            length = end - start
            kv[pid, 0, :length] = k[start:end]
            kv[pid, 1, :length] = v[start:end]


@dataclass
class RequestKVState:
    """Per-request KV cache state referencing pages in the pool."""

    page_ids: list[int]
    seq_len: int
    last_page_len: int


class PagedAttentionState:
    """Batch-level state for a single FlashInfer paged attention forward pass.

    Created once per batched forward, set on model blocks, consumed
    layer-by-layer, then discarded.
    """

    def __init__(
        self,
        pool: PagedKVPool,
        requests: list[RequestKVState],
        block_starts: list[int],
        block_ends: list[int],
        active_kv_lens: list[int] | None,
        num_q_heads: int,
        device: torch.device,
        workspace: torch.Tensor,
        kv_page_callback: Callable[[int, torch.Tensor, list[RequestKVState]], None]
        | None = None,
    ):
        self.pool = pool
        self.requests = requests
        self.current_layer_idx = 0
        self.kv_page_callback = kv_page_callback
        self._rope_cache: dict[
            tuple[torch.dtype, bool], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._cos_sin_cache: torch.Tensor | None = None

        num_reqs = len(requests)
        if active_kv_lens is None:
            active_kv_lens = [request.seq_len for request in requests]
        if len(block_starts) != num_reqs or len(block_ends) != num_reqs:
            raise ValueError("block_starts/block_ends must align with requests")
        if len(active_kv_lens) != num_reqs:
            raise ValueError("active_kv_lens must align with requests")
        q_lengths = [end - start for start, end in zip(block_starts, block_ends)]
        self.total_q_tokens = sum(q_lengths)

        # Q indptr
        qo_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.tensor(
            q_lengths, dtype=torch.int32, device=device
        ).cumsum(0)

        # Paged KV indptr
        page_counts: list[int] = []
        last_page_lengths: list[int] = []
        for request, active_kv_len, block_end in zip(
            requests, active_kv_lens, block_ends, strict=True
        ):
            if active_kv_len < block_end:
                raise ValueError(
                    "active_kv_len must cover the query span being written: "
                    f"active_kv_len={active_kv_len}, block_end={block_end}"
                )
            if active_kv_len > request.seq_len:
                raise ValueError(
                    "active_kv_len cannot exceed allocated request length: "
                    f"active_kv_len={active_kv_len}, seq_len={request.seq_len}"
                )
            page_count = math.ceil(active_kv_len / pool.page_size)
            if page_count > len(request.page_ids):
                raise ValueError(
                    "active_kv_len requires more pages than allocated: "
                    f"active_kv_len={active_kv_len}, pages={len(request.page_ids)}"
                )
            page_counts.append(page_count)
            if active_kv_len == 0:
                last_page_lengths.append(0)
            else:
                last_page_lengths.append(
                    active_kv_len % pool.page_size or pool.page_size
                )
        paged_kv_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        paged_kv_indptr[1:] = torch.tensor(
            page_counts, dtype=torch.int32, device=device
        ).cumsum(0)

        # Flat page indices
        all_page_ids: list[int] = []
        for request, page_count in zip(requests, page_counts, strict=True):
            all_page_ids.extend(request.page_ids[:page_count])
        paged_kv_indices = torch.tensor(all_page_ids, dtype=torch.int32, device=device)

        last_page_len = torch.tensor(
            last_page_lengths, dtype=torch.int32, device=device
        )

        # Pre-compute scatter indices for writing new K,V to pages.
        # Maps each Q token to (page_id, offset_within_page).
        flat_page_ids: list[int] = []
        flat_offsets: list[int] = []
        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            for pos in range(start, end):
                page_local_idx = pos // pool.page_size
                page_offset = pos % pool.page_size
                actual_page_id = requests[i].page_ids[page_local_idx]
                flat_page_ids.append(actual_page_id)
                flat_offsets.append(page_offset)

        self.scatter_page_ids = torch.tensor(
            flat_page_ids, dtype=torch.long, device=device
        )
        self.scatter_offsets = torch.tensor(
            flat_offsets, dtype=torch.long, device=device
        )

        # Pre-compute RoPE position indices for Q and new K
        rope_positions: list[int] = []
        for start, end in zip(block_starts, block_ends):
            rope_positions.extend(range(start, end))
        self.rope_positions = torch.tensor(
            rope_positions, dtype=torch.long, device=device
        )
        self.rope_max_pos = int(self.rope_positions.max().item()) + 1

        # Plan FlashInfer wrapper (bidirectional attention)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        self.wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_q_heads,
            num_kv_heads=pool.num_kv_heads,
            head_dim_qk=pool.head_dim,
            head_dim_vo=pool.head_dim,
            page_size=pool.page_size,
            causal=False,
            q_data_type=pool.dtype,
            kv_data_type=pool.dtype,
        )

    @classmethod
    def from_static_buffers(
        cls,
        *,
        pool: PagedKVPool,
        wrapper,
        scatter_page_ids: torch.Tensor,
        scatter_offsets: torch.Tensor,
        rope_positions: torch.Tensor,
        rope_max_pos: int,
    ) -> "PagedAttentionState":
        """Build a CUDA-graph-friendly state backed by persistent buffers.

        Unlike ``__init__`` this performs no per-call allocation and no host
        syncs (no ``.item()``): the FlashInfer ``wrapper`` is constructed once
        in CUDA-graph mode and re-``plan()``ed by the caller before each
        replay, and ``scatter_*``/``rope_positions`` are persistent tensors the
        caller updates in place. ``rope_max_pos`` is a fixed bound on the RoPE
        table size (never derived from data), so the per-layer RoPE gather is
        graph-safe.

        The caller owns capture/replay ordering and must call
        ``reset_layers()`` before each forward and ``clear_rope_cache()`` before
        a capture so the gather is recorded inside the graph.
        """
        state = cls.__new__(cls)
        state.pool = pool
        state.requests = []
        state.current_layer_idx = 0
        state.kv_page_callback = None
        state._rope_cache = {}
        state._cos_sin_cache = None
        state.wrapper = wrapper
        state.scatter_page_ids = scatter_page_ids
        state.scatter_offsets = scatter_offsets
        state.rope_positions = rope_positions
        state.rope_max_pos = rope_max_pos
        state.total_q_tokens = int(rope_positions.numel())
        return state

    def reset_layers(self) -> None:
        """Reset the per-forward layer counter (graph capture/replay reuse)."""
        self.current_layer_idx = 0

    def clear_rope_cache(self) -> None:
        """Drop cached RoPE tensors so the next gather is recomputed.

        Called before a graph capture so the RoPE gather is recorded inside the
        captured region rather than reusing a warmup-time result.
        """
        self._rope_cache = {}
        self._cos_sin_cache = None

    def next_layer_idx(self, expected_layer_idx: int | None = None) -> int:
        """Return the current layer index and validate ordering when requested."""
        if (
            expected_layer_idx is not None
            and self.current_layer_idx != expected_layer_idx
        ):
            raise RuntimeError(
                f"Paged attention layer order mismatch: expected {expected_layer_idx}, "
                f"got {self.current_layer_idx}"
            )
        return self.current_layer_idx

    def finish_layer(self, layer_idx: int) -> None:
        """Advance to the next layer, ensuring the caller consumed the expected layer."""
        if self.current_layer_idx != layer_idx:
            raise RuntimeError(
                f"Paged attention layer advance mismatch: expected {layer_idx}, "
                f"got {self.current_layer_idx}"
            )
        self.current_layer_idx += 1

    def get_rope_tensors(
        self,
        rotary_embedding,
        tensor_dtype: torch.dtype,
        full_precision: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached RoPE tensors selected for this batch's active positions."""
        cache_key = (tensor_dtype, full_precision)
        cached = self._rope_cache.get(cache_key)
        if cached is not None:
            return cached

        pos_sin, pos_cos = rotary_embedding.get_rotary_embedding(
            self.rope_max_pos, device
        )
        sel_sin = pos_sin[0, 0, self.rope_positions].unsqueeze(1)
        sel_cos = pos_cos[0, 0, self.rope_positions].unsqueeze(1)
        if full_precision:
            cached = (sel_sin.float(), sel_cos.float())
        else:
            cached = (sel_sin.to(dtype=tensor_dtype), sel_cos.to(dtype=tensor_dtype))
        self._rope_cache[cache_key] = cached
        return cached

    def apply_rope_inplace(
        self,
        rotary_embedding,
        q: torch.Tensor,
        k: torch.Tensor,
        head_dim: int,
    ) -> None:
        """Fused in-place RoPE (FlashInfer) on NHD q/k at this batch's positions.

        q: (total_q_tokens, num_q_heads, head_dim),
        k: (total_q_tokens, num_kv_heads, head_dim). Both rotated in place.
        The fp32 cos_sin_cache (rope_max_pos, head_dim) is built once and cached,
        so it is graph-stable across replays.
        """
        if self._cos_sin_cache is None:
            # pos_sin/pos_cos: [1, 1, rope_max_pos, head_dim] where the head_dim
            # axis holds cat((freqs, freqs)), so [..., :half] are the unique values.
            pos_sin, pos_cos = rotary_embedding.get_rotary_embedding(
                self.rope_max_pos, q.device
            )
            half = head_dim // 2
            cos = pos_cos[0, 0, :, :half]
            sin = pos_sin[0, 0, :, :half]
            # cos_sin_cache: (rope_max_pos, head_dim) fp32, [cos | sin] on rotary_dim.
            self._cos_sin_cache = torch.cat((cos, sin), dim=-1).to(
                dtype=torch.float32, device=q.device
            )
        flashinfer.apply_rope_with_cos_sin_cache_inplace(
            self.rope_positions,
            q.reshape(q.shape[0], -1),
            k.reshape(k.shape[0], -1),
            head_dim,
            self._cos_sin_cache,
            True,
        )

    def update_kv_pages(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write new K,V to pages. K must have RoPE already applied.

        k, v: (total_q_tokens, num_kv_heads, head_dim) in NHD layout.
        """
        kv = self.pool.kv_data[layer_idx]
        kv[self.scatter_page_ids, 0, self.scatter_offsets] = k
        kv[self.scatter_page_ids, 1, self.scatter_offsets] = v
        if self.kv_page_callback is not None:
            self.kv_page_callback(
                layer_idx,
                kv,
                self.requests,
            )

    def run_attention(self, q: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Run FlashInfer attention for one layer.

        q: (total_q_tokens, num_q_heads, head_dim) in NHD layout.
        Returns: (total_q_tokens, num_q_heads, head_dim).
        """
        return self.wrapper.run(q, self.pool.kv_data[layer_idx])
