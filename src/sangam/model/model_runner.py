"""Packed mixed-batch packing and forward execution for the model."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch

from sangam.kv_cache.paged_kv_cache import PagedAttentionState, RequestKVState


MixedBatchPhase = Literal["prefill", "decode"]


@dataclass
class MixedBatchItem:
    """Per-item metadata for one packed model execution."""

    request_id: str
    token_ids: Sequence[int]
    query_start: int
    query_end: int
    active_kv_len: int
    kv_state: RequestKVState
    phase: MixedBatchPhase

    @property
    def query_len(self) -> int:
        return self.query_end - self.query_start


@dataclass
class PackedMixedBatch:
    """Packed model inputs and paged-attention metadata for one forward."""

    items: list[MixedBatchItem]
    input_ids: torch.Tensor
    query_offsets: list[int]
    paged_state: PagedAttentionState


@dataclass
class MixedBatchForwardResult:
    """Packed logits and per-item logits views for a mixed forward pass."""

    packed_logits: torch.Tensor
    query_offsets: list[int]
    item_logits: list[torch.Tensor]
    operation_metrics_seconds: dict[str, float] | None = None


@dataclass
class ModelOperationMetricsContext:
    """Per-forward operation metrics collection context."""

    enabled: bool
    profile_layer_id: int
    num_layers: int
    _sampled_seconds: dict[str, float]

    @classmethod
    def create(
        cls,
        enabled: bool,
        profile_layer_id: int,
        num_layers: int,
    ) -> "ModelOperationMetricsContext":
        return cls(
            enabled=enabled,
            profile_layer_id=profile_layer_id,
            num_layers=num_layers,
            _sampled_seconds={},
        )

    def record_sample(
        self, metric_name: str, layer_id: int, elapsed_seconds: float
    ) -> None:
        if not self.enabled:
            return
        if layer_id != self.profile_layer_id:
            return
        if elapsed_seconds <= 0.0:
            return
        self._sampled_seconds[metric_name] = (
            self._sampled_seconds.get(metric_name, 0.0) + elapsed_seconds
        )

    def scaled_seconds(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {
            name: elapsed_s * self.num_layers
            for name, elapsed_s in self._sampled_seconds.items()
        }


def measure_operation_time(
    op_metrics_context: "ModelOperationMetricsContext | None",
    layer_id: int,
    metric_name: str,
    fn: Callable[[], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Time `fn()` and record it into the op-metrics context.

    Returns `fn()` verbatim (no timing) when the context is absent/disabled or
    `layer_id` is not the profiled layer, so non-profiling and CUDA-graph paths
    are unaffected. On the profiled layer the CUDA path issues an in-forward
    `end_event.synchronize()`, so it must not be used under graph capture/replay.
    """
    if (
        op_metrics_context is None
        or not op_metrics_context.enabled
        or layer_id != op_metrics_context.profile_layer_id
    ):
        return fn()

    if device.type == "cuda":
        with torch.cuda.device(device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = fn()
            end_event.record()
            end_event.synchronize()
        elapsed_seconds = start_event.elapsed_time(end_event) / 1e3
    else:
        start = time.perf_counter()
        output = fn()
        elapsed_seconds = max(0.0, time.perf_counter() - start)

    op_metrics_context.record_sample(
        metric_name=metric_name,
        layer_id=layer_id,
        elapsed_seconds=elapsed_seconds,
    )
    return output


def _iter_transformer_blocks(model: torch.nn.Module):
    """Yield the modules on which to install `_paged_attn_state`.

    For LLaDA each block carries the attention method itself, so we yield the
    block. For Dream the attention lives in `decoder_layer.self_attn`, so we
    yield that submodule.
    """
    inner = model.model
    if hasattr(inner, "transformer"):
        transformer = inner.transformer
        if hasattr(transformer, "blocks"):
            yield from transformer.blocks
            return
        if hasattr(transformer, "block_groups"):
            for group in transformer.block_groups:
                yield from group
            return
    if hasattr(inner, "layers"):
        for layer in inner.layers:
            yield layer.self_attn
        return
    raise RuntimeError("Unable to locate transformer blocks for paged attention state")


def _iter_op_metrics_blocks(model: torch.nn.Module):
    """Yield the modules on which to install `_op_metrics_context`.

    For LLaDA this is the same block that carries the paged-attention state. For
    Dream the qkv/attn ops live in `decoder_layer.self_attn` while the MLP op is
    timed at its call site on the parent decoder layer, so both must receive the
    context (the paged state still goes only to `self_attn`).
    """
    inner = model.model
    if hasattr(inner, "transformer"):
        transformer = inner.transformer
        if hasattr(transformer, "blocks"):
            yield from transformer.blocks
            return
        if hasattr(transformer, "block_groups"):
            for group in transformer.block_groups:
                yield from group
            return
    if hasattr(inner, "layers"):
        for layer in inner.layers:
            yield layer
            yield layer.self_attn
        return
    raise RuntimeError("Unable to locate transformer blocks for op metrics context")


@contextmanager
def paged_attention_context(
    model: torch.nn.Module,
    paged_state: PagedAttentionState,
    op_metrics_context: ModelOperationMetricsContext | None,
):
    paged_blocks = list(_iter_transformer_blocks(model))
    op_metrics_blocks = list(_iter_op_metrics_blocks(model))
    for block in paged_blocks:
        block._paged_attn_state = paged_state
    for block in op_metrics_blocks:
        block._op_metrics_context = op_metrics_context
    try:
        yield
    finally:
        for block in paged_blocks:
            block._paged_attn_state = None
        for block in op_metrics_blocks:
            block._op_metrics_context = None


def pack_mixed_batch(
    *,
    items: list[MixedBatchItem],
    pool,
    num_q_heads: int,
    device: torch.device,
    workspace: torch.Tensor,
    kv_page_callback: (
        Callable[[int, torch.Tensor, list[RequestKVState]], None] | None
    ) = None,
) -> PackedMixedBatch:
    """Pack mixed query tokens and build paged-attention state for one model call."""
    if not items:
        raise ValueError("items must not be empty")

    query_offsets = [0]
    packed_token_ids: list[int] = []
    for item in items:
        if item.query_len <= 0:
            raise ValueError(f"query span must be non-empty for {item.request_id}")
        if len(item.token_ids) != item.query_len:
            raise ValueError(
                f"token_ids/query span length mismatch for {item.request_id}: "
                f"len(token_ids)={len(item.token_ids)}, query_len={item.query_len}"
            )
        packed_token_ids.extend(item.token_ids)
        query_offsets.append(query_offsets[-1] + item.query_len)

    input_ids = torch.tensor([packed_token_ids], dtype=torch.long, device=device)
    paged_state = PagedAttentionState(
        pool=pool,
        requests=[item.kv_state for item in items],
        block_starts=[item.query_start for item in items],
        block_ends=[item.query_end for item in items],
        active_kv_lens=[item.active_kv_len for item in items],
        num_q_heads=num_q_heads,
        device=device,
        workspace=workspace,
        kv_page_callback=kv_page_callback,
    )
    return PackedMixedBatch(
        items=items,
        input_ids=input_ids,
        query_offsets=query_offsets,
        paged_state=paged_state,
    )


def _shift_logits_right_per_item(
    packed_logits: torch.Tensor,
    query_offsets: list[int],
) -> torch.Tensor:
    """Shift right by 1 within each item's query span, in-place.

    Dream's lm_head predicts position i+1 from hidden_states[i], so the
    prediction at position i is the model's raw output at position i-1. The
    first position of each item keeps its raw logit (off-by-one fallback at
    item boundaries; correctable later by widening the query span on the left).

    Per-item slice clone (not whole-tensor clone) keeps peak temp bounded by
    the largest item span; for Dream's 151936-token vocab a whole-tensor clone
    of a prefill batch would be over a gigabyte at bf16.
    """
    for start, end in zip(query_offsets[:-1], query_offsets[1:], strict=True):
        if end - start >= 2:
            src = packed_logits[:, start : end - 1, :].clone()
            packed_logits[:, start + 1 : end, :] = src
    return packed_logits


@torch.inference_mode()
def run_mixed_paged_forward(
    model: torch.nn.Module,
    packed_batch: PackedMixedBatch,
    op_metrics_context: ModelOperationMetricsContext | None = None,
) -> MixedBatchForwardResult:
    """Execute one packed forward and return per-item logits views."""
    with paged_attention_context(model, packed_batch.paged_state, op_metrics_context):
        output = model(packed_batch.input_ids)

    packed_logits = output.logits
    if getattr(model, "requires_logit_shift", False):
        packed_logits = _shift_logits_right_per_item(
            packed_logits, packed_batch.query_offsets
        )
    item_logits = [
        packed_logits[:, start:end, :]
        for start, end in zip(
            packed_batch.query_offsets[:-1],
            packed_batch.query_offsets[1:],
            strict=True,
        )
    ]
    return MixedBatchForwardResult(
        packed_logits=packed_logits,
        query_offsets=packed_batch.query_offsets,
        item_logits=item_logits,
        operation_metrics_seconds=(
            op_metrics_context.scaled_seconds()
            if op_metrics_context is not None and op_metrics_context.enabled
            else None
        ),
    )
