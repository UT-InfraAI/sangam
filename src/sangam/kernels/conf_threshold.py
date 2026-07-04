"""Fused argmax + max-softmax-confidence kernel for the temp==0 conf_threshold path.

For ``temperature == 0`` with the ``conf_threshold`` unmasking strategy, the
proposed token is exactly ``argmax(logits, dim=-1)`` (no Gumbel noise, mask token
not disallowed). Its confidence is therefore the maximum softmax probability,
which equals ``1 / sum_v exp(logit_v - max_logit)``. This module computes both the
argmax token id and that confidence in a single online-softmax reduction pass over
the vocab dimension, replacing the reference path's full ``float64`` softmax over V
plus gather plus a separate argmax (see ``sangam.sampler._compute_proposal_confidence``).

Numerics: the reduction accumulates in ``float64`` to closely track the reference
``F.softmax(logits.to(torch.float64))``. Differences are limited to the final
``1 / sum_exp`` rounding, far below any realistic confidence threshold margin.

Tie-breaking: ``torch.argmax`` returns the *first* maximal index. The kernel
reproduces this by taking, within each vocab tile, the smallest index among the
tile-maximal positions, and only adopting a tile's argmax when its max is strictly
greater than the running max so the globally-first index wins across tiles.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - environments without triton
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _argmax_conf_kernel(
        logits_ptr,
        token_ids_ptr,
        confidence_ptr,
        V,
        stride_r,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = logits_ptr + row * stride_r

        m = tl.full((), -float("inf"), dtype=tl.float64)
        arg = tl.zeros((), dtype=tl.int64)
        s = tl.zeros((), dtype=tl.float64)

        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float64)

            tile_max = tl.max(x, axis=0)
            is_max = x == tile_max
            tile_arg = tl.min(tl.where(is_max, offs, V), axis=0).to(tl.int64)

            new_m = tl.maximum(m, tile_max)
            s = s * tl.exp(m - new_m) + tl.sum(
                tl.where(mask, tl.exp(x - new_m), 0.0), axis=0
            )
            arg = tl.where(tile_max > m, tile_arg, arg)
            m = new_m

        tl.store(token_ids_ptr + row, arg.to(tl.int64))
        tl.store(confidence_ptr + row, 1.0 / s)


def _torch_fallback(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch equivalent of the kernel (used on CPU / when triton is absent)."""
    token_ids = torch.argmax(logits, dim=-1)
    max_logit = logits.max(dim=-1, keepdim=True).values
    sum_exp = torch.exp((logits - max_logit).to(torch.float64)).sum(dim=-1)
    return token_ids, 1.0 / sum_exp


def fused_argmax_confidence(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (argmax token ids, max-softmax confidence) for logits of shape (B, L, V).

    token ids are int64 (B, L); confidence is float64 (B, L). Falls back to a pure
    torch implementation when triton is unavailable or the tensor is not on CUDA.
    """
    if logits.dim() != 3:
        raise ValueError("logits must have shape (batch, block_length, vocab)")

    if not (_HAS_TRITON and logits.is_cuda):
        return _torch_fallback(logits)

    batch_size, block_length, vocab_size = logits.shape
    flat = logits.reshape(batch_size * block_length, vocab_size).contiguous()
    token_ids = torch.empty(
        batch_size * block_length, dtype=torch.int64, device=logits.device
    )
    confidence = torch.empty(
        batch_size * block_length, dtype=torch.float64, device=logits.device
    )

    _argmax_conf_kernel[(batch_size * block_length,)](
        flat,
        token_ids,
        confidence,
        vocab_size,
        flat.stride(0),
        BLOCK_V=8192,
        num_warps=8,
    )

    return (
        token_ids.view(batch_size, block_length),
        confidence.view(batch_size, block_length),
    )
