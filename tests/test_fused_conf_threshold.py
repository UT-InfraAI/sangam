"""Tests for the fused temp==0 + conf_threshold sampling fast path."""

import pytest
import torch
import torch.nn.functional as F

from sangam import sampler as sampler_module
from sangam.kernels import conf_threshold as kernel_module
from sangam.kernels import fused_argmax_confidence
from sangam.sampler import Sampler, SamplingRequest
from sangam.sampling_parameters import SamplingParameters

MASK_ID = 126336


def _threshold_request(request_id, seq_len, threshold, block_tokens_row):
    token_ids = block_tokens_row.unsqueeze(0)
    return SamplingRequest(
        request_id=request_id,
        request_seed=hash(request_id) % 1000,
        token_ids=token_ids,
        block_start_idx=0,
        block_end_index=seq_len,
        step_index=0,
        sampling_parameters=SamplingParameters(
            temperature=0.0,
            unmasking_strategy="conf_threshold",
            confidence_threshold=threshold,
        ),
        mask_token_id=MASK_ID,
    )


def _reference_threshold(logits, block_tokens, threshold):
    """Float64-softmax reference mirroring Sampler.sample_batch's threshold branch."""
    mask_index = block_tokens == MASK_ID
    proposed = torch.argmax(logits, dim=-1)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    proposal_confidence = torch.gather(p, dim=-1, index=proposed.unsqueeze(-1)).squeeze(
        -1
    )
    proposed = torch.where(mask_index, proposed, block_tokens)
    neg_inf = torch.full_like(
        proposal_confidence, torch.finfo(proposal_confidence.dtype).min
    )
    confidence = torch.where(mask_index, proposal_confidence, neg_inf)
    transfer = mask_index & (confidence >= threshold)
    force = torch.zeros_like(transfer).scatter_(
        1, torch.argmax(confidence, dim=1, keepdim=True), True
    )
    transfer_index = (transfer | force) & mask_index
    updated = torch.where(transfer_index, proposed, block_tokens)
    counts = transfer_index.sum(dim=1, dtype=torch.long)
    return updated, counts, proposal_confidence


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("vocab_size", [257, 50257])
def test_fused_matches_torch_path_random_logits(batch_size, vocab_size):
    torch.manual_seed(0)
    seq_len = 32
    threshold = 0.9
    logits = torch.randn(batch_size, seq_len, vocab_size)
    block_tokens = torch.full((batch_size, seq_len), MASK_ID, dtype=torch.long)
    # Leave some positions already unmasked with real token ids.
    block_tokens[:, :4] = torch.randint(0, vocab_size, (batch_size, 4))

    reqs = [
        _threshold_request(f"req-{i}", seq_len, threshold, block_tokens[i])
        for i in range(batch_size)
    ]
    updated, counts = Sampler.sample_batch(reqs, block_tokens, logits)

    ref_updated, ref_counts, _ = _reference_threshold(logits, block_tokens, threshold)
    assert torch.equal(updated, ref_updated)
    assert torch.equal(counts, ref_counts)


def test_fused_token_ids_exact_argmax():
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 512)
    token_ids, _ = fused_argmax_confidence(logits)
    assert torch.equal(token_ids, torch.argmax(logits, dim=-1))


def test_fused_token_ids_first_index_on_tie():
    logits = torch.zeros(1, 1, 16)
    logits[0, 0, 3] = 5.0
    logits[0, 0, 9] = 5.0  # tie: torch.argmax returns the first (index 3)
    token_ids, _ = fused_argmax_confidence(logits)
    assert token_ids[0, 0].item() == 3


def test_fused_confidence_formula():
    # Two equal-probability logits -> max softmax prob == 0.5.
    logits = torch.tensor([[[2.0, 2.0]]])
    _, confidence = fused_argmax_confidence(logits)
    assert confidence.dtype == torch.float64
    assert torch.allclose(confidence, torch.tensor([[0.5]], dtype=torch.float64))

    # Reference parity on random logits.
    torch.manual_seed(2)
    logits = torch.randn(3, 5, 1000)
    _, confidence = fused_argmax_confidence(logits)
    ref = F.softmax(logits.to(torch.float64), dim=-1).max(dim=-1).values
    assert torch.allclose(confidence, ref, atol=1e-9)


def test_fast_path_not_taken_when_mixed(monkeypatch):
    seq_len = 3
    block_tokens = torch.full((2, seq_len), MASK_ID, dtype=torch.long)
    block_tokens[:, 0] = torch.tensor([1, 2])
    logits = torch.randn(2, seq_len, 16)

    reqs = [
        _threshold_request("req-thr", seq_len, 0.9, block_tokens[0]),
        SamplingRequest(
            request_id="req-rand",
            request_seed=7,
            token_ids=block_tokens[1].unsqueeze(0),
            block_start_idx=0,
            block_end_index=seq_len,
            step_index=0,
            sampling_parameters=SamplingParameters(
                temperature=0.0, unmasking_strategy="random"
            ),
            mask_token_id=MASK_ID,
        ),
    ]

    called = False

    def _spy(logits_arg):
        nonlocal called
        called = True
        return kernel_module.fused_argmax_confidence(logits_arg)

    monkeypatch.setattr(sampler_module, "fused_argmax_confidence", _spy)

    Sampler.sample_batch(reqs, block_tokens, logits)
    assert not called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for triton")
@pytest.mark.parametrize("vocab_size", [257, 50257, 151936])
def test_kernel_matches_fallback_on_gpu(vocab_size):
    torch.manual_seed(3)
    logits = torch.randn(4, 16, vocab_size)
    gpu_tokens, gpu_conf = fused_argmax_confidence(logits.cuda())
    cpu_tokens, cpu_conf = kernel_module._torch_fallback(logits)
    assert torch.equal(gpu_tokens.cpu(), cpu_tokens)
    assert torch.allclose(gpu_conf.cpu(), cpu_conf, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for triton")
def test_kernel_first_max_across_tile_boundary():
    # Ties straddling the BLOCK_V=8192 boundary must resolve to the lower index.
    vocab_size = 16384
    logits = torch.zeros(1, 1, vocab_size)
    logits[0, 0, 5] = 9.0
    logits[0, 0, 9000] = 9.0
    token_ids, _ = fused_argmax_confidence(logits.cuda())
    assert token_ids[0, 0].item() == 5
