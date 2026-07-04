"""Tests for token sampling helpers and Sampler strategy dispatch."""

import pytest
import torch

from sangam import sampler as sampler_module
from sangam.sampler import (
    Sampler,
    SamplingRequest,
    add_gumbel_noise,
    get_transfer_index,
    get_transfer_index_dynamic,
)
from sangam.sampling_parameters import SamplingParameters

MASK_ID = 126336


def _sampling_request(**kwargs) -> SamplingRequest:
    defaults = dict(
        request_id="req-1",
        request_seed=123,
        token_ids=torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long),
        block_start_idx=1,
        block_end_index=4,
        step_index=0,
        sampling_parameters=SamplingParameters(),
        mask_token_id=MASK_ID,
    )
    defaults.update(kwargs)
    return SamplingRequest(**defaults)


class TestAddGumbelNoise:
    def test_zero_temperature_is_identity(self):
        logits = torch.randn(2, 10)
        out = add_gumbel_noise(logits, temperature=0, seed=123)
        assert torch.equal(out, logits)

    def test_nonzero_temperature_preserves_shape(self):
        logits = torch.randn(2, 10)
        out = add_gumbel_noise(logits, temperature=1.0, seed=123)
        assert out.shape == logits.shape

    def test_nonzero_temperature_is_request_seeded_not_global_rng(self):
        logits = torch.randn(2, 10)
        torch.manual_seed(1)
        out_a = add_gumbel_noise(logits, temperature=1.0, seed=123)
        torch.manual_seed(999)
        out_b = add_gumbel_noise(logits, temperature=1.0, seed=123)
        assert torch.equal(out_a, out_b)


class TestGetTransferIndex:
    def _make_inputs(self, seq_len=8, vocab_size=16, batch_size=1):
        logits = torch.randn(batch_size, seq_len, vocab_size)
        x = torch.full((batch_size, seq_len), MASK_ID, dtype=torch.long)
        x[:, :2] = torch.randint(0, vocab_size, (batch_size, 2))
        mask_index = x == MASK_ID
        return logits, mask_index, x

    def test_topk_confidence_mode(self):
        logits, mask_index, x = self._make_inputs()
        _, transfer = get_transfer_index(
            logits=logits,
            temperature=0,
            temperature_seed=123,
            selection_mode="confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=torch.tensor([3]),
            confidence_threshold=None,
            disallow_token_id=None,
        )
        assert transfer.sum().item() == 3
        assert (transfer & ~mask_index).sum().item() == 0

    def test_topk_random_mode(self):
        logits, mask_index, x = self._make_inputs()
        _, transfer = get_transfer_index(
            logits=logits,
            temperature=0,
            temperature_seed=123,
            selection_mode="random",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=torch.tensor([2]),
            confidence_threshold=None,
            disallow_token_id=None,
        )
        assert transfer.sum().item() == 2

    def test_threshold_mode(self):
        logits = torch.randn(1, 6, 16)
        x = torch.full((1, 6), MASK_ID, dtype=torch.long)
        x[0, :2] = torch.tensor([1, 2])
        mask_index = x == MASK_ID

        _, transfer = get_transfer_index(
            logits=logits,
            temperature=0,
            temperature_seed=123,
            selection_mode="confidence",
            mask_index=mask_index,
            x=x,
            num_transfer_tokens=None,
            confidence_threshold=0.5,
            disallow_token_id=None,
        )
        assert (transfer & ~mask_index).sum().item() == 0

    def test_invalid_confidence_mode(self):
        logits, mask_index, x = self._make_inputs()
        with pytest.raises(NotImplementedError):
            get_transfer_index(
                logits=logits,
                temperature=0,
                temperature_seed=123,
                selection_mode="invalid",
                mask_index=mask_index,
                x=x,
                num_transfer_tokens=torch.tensor([1]),
                confidence_threshold=None,
                disallow_token_id=None,
            )


class TestGetTransferIndexDynamic:
    def test_transfers_at_least_one(self):
        logits = torch.randn(1, 8, 16)
        x = torch.full((1, 8), MASK_ID, dtype=torch.long)
        x[0, :2] = torch.tensor([1, 2])
        mask_index = x == MASK_ID

        _, transfer = get_transfer_index_dynamic(
            logits=logits,
            temperature=0,
            temperature_seed=123,
            selection_mode="confidence",
            mask_index=mask_index,
            x=x,
            dynamic_unmask_factor=1.0,
        )
        assert transfer.sum().item() >= 1
        assert (transfer & ~mask_index).sum().item() == 0


class TestSamplerStrategies:
    def test_sample_batch_matches_single_request(self):
        sampler = Sampler()
        reqs = [
            _sampling_request(
                request_id="req-random",
                request_seed=101,
                token_ids=torch.tensor(
                    [[0, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
                ),
                block_start_idx=1,
                block_end_index=4,
                step_index=3,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
            ),
            _sampling_request(
                request_id="req-quota",
                request_seed=102,
                token_ids=torch.tensor(
                    [[1, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
                ),
                block_start_idx=1,
                block_end_index=4,
                step_index=0,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="conf_quota",
                    fixed_unmask_quota=2,
                ),
            ),
            _sampling_request(
                request_id="req-threshold",
                request_seed=103,
                token_ids=torch.tensor(
                    [[2, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
                ),
                block_start_idx=1,
                block_end_index=4,
                step_index=1,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="conf_threshold",
                    confidence_threshold=0.95,
                ),
            ),
            _sampling_request(
                request_id="req-dynamic",
                request_seed=104,
                token_ids=torch.tensor(
                    [[3, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
                ),
                block_start_idx=1,
                block_end_index=4,
                step_index=2,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="conf_dynamic",
                    dynamic_unmask_factor=1.0,
                ),
            ),
        ]
        logits = torch.tensor(
            [
                [[0.3, 0.2, 0.5], [0.9, 0.1, 0.0], [0.2, 0.7, 0.1]],
                [[0.1, 0.9, 0.0], [0.2, 0.1, 0.8], [0.7, 0.1, 0.2]],
                [[1.0, 1.0, 1.0], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
                [[0.1, 3.0, 0.0], [0.0, 2.0, 0.1], [0.0, 1.5, 0.3]],
            ],
            dtype=torch.float32,
        )
        block_tokens = torch.cat(
            [
                req.token_ids[:, req.block_start_idx : req.block_end_index]
                for req in reqs
            ],
            dim=0,
        )

        updated_blocks, counts = sampler.sample_batch(reqs, block_tokens, logits)

        for idx, req in enumerate(reqs):
            single_out, single_count = sampler.sample_from_logits(
                req, logits[idx : idx + 1]
            )
            assert torch.equal(
                updated_blocks[idx],
                single_out[0, req.block_start_idx : req.block_end_index],
            )
            assert counts[idx].item() == single_count

    def test_random_unmasks_at_least_one(self):
        sampler = Sampler()
        token_ids = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long)
        logits = torch.randn(1, 3, 16)

        out, num_unmasked_tokens = sampler.sample_from_logits(
            _sampling_request(
                request_id="req-random",
                request_seed=201,
                token_ids=token_ids.clone(),
                block_start_idx=1,
                block_end_index=4,
                step_index=5,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
            ),
            logits,
        )
        assert (out[:, 1:4] != MASK_ID).sum().item() >= 1
        assert num_unmasked_tokens >= 1

    def test_random_unmasks_up_to_all_masked_tokens(self, monkeypatch):
        sampler = Sampler()
        token_ids = torch.tensor(
            [[1, MASK_ID, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
        )
        logits = torch.full((1, 4, 8), -5.0)
        logits[:, :, 2] = 9.0

        monkeypatch.setattr(
            torch,
            "randint",
            lambda low, high, size, device, generator=None: torch.full(
                size, high - 1, device=device
            ),
        )

        out, num_unmasked_tokens = sampler.sample_from_logits(
            _sampling_request(
                request_id="req-random-all",
                request_seed=202,
                token_ids=token_ids.clone(),
                block_start_idx=1,
                block_end_index=5,
                step_index=2,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
            ),
            logits,
        )
        assert (out[:, 1:5] != MASK_ID).sum().item() == 4
        assert num_unmasked_tokens == 4

    def test_conf_quota_respects_fixed_k(self):
        sampler = Sampler()
        token_ids = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long)
        logits = torch.randn(1, 3, 16)

        out, num_unmasked_tokens = sampler.sample_from_logits(
            _sampling_request(
                request_id="req-quota",
                request_seed=203,
                token_ids=token_ids.clone(),
                block_start_idx=1,
                block_end_index=4,
                step_index=0,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="conf_quota",
                    fixed_unmask_quota=2,
                ),
            ),
            logits,
        )
        assert (out[:, 1:4] != MASK_ID).sum().item() == 2
        assert num_unmasked_tokens == 2

    def test_conf_quota_zero_falls_back_to_one_with_warning(self, monkeypatch):
        sampler = Sampler()
        token_ids = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long)
        logits = torch.randn(1, 3, 16)
        warning_messages = []

        def _capture_warning(msg, *args):
            warning_messages.append(msg % args if args else msg)

        monkeypatch.setattr(sampler_module.logger, "warning", _capture_warning)

        out, num_unmasked_tokens = sampler.sample_from_logits(
            _sampling_request(
                request_id="req-quota-zero",
                request_seed=204,
                token_ids=token_ids.clone(),
                block_start_idx=1,
                block_end_index=4,
                step_index=0,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="conf_quota",
                    fixed_unmask_quota=0,
                ),
            ),
            logits,
        )
        assert (out[:, 1:4] != MASK_ID).sum().item() == 1
        assert num_unmasked_tokens == 1
        assert warning_messages
        assert (
            "No tokens selected for 1 row(s) under strategy=conf_quota"
            in warning_messages[0]
        )

    def test_conf_threshold_requires_threshold(self):
        sampler = Sampler()
        token_ids = torch.tensor([[1, MASK_ID, MASK_ID]], dtype=torch.long)
        logits = torch.randn(1, 2, 16)

        with pytest.raises(ValueError, match="confidence_threshold"):
            sampler.sample_from_logits(
                _sampling_request(
                    request_id="req-threshold",
                    request_seed=205,
                    token_ids=token_ids,
                    block_start_idx=1,
                    block_end_index=3,
                    step_index=0,
                    sampling_parameters=SamplingParameters(
                        temperature=0.0,
                        unmasking_strategy="conf_threshold",
                    ),
                ),
                logits,
            )

    def test_random_selects_arbitrary_masked_positions(self, monkeypatch):
        sampler = Sampler()
        token_ids = torch.tensor(
            [[1, MASK_ID, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
        )
        logits = torch.randn(1, 4, 8)
        monkeypatch.setattr(
            torch,
            "randint",
            lambda low, high, size, device, generator=None: torch.ones(
                size, device=device, dtype=torch.long
            ),
        )

        out, num_unmasked_tokens = sampler.sample_from_logits(
            _sampling_request(
                request_id="req-random-arbitrary",
                request_seed=206,
                token_ids=token_ids.clone(),
                block_start_idx=1,
                block_end_index=5,
                step_index=7,
                sampling_parameters=SamplingParameters(
                    temperature=0.0,
                    unmasking_strategy="random",
                ),
            ),
            logits,
        )
        masked_before = int((token_ids[:, 1:5] == MASK_ID).sum().item())
        masked_after = int((out[:, 1:5] == MASK_ID).sum().item())
        assert num_unmasked_tokens == 1
        assert 1 <= num_unmasked_tokens <= masked_before
        assert masked_before - masked_after == 1

    def test_random_is_request_scoped_and_order_independent(self):
        sampler = Sampler()
        token_ids = torch.tensor(
            [[1, MASK_ID, MASK_ID, MASK_ID, MASK_ID]], dtype=torch.long
        )
        logits = torch.randn(1, 4, 32)

        req_a = _sampling_request(
            request_id="req-a",
            request_seed=301,
            token_ids=token_ids.clone(),
            block_start_idx=1,
            block_end_index=5,
            step_index=3,
            sampling_parameters=SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
        )
        req_b = _sampling_request(
            request_id="req-b",
            request_seed=302,
            token_ids=token_ids.clone(),
            block_start_idx=1,
            block_end_index=5,
            step_index=3,
            sampling_parameters=SamplingParameters(
                temperature=0.0,
                unmasking_strategy="random",
            ),
        )

        _, count_a_first = sampler.sample_from_logits(req_a, logits)
        _, count_b_first = sampler.sample_from_logits(req_b, logits)

        torch.manual_seed(12345)
        _, count_b_second = sampler.sample_from_logits(req_b, logits)
        torch.manual_seed(67890)
        _, count_a_second = sampler.sample_from_logits(req_a, logits)

        assert count_a_first == count_a_second
        assert count_b_first == count_b_second

    def test_conf_threshold_temperature_ignores_global_manual_seed(self):
        sampler = Sampler()
        req = _sampling_request(
            request_id="req-threshold-temp",
            request_seed=401,
            token_ids=torch.tensor([[1, MASK_ID, MASK_ID]], dtype=torch.long),
            block_start_idx=1,
            block_end_index=3,
            sampling_parameters=SamplingParameters(
                temperature=1.0,
                unmasking_strategy="conf_threshold",
                confidence_threshold=0.0,
            ),
        )
        logits = torch.randn(1, 2, 16)

        torch.manual_seed(1)
        out_a, count_a = sampler.sample_from_logits(req, logits)
        torch.manual_seed(999)
        out_b, count_b = sampler.sample_from_logits(req, logits)

        assert torch.equal(out_a, out_b)
        assert count_a == count_b

    def test_temperature_sample_batch_matches_single_request(self):
        sampler = Sampler()
        reqs = [
            _sampling_request(
                request_id="req-a",
                request_seed=501,
                token_ids=torch.tensor([[1, MASK_ID, MASK_ID]], dtype=torch.long),
                block_start_idx=1,
                block_end_index=3,
                step_index=2,
                sampling_parameters=SamplingParameters(
                    temperature=1.0,
                    unmasking_strategy="conf_threshold",
                    confidence_threshold=0.1,
                ),
            ),
            _sampling_request(
                request_id="req-b",
                request_seed=502,
                token_ids=torch.tensor([[2, MASK_ID, MASK_ID]], dtype=torch.long),
                block_start_idx=1,
                block_end_index=3,
                step_index=2,
                sampling_parameters=SamplingParameters(
                    temperature=1.0,
                    unmasking_strategy="conf_threshold",
                    confidence_threshold=0.1,
                ),
            ),
        ]
        logits = torch.randn(2, 2, 8)
        block_tokens = torch.cat(
            [
                req.token_ids[:, req.block_start_idx : req.block_end_index]
                for req in reqs
            ],
            dim=0,
        )

        updated_blocks, counts = sampler.sample_batch(reqs, block_tokens, logits)

        for idx, req in enumerate(reqs):
            single_out, single_count = sampler.sample_from_logits(
                req, logits[idx : idx + 1]
            )
            assert torch.equal(
                updated_blocks[idx],
                single_out[0, req.block_start_idx : req.block_end_index],
            )
            assert counts[idx].item() == single_count

    def test_different_request_seed_changes_temperature_outcome(self):
        sampler = Sampler()
        logits = torch.zeros((1, 2, 64), dtype=torch.float32)
        req_a = _sampling_request(
            request_id="req-a",
            request_seed=601,
            token_ids=torch.tensor([[1, MASK_ID, MASK_ID]], dtype=torch.long),
            block_start_idx=1,
            block_end_index=3,
            sampling_parameters=SamplingParameters(
                temperature=1.0,
                unmasking_strategy="conf_threshold",
                confidence_threshold=0.0,
            ),
        )
        req_b = _sampling_request(
            request_id="req-b",
            request_seed=602,
            token_ids=torch.tensor([[1, MASK_ID, MASK_ID]], dtype=torch.long),
            block_start_idx=1,
            block_end_index=3,
            sampling_parameters=req_a.sampling_parameters,
        )

        out_a, _ = sampler.sample_from_logits(req_a, logits)
        out_b, _ = sampler.sample_from_logits(req_b, logits)

        assert not torch.equal(out_a, out_b)
