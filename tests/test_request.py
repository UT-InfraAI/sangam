"""Tests for Request and BlockState data models."""

import pytest

from sangam.request import BlockState, Request
from sangam.sampling_parameters import SamplingParameters

MASK_ID = 126336


def _make_request(**kwargs):
    defaults = dict(
        prompt_token_ids=[1],
        gen_length=4,
        block_length=4,
        sampling_parameters=SamplingParameters(),
        mask_id=MASK_ID,
        request_seed=123,
    )
    defaults.update(kwargs)
    return Request(**defaults)


class TestBlockState:
    def test_duration_properties_set(self):
        b = BlockState(block_index=0, block_start=0, block_end=32)
        b.prefill_start_time = 100.0
        b.prefill_end_time = 100.5
        b.kv_transfer_start_time = 100.5
        b.kv_transfer_end_time = 100.7
        b.decode_start_time = 100.7
        b.decode_end_time = 101.2
        assert b.prefill_duration == pytest.approx(0.5)
        assert b.kv_transfer_duration == pytest.approx(0.2)
        assert b.decode_duration == pytest.approx(0.5)


class TestRequestConstruction:
    def test_single_block(self):
        req = _make_request(prompt_token_ids=[1, 2, 3], gen_length=32, block_length=32)
        assert len(req.block_states) == 1
        assert req.block_states[0].block_start == 3
        assert req.block_states[0].block_end == 35

    def test_multi_block_decomposition(self):
        req = _make_request(prompt_token_ids=[1, 2], gen_length=64, block_length=32)
        assert len(req.block_states) == 2
        b0, b1 = req.block_states
        assert (b0.block_start, b0.block_end) == (2, 34)
        assert (b1.block_start, b1.block_end) == (34, 66)

    def test_sequence_ids_initialized_with_mask(self):
        req = _make_request(prompt_token_ids=[10, 20], gen_length=4, block_length=4)
        assert req.sequence_ids[:2] == [10, 20]
        # Remaining positions filled with mask_id
        assert all(t == req.mask_id for t in req.sequence_ids[2:])
        assert len(req.sequence_ids) == 6

    def test_gen_length_not_divisible_by_block_length_raises(self):
        with pytest.raises(ValueError, match="gen_length.*divisible.*block_length"):
            _make_request(prompt_token_ids=[1], gen_length=33, block_length=32)

    def test_unique_request_ids(self):
        r1 = _make_request(prompt_token_ids=[1], gen_length=4, block_length=4)
        r2 = _make_request(prompt_token_ids=[1], gen_length=4, block_length=4)
        assert r1.request_id != r2.request_id

    def test_target_blocks_defaults_to_full_buffer(self):
        req = _make_request(prompt_token_ids=[1], gen_length=128, block_length=32)
        assert len(req.block_states) == 4
        assert req.target_blocks == 4

    def test_target_blocks_smaller_than_buffer(self):
        req = _make_request(
            prompt_token_ids=[1],
            gen_length=128,
            block_length=32,
            target_blocks=2,
        )
        # Mask buffer covers the full gen_length so the model still sees 128
        # mask positions, but only 2 blocks will actually be decoded.
        assert len(req.sequence_ids) == 1 + 128
        assert len(req.block_states) == 4
        assert req.target_blocks == 2
        assert req.target_gen_tokens == 64
        # Accounting tracks the buffer footprint (prompt + gen_length), not
        # the early-exit target, since the worker holds the full buffer
        # until the request exits.
        assert req.request_accounting_tokens == 1 + 128

    def test_target_blocks_overflow_raises(self):
        with pytest.raises(ValueError, match="target_blocks.*exceeds"):
            _make_request(
                prompt_token_ids=[1],
                gen_length=64,
                block_length=32,
                target_blocks=5,
            )


class TestRequestProperties:
    def test_current_block(self):
        req = _make_request(prompt_token_ids=[1], gen_length=64, block_length=32)
        assert req.current_block is req.block_states[0]
        req.current_block_index = 1
        assert req.current_block is req.block_states[1]

    def test_current_block_past_end(self):
        req = _make_request(prompt_token_ids=[1], gen_length=32, block_length=32)
        req.current_block_index = 1
        assert req.current_block is None

    def test_prompt_length(self):
        req = _make_request(
            prompt_token_ids=[1, 2, 3, 4, 5], gen_length=4, block_length=4
        )
        assert req.prompt_length == 5

    def test_request_accounting_tokens(self):
        req = _make_request(
            prompt_token_ids=[1, 2, 3, 4, 5], gen_length=8, block_length=4
        )
        assert req.request_accounting_tokens == 13

    def test_e2e_time(self):
        req = _make_request(prompt_token_ids=[1], gen_length=4, block_length=4)
        assert req.e2e_time == 0.0  # not completed
        req.submit_time = 100.0
        req.complete_time = 102.5
        assert req.e2e_time == pytest.approx(2.5)

    def test_e2e_time_normalized(self):
        req = _make_request(prompt_token_ids=[1], gen_length=8, block_length=8)
        req.submit_time = 100.0
        req.complete_time = 104.0
        assert req.e2e_time_normalized == pytest.approx(0.5)

    def test_total_queue_wait_time(self):
        req = _make_request(prompt_token_ids=[1], gen_length=4, block_length=4)
        req.block_states[0].prefill_scheduler_wait_duration = 0.3
        req.block_states[0].decode_scheduler_wait_duration = 0.2
        req.block_states[0].scheduler_wait_duration = 0.5
        req.block_states[0].prefill_queue_wait_duration = 0.2
        req.block_states[0].decode_queue_wait_duration = 0.1
        assert req.total_prefill_queue_wait_time == pytest.approx(0.5)
        assert req.total_decode_queue_wait_time == pytest.approx(0.3)
        assert req.total_queue_wait_time == pytest.approx(0.8)

    def test_execution_time(self):
        req = _make_request(prompt_token_ids=[1], gen_length=4, block_length=4)
        req.submit_time = 100.0
        req.complete_time = 102.0
        req.block_states[0].prefill_scheduler_wait_duration = 0.2
        req.block_states[0].prefill_queue_wait_duration = 0.4
        req.block_states[0].decode_queue_wait_duration = 0.1
        assert req.execution_time == pytest.approx(1.3)
        assert req.execution_time_normalized == pytest.approx(0.325)

    def test_aggregate_timing(self):
        req = _make_request(prompt_token_ids=[1], gen_length=64, block_length=32)
        for i, b in enumerate(req.block_states):
            b.prefill_start_time = float(i * 10)
            b.prefill_end_time = float(i * 10 + 1)
            b.kv_transfer_start_time = float(i * 10 + 1)
            b.kv_transfer_end_time = float(i * 10 + 1.5)
            b.decode_start_time = float(i * 10 + 2)
            b.decode_end_time = float(i * 10 + 5)
        assert req.total_prefill_time == pytest.approx(2.0)
        assert req.total_kv_transfer_time == pytest.approx(1.0)
        assert req.total_decode_time == pytest.approx(6.0)

    def test_total_kv_transfer_time_nonoverlapped(self):
        req = _make_request(prompt_token_ids=[1], gen_length=64, block_length=32)
        first, second = req.block_states
        first.prefill_end_time = 1.0
        first.kv_transfer_start_time = 1.0
        first.kv_transfer_end_time = 1.4
        second.prefill_end_time = 3.0
        second.kv_transfer_start_time = 2.6
        second.kv_transfer_end_time = 3.5
        assert req.total_kv_transfer_time == pytest.approx(1.3)
        assert req.total_kv_transfer_time_nonoverlapped == pytest.approx(0.9)

    def test_unaccounted_time(self):
        req = _make_request(prompt_token_ids=[1], gen_length=4, block_length=4)
        req.submit_time = 10.0
        req.complete_time = 12.5
        block = req.block_states[0]
        block.prefill_start_time = 10.2
        block.prefill_end_time = 10.7
        block.kv_transfer_start_time = 10.6
        block.kv_transfer_end_time = 10.9
        block.decode_start_time = 11.1
        block.decode_end_time = 11.9
        block.prefill_scheduler_wait_duration = 0.1
        block.prefill_queue_wait_duration = 0.1
        block.decode_scheduler_wait_duration = 0.2
        block.decode_queue_wait_duration = 0.2
        assert req.verification_component_sum == pytest.approx(2.1)
        assert req.unaccounted_time == pytest.approx(0.4)
