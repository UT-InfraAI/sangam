import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from sangam.kernels import fused_argmax_confidence
from sangam.logger import init_logger
from sangam.sampling_parameters import SamplingParameters

logger = init_logger(__name__)


@dataclass
class SamplingRequest:
    request_id: str
    request_seed: int
    token_ids: torch.Tensor
    block_start_idx: int
    block_end_index: int
    step_index: int
    sampling_parameters: SamplingParameters
    mask_token_id: int


class Sampler:
    """Shared sampling logic for prefill and decode phases."""

    STRATEGY_RANDOM = "random"
    STRATEGY_CONF_THRESHOLD = "conf_threshold"
    STRATEGY_CONF_QUOTA = "conf_quota"
    STRATEGY_CONF_DYNAMIC = "conf_dynamic"
    _SELECTION_MODE_RANDOM = "random"
    _SELECTION_MODE_CONFIDENCE = "confidence"
    _STRATEGY_TO_CODE = {
        STRATEGY_RANDOM: 0,
        STRATEGY_CONF_THRESHOLD: 1,
        STRATEGY_CONF_QUOTA: 2,
        STRATEGY_CONF_DYNAMIC: 3,
    }

    @staticmethod
    @torch.inference_mode()
    def sample_from_logits(
        req: SamplingRequest,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Apply one sampling step from logits."""
        token_ids = req.token_ids
        block_tokens = token_ids[:, req.block_start_idx : req.block_end_index]
        updated_block_tokens, num_unmasked_tokens = Sampler.sample_batch(
            reqs=[req],
            block_tokens=block_tokens,
            logits=logits,
        )
        updated_token_ids = Sampler._apply_updates(
            req=req,
            token_ids=token_ids,
            block_tokens=block_tokens,
            updated_block_tokens=updated_block_tokens,
        )
        return updated_token_ids, int(num_unmasked_tokens[0].item())

    @staticmethod
    @torch.inference_mode()
    def sample_batch(
        reqs: Sequence[SamplingRequest],
        block_tokens: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one sampling step for a batch of blocks."""
        if len(reqs) == 0:
            empty_counts = torch.zeros(0, dtype=torch.long, device=block_tokens.device)
            return block_tokens, empty_counts

        if block_tokens.dim() != 2:
            raise ValueError("block_tokens must have shape (batch, block_length)")
        if logits.dim() != 3:
            raise ValueError("logits must have shape (batch, block_length, vocab)")
        if block_tokens.size(0) != len(reqs) or logits.size(0) != len(reqs):
            raise ValueError("reqs, block_tokens, and logits batch sizes must match")

        metadata = _build_batch_sampling_metadata(reqs, block_tokens.device)
        mask_index = block_tokens == metadata.mask_token_ids.unsqueeze(1)
        if not mask_index.any():
            empty_counts = torch.zeros(
                len(reqs), dtype=torch.long, device=block_tokens.device
            )
            return block_tokens, empty_counts

        all_temp_zero = bool(torch.all(metadata.temperatures == 0).item())
        all_threshold = bool(
            torch.all(
                metadata.strategy_codes
                == Sampler._STRATEGY_TO_CODE[Sampler.STRATEGY_CONF_THRESHOLD]
            ).item()
        )
        if all_temp_zero and all_threshold:
            return Sampler._sample_batch_fused_conf_threshold(
                block_tokens=block_tokens,
                logits=logits,
                mask_index=mask_index,
                confidence_thresholds=metadata.confidence_thresholds,
            )

        proposed_token_ids = _batched_proposed_token_ids(
            logits=logits,
            temperatures=metadata.temperatures,
            temperature_seeds=metadata.temperature_seeds,
            strategy_codes=metadata.strategy_codes,
            mask_token_ids=metadata.mask_token_ids,
        )

        proposal_confidence = _compute_proposal_confidence(
            logits=logits,
            proposed_token_ids=proposed_token_ids,
            selection_mode=Sampler._SELECTION_MODE_CONFIDENCE,
        )
        proposed_token_ids = torch.where(mask_index, proposed_token_ids, block_tokens)
        neg_inf = torch.full_like(
            proposal_confidence, torch.finfo(proposal_confidence.dtype).min
        )
        confidence = torch.where(mask_index, proposal_confidence, neg_inf)

        transfer_index = torch.zeros_like(mask_index)

        threshold_rows = (
            metadata.strategy_codes
            == Sampler._STRATEGY_TO_CODE[Sampler.STRATEGY_CONF_THRESHOLD]
        )
        if threshold_rows.any():
            threshold_conf = confidence[threshold_rows]
            threshold_mask = mask_index[threshold_rows]
            threshold_values = metadata.confidence_thresholds[threshold_rows].unsqueeze(
                1
            )
            threshold_transfer = threshold_mask & (threshold_conf >= threshold_values)
            max_conf_indices = torch.argmax(threshold_conf, dim=1, keepdim=True)
            force_mask = torch.zeros_like(threshold_transfer).scatter_(
                1, max_conf_indices, True
            )
            transfer_index[threshold_rows] = (
                threshold_transfer | force_mask
            ) & threshold_mask

        random_rows = (
            metadata.strategy_codes
            == Sampler._STRATEGY_TO_CODE[Sampler.STRATEGY_RANDOM]
        )
        quota_rows = (
            metadata.strategy_codes
            == Sampler._STRATEGY_TO_CODE[Sampler.STRATEGY_CONF_QUOTA]
        )
        dynamic_rows = (
            metadata.strategy_codes
            == Sampler._STRATEGY_TO_CODE[Sampler.STRATEGY_CONF_DYNAMIC]
        )
        topk_rows = random_rows | quota_rows | dynamic_rows
        if topk_rows.any():
            row_confidence = confidence[topk_rows]
            row_mask = mask_index[topk_rows]
            row_counts = row_mask.sum(dim=1)

            random_topk_rows = random_rows[topk_rows]
            if random_topk_rows.any():
                random_confidence, random_num_transfer_tokens = (
                    _build_exact_random_sampling_inputs(
                        reqs=[
                            req
                            for idx, req in enumerate(reqs)
                            if topk_rows[idx] and random_rows[idx]
                        ],
                        masked_counts=row_counts[random_topk_rows],
                        seq_len=block_tokens.shape[1],
                        device=block_tokens.device,
                        dtype=row_confidence.dtype,
                    )
                )
                row_confidence[random_topk_rows] = torch.where(
                    row_mask[random_topk_rows], random_confidence, neg_inf[0]
                )

            num_transfer_tokens = torch.zeros_like(row_counts)

            if random_topk_rows.any():
                num_transfer_tokens[random_topk_rows] = random_num_transfer_tokens

            quota_topk_rows = quota_rows[topk_rows]
            if quota_topk_rows.any():
                num_transfer_tokens[quota_topk_rows] = metadata.fixed_unmask_quotas[
                    topk_rows
                ][quota_topk_rows]

            dynamic_topk_rows = dynamic_rows[topk_rows]
            if dynamic_topk_rows.any():
                dynamic_counts = _compute_dynamic_selected_counts(
                    confidence=row_confidence[dynamic_topk_rows],
                    mask_index=row_mask[dynamic_topk_rows],
                    dynamic_unmask_factor=metadata.dynamic_unmask_factors[topk_rows][
                        dynamic_topk_rows
                    ],
                )
                num_transfer_tokens[dynamic_topk_rows] = dynamic_counts

            _, idx = torch.sort(row_confidence, dim=1, descending=True)
            batch_size, seq_len = row_confidence.shape
            cols = (
                torch.arange(seq_len, device=row_confidence.device)
                .unsqueeze(0)
                .expand(batch_size, seq_len)
            )
            k_expanded = num_transfer_tokens.unsqueeze(1).expand(batch_size, seq_len)
            select_sorted = cols < torch.clamp(k_expanded, min=0)

            transfer_int = torch.zeros(
                batch_size, seq_len, device=row_confidence.device, dtype=torch.int8
            )
            transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
            transfer_index[topk_rows] = transfer_int.bool() & row_mask

        transfer_index = Sampler._ensure_minimum_transfer_batch(
            reqs=reqs,
            mask_index=mask_index,
            proposal_confidence=proposal_confidence,
            transfer_index=transfer_index,
            strategy_codes=metadata.strategy_codes,
        )

        updated_block_tokens = torch.where(
            transfer_index, proposed_token_ids, block_tokens
        )
        num_unmasked_tokens = transfer_index.sum(dim=1, dtype=torch.long)
        return updated_block_tokens, num_unmasked_tokens

    @staticmethod
    def _sample_batch_fused_conf_threshold(
        block_tokens: torch.Tensor,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        confidence_thresholds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast path for temperature==0 + conf_threshold across all rows.

        Replaces the full float64 softmax + gather + argmax of the generic path with
        a single fused argmax/confidence pass. The force-highest-confidence step
        guarantees at least one transfer per row with a masked token, so
        ``_ensure_minimum_transfer_batch`` is unnecessary here.
        """
        proposed_token_ids, proposal_confidence = fused_argmax_confidence(logits)
        proposed_token_ids = torch.where(mask_index, proposed_token_ids, block_tokens)
        neg_inf = torch.full_like(
            proposal_confidence, torch.finfo(proposal_confidence.dtype).min
        )
        confidence = torch.where(mask_index, proposal_confidence, neg_inf)

        transfer = mask_index & (confidence >= confidence_thresholds.unsqueeze(1))
        force_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer).scatter_(1, force_indices, True)
        transfer_index = (transfer | force_mask) & mask_index

        updated_block_tokens = torch.where(
            transfer_index, proposed_token_ids, block_tokens
        )
        num_unmasked_tokens = transfer_index.sum(dim=1, dtype=torch.long)
        return updated_block_tokens, num_unmasked_tokens

    @staticmethod
    def _apply_updates(
        req: SamplingRequest,
        token_ids: torch.Tensor,
        block_tokens: torch.Tensor,
        updated_block_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                token_ids[:, : req.block_start_idx],
                updated_block_tokens,
                token_ids[:, req.block_end_index :],
            ],
            dim=1,
        )

    @staticmethod
    def _ensure_minimum_transfer(
        req: SamplingRequest,
        logits: torch.Tensor,
        proposed_token_ids: torch.Tensor,
        mask_index: torch.Tensor,
        transfer_index: torch.Tensor,
    ) -> torch.Tensor:
        masked_rows = mask_index.any(dim=1)
        empty_rows = masked_rows & (~transfer_index.any(dim=1))
        if not empty_rows.any():
            return transfer_index

        proposal_confidence = _compute_proposal_confidence(
            logits=logits,
            proposed_token_ids=proposed_token_ids,
            selection_mode=Sampler._SELECTION_MODE_CONFIDENCE,
        )
        neg_inf = torch.tensor(
            torch.finfo(proposal_confidence.dtype).min,
            device=proposal_confidence.device,
            dtype=proposal_confidence.dtype,
        )
        confidence = torch.where(mask_index, proposal_confidence, neg_inf)
        fallback_indices = torch.argmax(confidence, dim=1, keepdim=True)
        fallback_mask = torch.zeros_like(transfer_index).scatter_(
            1, fallback_indices, True
        )
        transfer_index = transfer_index | (fallback_mask & empty_rows.unsqueeze(1))

        logger.warning(
            "No tokens selected for %d row(s) under strategy=%s; "
            "forcing highest-confidence masked token",
            int(empty_rows.sum().item()),
            req.sampling_parameters.unmasking_strategy,
        )
        return transfer_index

    @staticmethod
    def _make_deterministic_generator(
        req: SamplingRequest,
        salt: str,
    ) -> torch.Generator:
        seed = _make_deterministic_seed(req, salt)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _ensure_minimum_transfer_batch(
        reqs: Sequence[SamplingRequest],
        mask_index: torch.Tensor,
        proposal_confidence: torch.Tensor,
        transfer_index: torch.Tensor,
        strategy_codes: torch.Tensor,
    ) -> torch.Tensor:
        masked_rows = mask_index.any(dim=1)
        empty_rows = masked_rows & (~transfer_index.any(dim=1))
        if not empty_rows.any():
            return transfer_index

        neg_inf = torch.full_like(
            proposal_confidence, torch.finfo(proposal_confidence.dtype).min
        )
        confidence = torch.where(mask_index, proposal_confidence, neg_inf)
        fallback_indices = torch.argmax(confidence, dim=1, keepdim=True)
        fallback_mask = torch.zeros_like(transfer_index).scatter_(
            1, fallback_indices, True
        )
        transfer_index = transfer_index | (fallback_mask & empty_rows.unsqueeze(1))

        for strategy_code in torch.unique(strategy_codes[empty_rows]).tolist():
            strategy = _strategy_name_from_code(strategy_code)
            count = int((empty_rows & (strategy_codes == strategy_code)).sum().item())
            logger.warning(
                "No tokens selected for %d row(s) under strategy=%s; "
                "forcing highest-confidence masked token",
                count,
                strategy,
            )
        return transfer_index


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    if temperature == 0:
        return logits
    if logits.dim() == 2:
        return _add_gumbel_noise_batched(
            logits=logits.unsqueeze(0),
            temperatures=torch.tensor([temperature], device=logits.device),
            seeds=torch.tensor([seed], device=logits.device, dtype=torch.long),
        )[0]
    return _add_gumbel_noise_batched(
        logits=logits,
        temperatures=torch.tensor([temperature], device=logits.device),
        seeds=torch.tensor([seed], device=logits.device, dtype=torch.long),
    )[0:1]


@dataclass
class _BatchSamplingMetadata:
    temperatures: torch.Tensor
    strategy_codes: torch.Tensor
    mask_token_ids: torch.Tensor
    confidence_thresholds: torch.Tensor
    fixed_unmask_quotas: torch.Tensor
    dynamic_unmask_factors: torch.Tensor
    temperature_seeds: torch.Tensor
    random_count_seeds: torch.Tensor
    random_selection_seeds: torch.Tensor


def _build_batch_sampling_metadata(
    reqs: Sequence[SamplingRequest],
    device: torch.device,
) -> _BatchSamplingMetadata:
    temperatures = []
    strategy_codes = []
    mask_token_ids = []
    confidence_thresholds = []
    fixed_unmask_quotas = []
    dynamic_unmask_factors = []
    temperature_seeds = []
    random_count_seeds = []
    random_selection_seeds = []

    for req in reqs:
        params = req.sampling_parameters
        temperatures.append(params.temperature)
        strategy_codes.append(Sampler._STRATEGY_TO_CODE[params.unmasking_strategy])
        mask_token_ids.append(req.mask_token_id)
        confidence_thresholds.append(
            0.0 if params.confidence_threshold is None else params.confidence_threshold
        )
        fixed_unmask_quotas.append(
            0 if params.fixed_unmask_quota is None else params.fixed_unmask_quota
        )
        dynamic_unmask_factors.append(
            0.0
            if params.dynamic_unmask_factor is None
            else params.dynamic_unmask_factor
        )
        temperature_seeds.append(_make_deterministic_seed(req, "temperature_gumbel"))
        random_count_seeds.append(_make_deterministic_seed(req, "random_unmask_count"))
        random_selection_seeds.append(
            _make_deterministic_seed(req, "random_unmask_selection")
        )

    return _BatchSamplingMetadata(
        temperatures=torch.tensor(temperatures, device=device, dtype=torch.float32),
        strategy_codes=torch.tensor(strategy_codes, device=device, dtype=torch.long),
        mask_token_ids=torch.tensor(mask_token_ids, device=device, dtype=torch.long),
        confidence_thresholds=torch.tensor(
            confidence_thresholds, device=device, dtype=torch.float32
        ),
        fixed_unmask_quotas=torch.tensor(
            fixed_unmask_quotas, device=device, dtype=torch.long
        ),
        dynamic_unmask_factors=torch.tensor(
            dynamic_unmask_factors, device=device, dtype=torch.float32
        ),
        temperature_seeds=torch.tensor(
            temperature_seeds, device=device, dtype=torch.long
        ),
        random_count_seeds=torch.tensor(
            random_count_seeds, device=device, dtype=torch.long
        ),
        random_selection_seeds=torch.tensor(
            random_selection_seeds, device=device, dtype=torch.long
        ),
    )


def _make_deterministic_seed(req: SamplingRequest, salt: str) -> int:
    return derive_sampling_seed(
        request_seed=req.request_seed,
        block_start_idx=req.block_start_idx,
        block_end_index=req.block_end_index,
        step_index=req.step_index,
        salt=salt,
    )


def _strategy_name_from_code(strategy_code: int) -> str:
    for name, code in Sampler._STRATEGY_TO_CODE.items():
        if code == strategy_code:
            return name
    raise KeyError(strategy_code)


def _batched_proposed_token_ids(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    temperature_seeds: torch.Tensor,
    strategy_codes: torch.Tensor,
    mask_token_ids: torch.Tensor,
) -> torch.Tensor:
    logits_with_noise = _add_gumbel_noise_batched(
        logits=logits,
        temperatures=temperatures,
        seeds=temperature_seeds,
    )
    random_rows = strategy_codes == Sampler._STRATEGY_TO_CODE[Sampler.STRATEGY_RANDOM]
    if random_rows.any():
        logits_with_noise = logits_with_noise.clone()
        for row_idx in torch.nonzero(random_rows, as_tuple=False).squeeze(1).tolist():
            disallow_token_id = int(mask_token_ids[row_idx].item())
            if 0 <= disallow_token_id < logits_with_noise.shape[-1]:
                logits_with_noise[row_idx, :, disallow_token_id] = torch.finfo(
                    logits_with_noise.dtype
                ).min
    return torch.argmax(logits_with_noise, dim=-1)


def _add_gumbel_noise_batched(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
) -> torch.Tensor:
    if torch.all(temperatures == 0):
        return logits

    logits_fp64 = logits.to(torch.float64)
    batch_size, seq_len, vocab_size = logits.shape
    noise = _deterministic_uniform_3d(
        seeds=seeds,
        seq_len=seq_len,
        width=vocab_size,
        device=logits.device,
        dtype=torch.float64,
    ).reshape(batch_size, seq_len, vocab_size)
    gumbel_noise = (-torch.log(noise)) ** temperatures.to(
        device=logits.device, dtype=torch.float64
    ).view(-1, 1, 1)
    noisy_logits = logits_fp64.exp() / gumbel_noise
    zero_temp_rows = temperatures == 0
    if zero_temp_rows.any():
        noisy_logits[zero_temp_rows] = logits_fp64[zero_temp_rows]
    return noisy_logits


def _deterministic_uniform_matrix(
    seeds: torch.Tensor,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if seeds.numel() == 0:
        return torch.empty((0, width), device=device, dtype=dtype)

    positions = torch.arange(width, device=device, dtype=torch.long).unsqueeze(0)
    x = seeds.unsqueeze(1) + positions * 6364136223846793005 + 1442695040888963407
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 33))
    x = x * 2862933555777941757 + 3037000493
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 29))
    mantissa = torch.bitwise_and(x, (1 << 53) - 1).to(torch.float64)
    uniform = mantissa / float(1 << 53)
    return uniform.to(dtype=dtype)


def _deterministic_uniform_3d(
    seeds: torch.Tensor,
    seq_len: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _deterministic_uniform_matrix(
        seeds=seeds,
        width=seq_len * width,
        device=device,
        dtype=dtype,
    )


def derive_sampling_seed(
    request_seed: int,
    block_start_idx: int,
    block_end_index: int,
    step_index: int,
    salt: str,
) -> int:
    seed_material = (
        f"{request_seed}|{block_start_idx}|{block_end_index}|{step_index}|{salt}"
    )
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return seed % (2**63 - 1)


def _build_exact_random_sampling_inputs(
    reqs: Sequence[SamplingRequest],
    masked_counts: torch.Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    confidence_rows = []
    num_transfer_tokens = []

    for req, masked_count in zip(reqs, masked_counts.tolist(), strict=True):
        if masked_count == 0:
            confidence_rows.append(torch.zeros((seq_len,), dtype=torch.float64))
            num_transfer_tokens.append(torch.tensor(0, dtype=torch.long))
            continue

        selection_generator = Sampler._make_deterministic_generator(
            req=req,
            salt="random_unmask_selection",
        )
        confidence_rows.append(
            torch.rand(
                (seq_len,),
                device="cpu",
                dtype=torch.float64,
                generator=selection_generator,
            )
        )

        count_generator = Sampler._make_deterministic_generator(
            req=req,
            salt="random_unmask_count",
        )
        sampled_count = torch.randint(
            low=1,
            high=masked_count + 1,
            size=(1,),
            device="cpu",
            generator=count_generator,
        )[0]
        num_transfer_tokens.append(sampled_count)

    confidence = torch.stack(confidence_rows, dim=0).to(device=device, dtype=dtype)
    counts = torch.stack(num_transfer_tokens).to(device=device, dtype=torch.long)
    return confidence, counts


def _compute_dynamic_selected_counts(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    dynamic_unmask_factor: torch.Tensor,
) -> torch.Tensor:
    if confidence.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=confidence.device)

    sorted_confidence, _ = torch.sort(confidence, dim=1, descending=True)
    masked_counts = mask_index.sum(dim=1)
    seq_len = confidence.shape[1]
    positions = torch.arange(seq_len, device=confidence.device).unsqueeze(0)
    thresholds = 1 - dynamic_unmask_factor.unsqueeze(1) / (positions + 2)
    thresholds[:, 0] = -1
    valid_positions = positions < masked_counts.unsqueeze(1)
    fail_mask = valid_positions & (sorted_confidence < thresholds)
    first_fail = torch.argmax(fail_mask.to(torch.long), dim=1)
    has_fail = fail_mask.any(dim=1)
    selected_count = torch.where(has_fail, first_fail, masked_counts)
    selected_count = torch.where(masked_counts > 0, selected_count.clamp(min=1), 0)
    return selected_count.to(torch.long)


def _compute_proposal_confidence(
    logits: torch.Tensor,
    proposed_token_ids: torch.Tensor,
    selection_mode: str,
    random_generator: torch.Generator | None = None,
) -> torch.Tensor:
    if selection_mode == Sampler._SELECTION_MODE_CONFIDENCE:
        p = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(
            p,
            dim=-1,
            index=proposed_token_ids.unsqueeze(-1),
        ).squeeze(-1)
    if selection_mode == Sampler._SELECTION_MODE_RANDOM:
        return torch.rand(
            proposed_token_ids.shape,
            device="cpu",
            dtype=torch.float64,
            generator=random_generator,
        ).to(proposed_token_ids.device)
    raise NotImplementedError(selection_mode)


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    temperature_seed: int,
    selection_mode: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    confidence_threshold: float | None,
    disallow_token_id: int | None,
    random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(
        logits,
        temperature=temperature,
        seed=temperature_seed,
    )
    if (
        disallow_token_id is not None
        and 0 <= disallow_token_id < logits_with_noise.shape[-1]
    ):
        logits_with_noise = logits_with_noise.clone()
        logits_with_noise[..., disallow_token_id] = torch.finfo(
            logits_with_noise.dtype
        ).min
    proposed_token_ids = torch.argmax(logits_with_noise, dim=-1)
    proposal_confidence = _compute_proposal_confidence(
        logits=logits,
        proposed_token_ids=proposed_token_ids,
        selection_mode=selection_mode,
        random_generator=random_generator,
    )

    proposed_token_ids = torch.where(mask_index, proposed_token_ids, x)
    neg_inf = torch.tensor(
        torch.finfo(proposal_confidence.dtype).min,
        device=proposal_confidence.device,
        dtype=proposal_confidence.dtype,
    )
    confidence = torch.where(mask_index, proposal_confidence, neg_inf)

    if confidence_threshold is not None:
        transfer_index = mask_index & (confidence >= confidence_threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(
            1, max_conf_indices, True
        )
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index
        return proposed_token_ids, transfer_index

    if num_transfer_tokens is None:
        raise ValueError(
            "num_transfer_tokens must be set when confidence_threshold is None"
        )

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device
    )
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    _, idx = torch.sort(confidence, dim=1, descending=True)
    batch_size, seq_len = confidence.shape
    cols = (
        torch.arange(seq_len, device=confidence.device)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
    )
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(batch_size, seq_len)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(
        batch_size, seq_len, device=confidence.device, dtype=torch.int8
    )
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return proposed_token_ids, transfer_index


def get_transfer_index_dynamic(
    logits: torch.Tensor,
    temperature: float,
    temperature_seed: int,
    selection_mode: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    dynamic_unmask_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(
        logits,
        temperature=temperature,
        seed=temperature_seed,
    )
    proposed_token_ids = torch.argmax(logits_with_noise, dim=-1)
    proposal_confidence = _compute_proposal_confidence(
        logits=logits,
        proposed_token_ids=proposed_token_ids,
        selection_mode=selection_mode,
    )

    proposed_token_ids = torch.where(mask_index, proposed_token_ids, x)
    neg_inf = torch.tensor(
        torch.finfo(proposal_confidence.dtype).min,
        device=proposal_confidence.device,
        dtype=proposal_confidence.dtype,
    )
    confidence = torch.where(mask_index, proposal_confidence, neg_inf)

    transfer_index = torch.zeros_like(
        proposed_token_ids,
        dtype=torch.bool,
        device=proposed_token_ids.device,
    )
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue

        ns = list(range(1, num_tokens + 1))
        es = [dynamic_unmask_factor / (n + 1) for n in ns]
        thresholds = [1 - e for e in es]

        thresholds[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]

        selected_count = len(thresholds)
        for top_i, threshold in enumerate(thresholds):
            if sorted_confidence[top_i] < threshold:
                selected_count = top_i
                break

        if selected_count == 0:
            selected_count = 1

        _, select_index = torch.topk(confidence[j], k=selected_count)
        transfer_index[j, select_index] = True

    return proposed_token_ids, transfer_index
