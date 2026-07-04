"""Shared worker-local prefill queue ordering helpers."""

from __future__ import annotations

from typing import Protocol

from sangam.types import PrefillQueuePolicy


class PrefillQueueRequest(Protocol):
    request_id: str
    arrival_time: float
    block_index: int
    total_generation_blocks: int


def compute_prefill_queue_priority(
    request: PrefillQueueRequest,
    queue_policy: PrefillQueuePolicy,
) -> tuple[int, float, str] | tuple[float, str]:
    if queue_policy is PrefillQueuePolicy.FEWEST_REMAINING_BLOCKS:
        remaining_generation_blocks = (
            request.total_generation_blocks - request.block_index - 1
        )
        return (
            remaining_generation_blocks,
            request.arrival_time,
            request.request_id,
        )
    return (request.arrival_time, request.request_id)
