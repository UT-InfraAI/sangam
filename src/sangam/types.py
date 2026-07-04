"""Shared type definitions for sangam."""

from enum import Enum

import torch


class WorkerType(Enum):
    """Type of worker in the serving pipeline."""

    PREFILL = "prefill"
    DECODE = "decode"
    COLOCATED = "colocated"


class PrefillSchedulerPolicy(Enum):
    """Policy for assigning prefill work to workers."""

    ROUND_ROBIN = "round_robin"
    LEAST_OUTSTANDING_PREFILL_TOKENS = "least_outstanding_prefill_tokens"
    LEAST_OUTSTANDING_REQUESTS = "least_outstanding_requests"
    LEAST_REQUEST_LENGTH_SUM = "least_request_length_sum"
    BALANCED_LENGTH_CLUSTERING = "balanced_length_clustering"


class PrefillQueuePolicy(Enum):
    """Policy for ordering queued prefill work within a worker."""

    ARRIVAL_ORDER = "arrival_order"
    FEWEST_REMAINING_BLOCKS = "fewest_remaining_blocks"


class DecodeSchedulerPolicy(Enum):
    """Policy for assigning decode work to workers."""

    ROUND_ROBIN = "round_robin"
    MAX_FREE_MEMORY = "max_free_memory"
    TOPOLOGY_GUARDED_MEMORY = "topology_guarded_memory"
    BALANCED_LENGTH_CLUSTERING = "balanced_length_clustering"


KVCache = list[tuple[torch.Tensor, torch.Tensor]]
