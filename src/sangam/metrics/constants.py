"""Metric name enums for sangam."""

import enum


class RequestMetricsTimeDistribution(enum.Enum):
    REQUEST_E2E_TIME = "request_e2e_time"
    REQUEST_E2E_TIME_NORMALIZED = "request_e2e_time_normalized"
    REQUEST_E2E_TIME_EXCL_FIRST_BLOCK_QUEUE = "request_e2e_time_excl_first_block_queue"
    REQUEST_E2E_TIME_EXCL_FIRST_BLOCK_QUEUE_NORMALIZED = (
        "request_e2e_time_excl_first_block_queue_normalized"
    )
    REQUEST_FIRST_BLOCK_QUEUE = "request_first_block_queue"
    REQUEST_SCHEDULING_DELAY = "request_scheduling_delay"
    REQUEST_PREFILL_SCHEDULING_DELAY = "request_scheduling_delay_prefill"
    REQUEST_DECODE_SCHEDULING_DELAY = "request_scheduling_delay_decode"
    REQUEST_EXECUTION_TIME = "request_execution_time"
    REQUEST_EXECUTION_TIME_NORMALIZED = "request_execution_time_normalized"
    REQUEST_PREFILL_TIME = "request_prefill_time"
    REQUEST_KV_TRANSFER_TIME = "request_kv_transfer_time"
    REQUEST_KV_TRANSFER_TIME_NONOVERLAPPED = "request_kv_transfer_time_nonoverlapped"
    REQUEST_DECODE_TIME = "request_decode_time"
    REQUEST_DECODE_TIME_NORMALIZED = "request_decode_time_normalized"
    REQUEST_UNACCOUNTED_TIME = "request_unaccounted_time"


class RequestMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_PROMPT_TOKENS = "request_num_prompt_tokens"
    REQUEST_NUM_GEN_TOKENS = "request_num_gen_tokens"
    REQUEST_NUM_BLOCKS = "request_num_blocks"
    REQUEST_NUM_FORWARD_PASSES = "request_num_forward_passes"


class RequestMetricsCDFSketch(enum.Enum):
    REQUEST_TIME_BETWEEN_TOKENS = "request_time_between_tokens"
    REQUEST_TOKENS_UNMASKED_PER_FORWARD_PASS = (
        "request_tokens_unmasked_per_forward_pass"
    )


class BlockMetricsTimeDistribution(enum.Enum):
    BLOCK_PREFILL_TIME = "block_prefill_time"
    BLOCK_KV_TRANSFER_TIME_NONOVERLAPPED = "block_kv_transfer_time_nonoverlapped"
    BLOCK_DECODE_TIME = "block_decode_time"
    BLOCK_TOTAL_TIME = "block_total_time"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_prompt_len = "batch_prompt_len"
    BATCH_gen_len = "batch_gen_len"
    BATCH_NUM_UNMASKED_TOKENS = "batch_num_unmasked_tokens"
    BATCH_SIZE = "batch_size"
    BATCH_DECODE_LENGTH_STD = "batch_decode_length_std"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"
    BATCH_SAMPLING_TIME = "batch_sampling_time"
    BATCH_TOKEN_THROUGHPUT = "batch_token_throughput"
    INTER_BATCH_DELAY = "inter_batch_delay"
    BATCH_EXECUTION_TIME_PREFILL = "batch_execution_time_prefill"
    BATCH_EXECUTION_TIME_DECODE = "batch_execution_time_decode"
    BATCH_EXECUTION_TIME_MIXED = "batch_execution_time_mixed"


class WorkerSystemMetricsDistribution(enum.Enum):
    QUEUE_DEPTH_WAITING = "queue_depth_waiting"
    QUEUE_DEPTH_ACTIVE = "queue_depth_active"
    OUTSTANDING_PREFILL_TOKENS = "outstanding_prefill_tokens"
    KV_PAGE_UTILIZATION_RATIO = "kv_page_utilization_ratio"


class WorkerStateTimeline(enum.Enum):
    IDLE = "idle"
    QUEUED = "queued"
    BUSY = "busy"


class CompletionMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"


class WorkerBatchTimeSeries(enum.Enum):
    DECODE_LENGTH_SUM = "decode_length_sum"
    DEFICIT_TOKENS = "deficit_tokens"


class SchedulerQueueTimeSeries(enum.Enum):
    PENDING_REQUESTS = "scheduler_pending_requests"
    DECODE_READY_REQUESTS = "scheduler_decode_ready_requests"
