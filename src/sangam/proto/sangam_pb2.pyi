from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BatchPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BATCH_PHASE_PREFILL: _ClassVar[BatchPhase]
    BATCH_PHASE_DECODE: _ClassVar[BatchPhase]
    BATCH_PHASE_MIXED: _ClassVar[BatchPhase]

class WorkerState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_STATE_UNSPECIFIED: _ClassVar[WorkerState]
    WORKER_STATE_IDLE: _ClassVar[WorkerState]
    WORKER_STATE_QUEUED: _ClassVar[WorkerState]
    WORKER_STATE_BUSY: _ClassVar[WorkerState]
BATCH_PHASE_PREFILL: BatchPhase
BATCH_PHASE_DECODE: BatchPhase
BATCH_PHASE_MIXED: BatchPhase
WORKER_STATE_UNSPECIFIED: WorkerState
WORKER_STATE_IDLE: WorkerState
WORKER_STATE_QUEUED: WorkerState
WORKER_STATE_BUSY: WorkerState

class SamplingParameters(_message.Message):
    __slots__ = ("temperature", "unmasking_strategy", "confidence_threshold", "dynamic_unmask_factor", "fixed_unmask_quota")
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    UNMASKING_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_UNMASK_FACTOR_FIELD_NUMBER: _ClassVar[int]
    FIXED_UNMASK_QUOTA_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    unmasking_strategy: str
    confidence_threshold: float
    dynamic_unmask_factor: float
    fixed_unmask_quota: int
    def __init__(self, temperature: _Optional[float] = ..., unmasking_strategy: _Optional[str] = ..., confidence_threshold: _Optional[float] = ..., dynamic_unmask_factor: _Optional[float] = ..., fixed_unmask_quota: _Optional[int] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("prompt_token_ids", "gen_length", "request_seed", "sampling_parameters")
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    GEN_LENGTH_FIELD_NUMBER: _ClassVar[int]
    REQUEST_SEED_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    gen_length: int
    request_seed: int
    sampling_parameters: SamplingParameters
    def __init__(self, prompt_token_ids: _Optional[_Iterable[int]] = ..., gen_length: _Optional[int] = ..., request_seed: _Optional[int] = ..., sampling_parameters: _Optional[_Union[SamplingParameters, _Mapping]] = ...) -> None: ...

class SubmitResponse(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class PollRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class PollResponse(_message.Message):
    __slots__ = ("status", "output_token_ids", "num_forward_evals", "error_message", "server_arrival_time", "server_start_time", "server_complete_time")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    NUM_FORWARD_EVALS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SERVER_ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    SERVER_START_TIME_FIELD_NUMBER: _ClassVar[int]
    SERVER_COMPLETE_TIME_FIELD_NUMBER: _ClassVar[int]
    status: str
    output_token_ids: _containers.RepeatedScalarFieldContainer[int]
    num_forward_evals: int
    error_message: str
    server_arrival_time: float
    server_start_time: float
    server_complete_time: float
    def __init__(self, status: _Optional[str] = ..., output_token_ids: _Optional[_Iterable[int]] = ..., num_forward_evals: _Optional[int] = ..., error_message: _Optional[str] = ..., server_arrival_time: _Optional[float] = ..., server_start_time: _Optional[float] = ..., server_complete_time: _Optional[float] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("request_id", "status", "output_token_ids", "num_forward_evals", "error_message", "server_arrival_time", "server_start_time", "server_complete_time")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    NUM_FORWARD_EVALS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SERVER_ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    SERVER_START_TIME_FIELD_NUMBER: _ClassVar[int]
    SERVER_COMPLETE_TIME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    status: str
    output_token_ids: _containers.RepeatedScalarFieldContainer[int]
    num_forward_evals: int
    error_message: str
    server_arrival_time: float
    server_start_time: float
    server_complete_time: float
    def __init__(self, request_id: _Optional[str] = ..., status: _Optional[str] = ..., output_token_ids: _Optional[_Iterable[int]] = ..., num_forward_evals: _Optional[int] = ..., error_message: _Optional[str] = ..., server_arrival_time: _Optional[float] = ..., server_start_time: _Optional[float] = ..., server_complete_time: _Optional[float] = ...) -> None: ...

class RegisterWorkerRequest(_message.Message):
    __slots__ = ("worker_id", "worker_type", "address", "dist_rank", "max_pages", "page_size", "is_conversion", "gpu_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_TYPE_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DIST_RANK_FIELD_NUMBER: _ClassVar[int]
    MAX_PAGES_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    IS_CONVERSION_FIELD_NUMBER: _ClassVar[int]
    GPU_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    worker_type: str
    address: str
    dist_rank: int
    max_pages: int
    page_size: int
    is_conversion: bool
    gpu_id: int
    def __init__(self, worker_id: _Optional[str] = ..., worker_type: _Optional[str] = ..., address: _Optional[str] = ..., dist_rank: _Optional[int] = ..., max_pages: _Optional[int] = ..., page_size: _Optional[int] = ..., is_conversion: bool = ..., gpu_id: _Optional[int] = ...) -> None: ...

class RegisterWorkerResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class BatchRequestUpdate(_message.Message):
    __slots__ = ("request_id", "block_index", "success", "updated_sequence", "num_unmasked_tokens", "num_forward_evals_in_batch_phase", "prefill_duration", "prefill_queue_wait_duration", "kv_transfer_duration", "decode_duration", "decode_queue_wait_duration", "num_layers", "num_kv_heads", "head_dim", "seq_length", "block_completed", "request_phase")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    NUM_UNMASKED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_FORWARD_EVALS_IN_BATCH_PHASE_FIELD_NUMBER: _ClassVar[int]
    PREFILL_DURATION_FIELD_NUMBER: _ClassVar[int]
    PREFILL_QUEUE_WAIT_DURATION_FIELD_NUMBER: _ClassVar[int]
    KV_TRANSFER_DURATION_FIELD_NUMBER: _ClassVar[int]
    DECODE_DURATION_FIELD_NUMBER: _ClassVar[int]
    DECODE_QUEUE_WAIT_DURATION_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NUM_KV_HEADS_FIELD_NUMBER: _ClassVar[int]
    HEAD_DIM_FIELD_NUMBER: _ClassVar[int]
    SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    BLOCK_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    REQUEST_PHASE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    block_index: int
    success: bool
    updated_sequence: _containers.RepeatedScalarFieldContainer[int]
    num_unmasked_tokens: int
    num_forward_evals_in_batch_phase: int
    prefill_duration: float
    prefill_queue_wait_duration: float
    kv_transfer_duration: float
    decode_duration: float
    decode_queue_wait_duration: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_length: int
    block_completed: bool
    request_phase: BatchPhase
    def __init__(self, request_id: _Optional[str] = ..., block_index: _Optional[int] = ..., success: bool = ..., updated_sequence: _Optional[_Iterable[int]] = ..., num_unmasked_tokens: _Optional[int] = ..., num_forward_evals_in_batch_phase: _Optional[int] = ..., prefill_duration: _Optional[float] = ..., prefill_queue_wait_duration: _Optional[float] = ..., kv_transfer_duration: _Optional[float] = ..., decode_duration: _Optional[float] = ..., decode_queue_wait_duration: _Optional[float] = ..., num_layers: _Optional[int] = ..., num_kv_heads: _Optional[int] = ..., head_dim: _Optional[int] = ..., seq_length: _Optional[int] = ..., block_completed: bool = ..., request_phase: _Optional[_Union[BatchPhase, str]] = ...) -> None: ...

class BatchMetricsReport(_message.Message):
    __slots__ = ("worker_id", "worker_type", "batch_size", "prompt_len", "gen_len", "batch_start_time", "batch_end_time", "kv_total_pages", "kv_used_pages", "kv_free_pages", "num_unmasked_tokens", "batch_phase", "request_updates", "worker_state_after", "sampling_duration", "batch_op_attn_time", "batch_op_mlp_time", "batch_op_qkv_time")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_TYPE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_LEN_FIELD_NUMBER: _ClassVar[int]
    GEN_LEN_FIELD_NUMBER: _ClassVar[int]
    BATCH_START_TIME_FIELD_NUMBER: _ClassVar[int]
    BATCH_END_TIME_FIELD_NUMBER: _ClassVar[int]
    KV_TOTAL_PAGES_FIELD_NUMBER: _ClassVar[int]
    KV_USED_PAGES_FIELD_NUMBER: _ClassVar[int]
    KV_FREE_PAGES_FIELD_NUMBER: _ClassVar[int]
    NUM_UNMASKED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    BATCH_PHASE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_UPDATES_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATE_AFTER_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_DURATION_FIELD_NUMBER: _ClassVar[int]
    BATCH_OP_ATTN_TIME_FIELD_NUMBER: _ClassVar[int]
    BATCH_OP_MLP_TIME_FIELD_NUMBER: _ClassVar[int]
    BATCH_OP_QKV_TIME_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    worker_type: str
    batch_size: int
    prompt_len: int
    gen_len: int
    batch_start_time: float
    batch_end_time: float
    kv_total_pages: int
    kv_used_pages: int
    kv_free_pages: int
    num_unmasked_tokens: int
    batch_phase: BatchPhase
    request_updates: _containers.RepeatedCompositeFieldContainer[BatchRequestUpdate]
    worker_state_after: WorkerStateSnapshot
    sampling_duration: float
    batch_op_attn_time: float
    batch_op_mlp_time: float
    batch_op_qkv_time: float
    def __init__(self, worker_id: _Optional[str] = ..., worker_type: _Optional[str] = ..., batch_size: _Optional[int] = ..., prompt_len: _Optional[int] = ..., gen_len: _Optional[int] = ..., batch_start_time: _Optional[float] = ..., batch_end_time: _Optional[float] = ..., kv_total_pages: _Optional[int] = ..., kv_used_pages: _Optional[int] = ..., kv_free_pages: _Optional[int] = ..., num_unmasked_tokens: _Optional[int] = ..., batch_phase: _Optional[_Union[BatchPhase, str]] = ..., request_updates: _Optional[_Iterable[_Union[BatchRequestUpdate, _Mapping]]] = ..., worker_state_after: _Optional[_Union[WorkerStateSnapshot, _Mapping]] = ..., sampling_duration: _Optional[float] = ..., batch_op_attn_time: _Optional[float] = ..., batch_op_mlp_time: _Optional[float] = ..., batch_op_qkv_time: _Optional[float] = ...) -> None: ...

class BatchMetricsAck(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class KVTransferReport(_message.Message):
    __slots__ = ("worker_id", "request_id", "block_index", "success", "transfer_start_time", "transfer_end_time", "error_message", "worker_state_after")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_START_TIME_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_END_TIME_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATE_AFTER_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    request_id: str
    block_index: int
    success: bool
    transfer_start_time: float
    transfer_end_time: float
    error_message: str
    worker_state_after: WorkerStateSnapshot
    def __init__(self, worker_id: _Optional[str] = ..., request_id: _Optional[str] = ..., block_index: _Optional[int] = ..., success: bool = ..., transfer_start_time: _Optional[float] = ..., transfer_end_time: _Optional[float] = ..., error_message: _Optional[str] = ..., worker_state_after: _Optional[_Union[WorkerStateSnapshot, _Mapping]] = ...) -> None: ...

class KVTransferAck(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class WorkerStateSnapshot(_message.Message):
    __slots__ = ("state", "timestamp", "waiting_queue_depth", "active_batch_size", "kv_total_pages", "kv_used_pages", "kv_free_pages", "deficit_tokens")
    STATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    WAITING_QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    KV_TOTAL_PAGES_FIELD_NUMBER: _ClassVar[int]
    KV_USED_PAGES_FIELD_NUMBER: _ClassVar[int]
    KV_FREE_PAGES_FIELD_NUMBER: _ClassVar[int]
    DEFICIT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    state: WorkerState
    timestamp: float
    waiting_queue_depth: int
    active_batch_size: int
    kv_total_pages: int
    kv_used_pages: int
    kv_free_pages: int
    deficit_tokens: int
    def __init__(self, state: _Optional[_Union[WorkerState, str]] = ..., timestamp: _Optional[float] = ..., waiting_queue_depth: _Optional[int] = ..., active_batch_size: _Optional[int] = ..., kv_total_pages: _Optional[int] = ..., kv_used_pages: _Optional[int] = ..., kv_free_pages: _Optional[int] = ..., deficit_tokens: _Optional[int] = ...) -> None: ...

class ExportMetricsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ExportMetricsResponse(_message.Message):
    __slots__ = ("success", "output_dir")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    output_dir: str
    def __init__(self, success: bool = ..., output_dir: _Optional[str] = ...) -> None: ...

class ResetMetricsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetMetricsResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class GetSchedulerStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetSchedulerStatusResponse(_message.Message):
    __slots__ = ("num_prefill_workers", "num_decode_workers", "num_colocated_workers", "ready_for_requests")
    NUM_PREFILL_WORKERS_FIELD_NUMBER: _ClassVar[int]
    NUM_DECODE_WORKERS_FIELD_NUMBER: _ClassVar[int]
    NUM_COLOCATED_WORKERS_FIELD_NUMBER: _ClassVar[int]
    READY_FOR_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    num_prefill_workers: int
    num_decode_workers: int
    num_colocated_workers: int
    ready_for_requests: bool
    def __init__(self, num_prefill_workers: _Optional[int] = ..., num_decode_workers: _Optional[int] = ..., num_colocated_workers: _Optional[int] = ..., ready_for_requests: bool = ...) -> None: ...

class StreamingDecodeTarget(_message.Message):
    __slots__ = ("worker_id", "dst_rank", "worker_address")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    DST_RANK_FIELD_NUMBER: _ClassVar[int]
    WORKER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    dst_rank: int
    worker_address: str
    def __init__(self, worker_id: _Optional[str] = ..., dst_rank: _Optional[int] = ..., worker_address: _Optional[str] = ...) -> None: ...

class EnqueuePrefillRequest(_message.Message):
    __slots__ = ("request_id", "sequence_ids", "block_start", "block_end", "block_index", "total_generation_blocks", "request_seed", "sampling_parameters", "streaming_decode_target", "mask_id", "arrival_time", "prefill_enqueue_time")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_START_FIELD_NUMBER: _ClassVar[int]
    BLOCK_END_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_GENERATION_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_SEED_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    STREAMING_DECODE_TARGET_FIELD_NUMBER: _ClassVar[int]
    MASK_ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    PREFILL_ENQUEUE_TIME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    sequence_ids: _containers.RepeatedScalarFieldContainer[int]
    block_start: int
    block_end: int
    block_index: int
    total_generation_blocks: int
    request_seed: int
    sampling_parameters: SamplingParameters
    streaming_decode_target: StreamingDecodeTarget
    mask_id: int
    arrival_time: float
    prefill_enqueue_time: float
    def __init__(self, request_id: _Optional[str] = ..., sequence_ids: _Optional[_Iterable[int]] = ..., block_start: _Optional[int] = ..., block_end: _Optional[int] = ..., block_index: _Optional[int] = ..., total_generation_blocks: _Optional[int] = ..., request_seed: _Optional[int] = ..., sampling_parameters: _Optional[_Union[SamplingParameters, _Mapping]] = ..., streaming_decode_target: _Optional[_Union[StreamingDecodeTarget, _Mapping]] = ..., mask_id: _Optional[int] = ..., arrival_time: _Optional[float] = ..., prefill_enqueue_time: _Optional[float] = ...) -> None: ...

class EnqueuePrefillResponse(_message.Message):
    __slots__ = ("success", "accepted_state")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_STATE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    accepted_state: WorkerStateSnapshot
    def __init__(self, success: bool = ..., accepted_state: _Optional[_Union[WorkerStateSnapshot, _Mapping]] = ...) -> None: ...

class TriggerKVTransferRequest(_message.Message):
    __slots__ = ("request_id", "block_index", "decode_worker_id", "decode_dst_rank", "decode_worker_address", "decode_request")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    DECODE_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    DECODE_DST_RANK_FIELD_NUMBER: _ClassVar[int]
    DECODE_WORKER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DECODE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    block_index: int
    decode_worker_id: str
    decode_dst_rank: int
    decode_worker_address: str
    decode_request: EnqueueDecodeRequest
    def __init__(self, request_id: _Optional[str] = ..., block_index: _Optional[int] = ..., decode_worker_id: _Optional[str] = ..., decode_dst_rank: _Optional[int] = ..., decode_worker_address: _Optional[str] = ..., decode_request: _Optional[_Union[EnqueueDecodeRequest, _Mapping]] = ...) -> None: ...

class TriggerKVTransferResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class ReturnQueuedPrefillsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReturnQueuedPrefillsResponse(_message.Message):
    __slots__ = ("requests",)
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[EnqueuePrefillRequest]
    def __init__(self, requests: _Optional[_Iterable[_Union[EnqueuePrefillRequest, _Mapping]]] = ...) -> None: ...

class ReceiveKVCacheRequest(_message.Message):
    __slots__ = ("request_id", "src_rank", "num_layers", "num_kv_heads", "head_dim", "seq_length", "block_start", "block_end", "block_index", "sequence_ids", "request_seed", "sampling_parameters", "streaming", "auto_enqueue_decode", "mask_id", "arrival_time", "decode_enqueue_time")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SRC_RANK_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NUM_KV_HEADS_FIELD_NUMBER: _ClassVar[int]
    HEAD_DIM_FIELD_NUMBER: _ClassVar[int]
    SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    BLOCK_START_FIELD_NUMBER: _ClassVar[int]
    BLOCK_END_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_IDS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_SEED_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    STREAMING_FIELD_NUMBER: _ClassVar[int]
    AUTO_ENQUEUE_DECODE_FIELD_NUMBER: _ClassVar[int]
    MASK_ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    DECODE_ENQUEUE_TIME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    src_rank: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_length: int
    block_start: int
    block_end: int
    block_index: int
    sequence_ids: _containers.RepeatedScalarFieldContainer[int]
    request_seed: int
    sampling_parameters: SamplingParameters
    streaming: bool
    auto_enqueue_decode: bool
    mask_id: int
    arrival_time: float
    decode_enqueue_time: float
    def __init__(self, request_id: _Optional[str] = ..., src_rank: _Optional[int] = ..., num_layers: _Optional[int] = ..., num_kv_heads: _Optional[int] = ..., head_dim: _Optional[int] = ..., seq_length: _Optional[int] = ..., block_start: _Optional[int] = ..., block_end: _Optional[int] = ..., block_index: _Optional[int] = ..., sequence_ids: _Optional[_Iterable[int]] = ..., request_seed: _Optional[int] = ..., sampling_parameters: _Optional[_Union[SamplingParameters, _Mapping]] = ..., streaming: bool = ..., auto_enqueue_decode: bool = ..., mask_id: _Optional[int] = ..., arrival_time: _Optional[float] = ..., decode_enqueue_time: _Optional[float] = ...) -> None: ...

class ReceiveKVCacheResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class EnqueueDecodeRequest(_message.Message):
    __slots__ = ("request_id", "sequence_ids", "block_start", "block_end", "block_index", "request_seed", "sampling_parameters", "mask_id", "arrival_time", "decode_enqueue_time")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_START_FIELD_NUMBER: _ClassVar[int]
    BLOCK_END_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    REQUEST_SEED_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    MASK_ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    DECODE_ENQUEUE_TIME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    sequence_ids: _containers.RepeatedScalarFieldContainer[int]
    block_start: int
    block_end: int
    block_index: int
    request_seed: int
    sampling_parameters: SamplingParameters
    mask_id: int
    arrival_time: float
    decode_enqueue_time: float
    def __init__(self, request_id: _Optional[str] = ..., sequence_ids: _Optional[_Iterable[int]] = ..., block_start: _Optional[int] = ..., block_end: _Optional[int] = ..., block_index: _Optional[int] = ..., request_seed: _Optional[int] = ..., sampling_parameters: _Optional[_Union[SamplingParameters, _Mapping]] = ..., mask_id: _Optional[int] = ..., arrival_time: _Optional[float] = ..., decode_enqueue_time: _Optional[float] = ...) -> None: ...

class EnqueueDecodeResponse(_message.Message):
    __slots__ = ("success", "accepted_state")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_STATE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    accepted_state: WorkerStateSnapshot
    def __init__(self, success: bool = ..., accepted_state: _Optional[_Union[WorkerStateSnapshot, _Mapping]] = ...) -> None: ...

class FreeDecodeStateRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class FreeDecodeStateResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class ConvertPhaseRequest(_message.Message):
    __slots__ = ("target_type",)
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    target_type: str
    def __init__(self, target_type: _Optional[str] = ...) -> None: ...

class ConvertPhaseResponse(_message.Message):
    __slots__ = ("success", "returned_prefill_requests", "returned_decode_requests", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    RETURNED_PREFILL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    RETURNED_DECODE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    returned_prefill_requests: _containers.RepeatedCompositeFieldContainer[EnqueuePrefillRequest]
    returned_decode_requests: _containers.RepeatedCompositeFieldContainer[EnqueueDecodeRequest]
    error_message: str
    def __init__(self, success: bool = ..., returned_prefill_requests: _Optional[_Iterable[_Union[EnqueuePrefillRequest, _Mapping]]] = ..., returned_decode_requests: _Optional[_Iterable[_Union[EnqueueDecodeRequest, _Mapping]]] = ..., error_message: _Optional[str] = ...) -> None: ...

class EnqueueColocatedRequest(_message.Message):
    __slots__ = ("request_id", "sequence_ids", "block_start", "block_end", "block_index", "total_generation_blocks", "request_seed", "sampling_parameters", "mask_id", "arrival_time", "prefill_enqueue_time")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_START_FIELD_NUMBER: _ClassVar[int]
    BLOCK_END_FIELD_NUMBER: _ClassVar[int]
    BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_GENERATION_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_SEED_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    MASK_ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    PREFILL_ENQUEUE_TIME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    sequence_ids: _containers.RepeatedScalarFieldContainer[int]
    block_start: int
    block_end: int
    block_index: int
    total_generation_blocks: int
    request_seed: int
    sampling_parameters: SamplingParameters
    mask_id: int
    arrival_time: float
    prefill_enqueue_time: float
    def __init__(self, request_id: _Optional[str] = ..., sequence_ids: _Optional[_Iterable[int]] = ..., block_start: _Optional[int] = ..., block_end: _Optional[int] = ..., block_index: _Optional[int] = ..., total_generation_blocks: _Optional[int] = ..., request_seed: _Optional[int] = ..., sampling_parameters: _Optional[_Union[SamplingParameters, _Mapping]] = ..., mask_id: _Optional[int] = ..., arrival_time: _Optional[float] = ..., prefill_enqueue_time: _Optional[float] = ...) -> None: ...

class EnqueueColocatedResponse(_message.Message):
    __slots__ = ("success", "accepted_state")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_STATE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    accepted_state: WorkerStateSnapshot
    def __init__(self, success: bool = ..., accepted_state: _Optional[_Union[WorkerStateSnapshot, _Mapping]] = ...) -> None: ...
