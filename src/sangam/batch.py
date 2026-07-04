from __future__ import annotations

from dataclasses import dataclass, field

from sangam.proto import sangam_pb2


@dataclass
class BatchRequestUpdate:
    request_id: str
    block_index: int
    success: bool
    updated_sequence: list[int]
    num_unmasked_tokens: int
    num_forward_evals_in_batch_phase: int
    request_phase: sangam_pb2.BatchPhase.ValueType
    prefill_duration: float = 0.0
    prefill_queue_wait_duration: float = 0.0
    kv_transfer_duration: float | None = None
    decode_duration: float = 0.0
    decode_queue_wait_duration: float = 0.0
    block_completed: bool = False

    def to_proto(self) -> sangam_pb2.BatchRequestUpdate:
        update = sangam_pb2.BatchRequestUpdate(
            request_id=self.request_id,
            block_index=self.block_index,
            success=self.success,
            updated_sequence=self.updated_sequence,
            num_unmasked_tokens=self.num_unmasked_tokens,
            num_forward_evals_in_batch_phase=self.num_forward_evals_in_batch_phase,
            request_phase=self.request_phase,
            prefill_duration=self.prefill_duration,
            prefill_queue_wait_duration=self.prefill_queue_wait_duration,
            decode_duration=self.decode_duration,
            decode_queue_wait_duration=self.decode_queue_wait_duration,
            block_completed=self.block_completed,
        )
        if self.kv_transfer_duration is not None:
            update.kv_transfer_duration = self.kv_transfer_duration
        return update


@dataclass
class Batch:
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
    batch_phase: sangam_pb2.BatchPhase.ValueType
    sampling_duration: float = 0.0
    batch_op_attn_time: float | None = None
    batch_op_mlp_time: float | None = None
    batch_op_qkv_time: float | None = None
    request_updates: list[BatchRequestUpdate] = field(default_factory=list)
    worker_state_after: sangam_pb2.WorkerStateSnapshot | None = None

    def to_proto(self) -> sangam_pb2.BatchMetricsReport:
        report = sangam_pb2.BatchMetricsReport(
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            batch_size=self.batch_size,
            prompt_len=self.prompt_len,
            gen_len=self.gen_len,
            batch_start_time=self.batch_start_time,
            batch_end_time=self.batch_end_time,
            kv_total_pages=self.kv_total_pages,
            kv_used_pages=self.kv_used_pages,
            kv_free_pages=self.kv_free_pages,
            num_unmasked_tokens=self.num_unmasked_tokens,
            batch_phase=self.batch_phase,
            sampling_duration=self.sampling_duration,
            request_updates=[update.to_proto() for update in self.request_updates],
        )
        if self.batch_op_attn_time is not None:
            report.batch_op_attn_time = self.batch_op_attn_time
        if self.batch_op_mlp_time is not None:
            report.batch_op_mlp_time = self.batch_op_mlp_time
        if self.batch_op_qkv_time is not None:
            report.batch_op_qkv_time = self.batch_op_qkv_time
        if self.worker_state_after is not None:
            report.worker_state_after.CopyFrom(self.worker_state_after)
        return report
