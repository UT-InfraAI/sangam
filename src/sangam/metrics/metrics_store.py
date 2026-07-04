"""Central metrics collection store for sangam."""

import logging
import os
import statistics
from collections import defaultdict
from typing import Dict, Union

import matplotlib
import pandas as pd
import seaborn as sns

from sangam.metrics.cdf_sketch import CDFSketch
from sangam.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    BlockMetricsTimeDistribution,
    CompletionMetricsTimeSeries,
    RequestMetricsCDFSketch,
    RequestMetricsHistogram,
    RequestMetricsTimeDistribution,
    SchedulerQueueTimeSeries,
    WorkerBatchTimeSeries,
    WorkerStateTimeline,
    WorkerSystemMetricsDistribution,
)
from sangam.metrics.data_series import (
    _MAX_TIME_SERIES_POINTS,
    DataSeries,
    downsample_rows,
)
from sangam.metrics.export_utils import (
    plot_worker_metric_cdf,
    save_merged_dataseries_csv,
    worker_plot_name,
)
from sangam.metrics.worker_observability import WorkerObservability
from sangam.proto import sangam_pb2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

REQUEST_ID_STR = "Request Id"
BLOCK_ID_STR = "Block Id"
TIME_STR = "Time (s)"
COUNT_STR = "Count"
RATIO_STR = "Ratio"
_BATCH_OP_ATTN_TIME = "batch_op_attn_time"
_BATCH_OP_MLP_TIME = "batch_op_mlp_time"
_BATCH_OP_QKV_TIME = "batch_op_qkv_time"


_BATCH_PHASE_TO_EXECUTION_METRIC = {
    sangam_pb2.BATCH_PHASE_PREFILL: BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_PREFILL,
    sangam_pb2.BATCH_PHASE_DECODE: BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_DECODE,
    sangam_pb2.BATCH_PHASE_MIXED: BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_MIXED,
}

_BATCH_PHASE_TO_LABEL = {
    sangam_pb2.BATCH_PHASE_PREFILL: "prefill",
    sangam_pb2.BATCH_PHASE_DECODE: "decode",
    sangam_pb2.BATCH_PHASE_MIXED: "mixed",
}

_BATCH_PHASE_EXECUTION_EXPORT_EXCLUSIONS = {
    BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_PREFILL,
    BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_DECODE,
    BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME_MIXED,
}


def _check_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return
        return func(self, *args, **kwargs)

    return wrapper


class MetricsStore:
    _instance: "MetricsStore | None" = None

    def __init__(
        self,
        output_dir: str,
        enabled: bool,
        enable_individual_batch_metrics: bool,
    ):
        self.enabled = enabled
        self.output_dir = output_dir
        self.enable_individual_batch_metrics = enable_individual_batch_metrics

        if not enabled:
            logger.debug("MetricsStore disabled")
            return

        self._next_batch_id: Dict[str, int] = defaultdict(int)
        self.reset()

    @classmethod
    def get_or_create_instance(
        cls,
        output_dir: str,
        enabled: bool,
        enable_individual_batch_metrics: bool,
    ):
        if cls._instance is None:
            cls._instance = cls(
                output_dir,
                enabled,
                enable_individual_batch_metrics,
            )
            return cls._instance

        existing_config = (
            cls._instance.output_dir,
            cls._instance.enabled,
            cls._instance.enable_individual_batch_metrics,
        )
        requested_config = (
            output_dir,
            enabled,
            enable_individual_batch_metrics,
        )
        if existing_config != requested_config:
            logger.warning(
                "MetricsStore singleton already initialized with a different "
                f"configuration: existing={existing_config}, requested={requested_config}"
            )

        return cls._instance

    @classmethod
    def get_instance(cls):
        return cls._instance

    def reset(self):
        if not self.enabled:
            return

        self.request_time_distributions: Dict[
            RequestMetricsTimeDistribution, DataSeries
        ] = {
            metric: DataSeries(REQUEST_ID_STR, metric.value)
            for metric in RequestMetricsTimeDistribution
        }
        self.request_histograms: Dict[RequestMetricsHistogram, DataSeries] = {
            metric: DataSeries(REQUEST_ID_STR, metric.value)
            for metric in RequestMetricsHistogram
        }
        self.request_cdf_sketches: Dict[RequestMetricsCDFSketch, CDFSketch] = {
            metric: CDFSketch(metric.value, num_quantiles_in_df=1001)
            for metric in RequestMetricsCDFSketch
        }
        self.block_time_distributions: Dict[
            BlockMetricsTimeDistribution, DataSeries
        ] = {
            metric: DataSeries(BLOCK_ID_STR, metric.value)
            for metric in BlockMetricsTimeDistribution
        }
        # Worker attribution side-tables for block_metrics.csv, keyed by Block Id.
        # Block ids/worker ids are unique, so these are plain dicts (not DataSeries,
        # which only holds numeric (x, y) pairs).
        self._block_prefill_worker_id: Dict[str, str] = {}
        self._block_decode_worker_id: Dict[str, str] = {}
        self._worker_id_to_type: Dict[str, str] = {}
        self.worker_batch_count: Dict[
            str, Dict[BatchMetricsCountDistribution, CDFSketch]
        ] = defaultdict(
            lambda: {
                metric: CDFSketch(metric.value)
                for metric in BatchMetricsCountDistribution
            }
        )
        self.worker_batch_time: Dict[
            str, Dict[BatchMetricsTimeDistribution, CDFSketch]
        ] = defaultdict(
            lambda: {
                metric: CDFSketch(metric.value)
                for metric in BatchMetricsTimeDistribution
            }
        )
        self.worker_system_metrics: Dict[
            str, Dict[WorkerSystemMetricsDistribution, CDFSketch]
        ] = defaultdict(
            lambda: {
                metric: CDFSketch(metric.value)
                for metric in WorkerSystemMetricsDistribution
            }
        )
        self.completion_time_series: Dict[CompletionMetricsTimeSeries, DataSeries] = {
            metric: DataSeries(TIME_STR, metric.value)
            for metric in CompletionMetricsTimeSeries
        }
        self.worker_batch_time_series: Dict[
            str, Dict[WorkerBatchTimeSeries, DataSeries]
        ] = defaultdict(
            lambda: {
                metric: DataSeries(TIME_STR, metric.value)
                for metric in WorkerBatchTimeSeries
            }
        )
        self.worker_batch_phase_time_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"prefill": 0.0, "decode": 0.0, "mixed": 0.0}
        )
        self.scheduler_queue_time_series: Dict[SchedulerQueueTimeSeries, DataSeries] = {
            metric: DataSeries(TIME_STR, metric.value)
            for metric in SchedulerQueueTimeSeries
        }
        self.worker_outstanding_prefill_tokens_time_series: Dict[str, DataSeries] = (
            defaultdict(
                lambda: DataSeries(
                    TIME_STR,
                    WorkerSystemMetricsDistribution.OUTSTANDING_PREFILL_TOKENS.value,
                )
            )
        )
        self.scheduler_pending_prefill_tokens_time_series: DataSeries = DataSeries(
            TIME_STR,
            WorkerSystemMetricsDistribution.OUTSTANDING_PREFILL_TOKENS.value,
        )

        self.last_request_arrived_at: float | None = None
        self._last_request_visibility_at: Dict[str, float] = {}
        self._last_batch_end_time: Dict[str, float] = {}
        self._worker_batch_rows: Dict[str, list[dict[str, Union[str, int, float]]]] = (
            defaultdict(list)
        )
        self._worker_observability = WorkerObservability()
        self._include_batch_operation_metrics = False

    @_check_enabled
    def on_request_arrival(self, request_id: str, submit_time: float) -> None:
        self.completion_time_series[CompletionMetricsTimeSeries.REQUEST_ARRIVAL].put(
            submit_time, 1
        )
        if self.last_request_arrived_at is not None:
            self.request_histograms[
                RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
            ].put(request_id, submit_time - self.last_request_arrived_at)
        self.last_request_arrived_at = submit_time

    @_check_enabled
    def on_request_end(self, request) -> None:
        request_id = request.request_id
        self.completion_time_series[CompletionMetricsTimeSeries.REQUEST_COMPLETION].put(
            request.complete_time, 1
        )

        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_E2E_TIME
        ].put(request_id, request.e2e_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_E2E_TIME_NORMALIZED
        ].put(request_id, request.e2e_time_normalized)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_E2E_TIME_EXCL_FIRST_BLOCK_QUEUE
        ].put(request_id, request.e2e_time_excl_first_block_queue)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_E2E_TIME_EXCL_FIRST_BLOCK_QUEUE_NORMALIZED
        ].put(request_id, request.e2e_time_excl_first_block_queue_normalized)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_FIRST_BLOCK_QUEUE
        ].put(request_id, request.first_block_queue_wait_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_SCHEDULING_DELAY
        ].put(request_id, request.total_queue_wait_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_PREFILL_SCHEDULING_DELAY
        ].put(request_id, request.total_prefill_queue_wait_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_DECODE_SCHEDULING_DELAY
        ].put(request_id, request.total_decode_queue_wait_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_EXECUTION_TIME
        ].put(request_id, request.execution_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request_id, request.execution_time_normalized)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_PREFILL_TIME
        ].put(request_id, request.total_prefill_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_KV_TRANSFER_TIME
        ].put(request_id, request.total_kv_transfer_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_KV_TRANSFER_TIME_NONOVERLAPPED
        ].put(request_id, request.total_kv_transfer_time_nonoverlapped)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_DECODE_TIME
        ].put(request_id, request.total_decode_time)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_DECODE_TIME_NORMALIZED
        ].put(request_id, request.total_decode_time_normalized)
        self.request_time_distributions[
            RequestMetricsTimeDistribution.REQUEST_UNACCOUNTED_TIME
        ].put(request_id, request.unaccounted_time)

        self.request_histograms[RequestMetricsHistogram.REQUEST_NUM_PROMPT_TOKENS].put(
            request_id, request.prompt_length
        )
        self.request_histograms[RequestMetricsHistogram.REQUEST_NUM_GEN_TOKENS].put(
            request_id, request.target_gen_tokens
        )
        self.request_histograms[RequestMetricsHistogram.REQUEST_NUM_BLOCKS].put(
            request_id, request.target_blocks
        )
        self.request_histograms[RequestMetricsHistogram.REQUEST_NUM_FORWARD_PASSES].put(
            request_id, request.num_forward_evals
        )

    @_check_enabled
    def on_request_visibility(
        self,
        request_id: str,
        timestamp: float,
        num_unmasked_tokens: int,
    ) -> None:
        if num_unmasked_tokens <= 0:
            return
        last_visible_at = self._last_request_visibility_at.get(request_id)
        if last_visible_at is not None:
            self.request_cdf_sketches[
                RequestMetricsCDFSketch.REQUEST_TIME_BETWEEN_TOKENS
            ].put(max(0.0, timestamp - last_visible_at))
        for _ in range(num_unmasked_tokens - 1):
            self.request_cdf_sketches[
                RequestMetricsCDFSketch.REQUEST_TIME_BETWEEN_TOKENS
            ].put(0.0)
        self._last_request_visibility_at[request_id] = timestamp
        self.request_cdf_sketches[
            RequestMetricsCDFSketch.REQUEST_TOKENS_UNMASKED_PER_FORWARD_PASS
        ].put(num_unmasked_tokens)

    @_check_enabled
    def on_block_prefill_end(
        self, request_id: str, block_index: int, duration: float, worker_id: str
    ) -> None:
        block_id = f"{request_id}_b{block_index}"
        self.block_time_distributions[
            BlockMetricsTimeDistribution.BLOCK_PREFILL_TIME
        ].put(block_id, duration)
        self._block_prefill_worker_id[block_id] = worker_id

    @_check_enabled
    def on_block_kv_transfer_end(
        self, request_id: str, block_index: int, duration: float
    ) -> None:
        self.block_time_distributions[
            BlockMetricsTimeDistribution.BLOCK_KV_TRANSFER_TIME_NONOVERLAPPED
        ].put(f"{request_id}_b{block_index}", duration)

    @_check_enabled
    def on_block_decode_end(
        self, request_id: str, block_index: int, duration: float
    ) -> None:
        self.block_time_distributions[
            BlockMetricsTimeDistribution.BLOCK_DECODE_TIME
        ].put(f"{request_id}_b{block_index}", duration)

    @_check_enabled
    def on_block_end(
        self, request_id: str, block_index: int, duration: float, worker_id: str
    ) -> None:
        block_id = f"{request_id}_b{block_index}"
        self.block_time_distributions[
            BlockMetricsTimeDistribution.BLOCK_TOTAL_TIME
        ].put(block_id, duration)
        self._block_decode_worker_id[block_id] = worker_id

    @_check_enabled
    def on_batch_end(
        self,
        worker_id: str,
        worker_type: str,
        batch_size: int,
        prompt_len: int,
        gen_len: int,
        batch_start_time: float,
        batch_end_time: float,
        kv_total_pages: int,
        kv_used_pages: int,
        kv_free_pages: int,
        num_unmasked_tokens: int,
        batch_phase: int,
        sampling_duration: float,
        request_updates,
        batch_op_attn_time: float | None = None,
        batch_op_mlp_time: float | None = None,
        batch_op_qkv_time: float | None = None,
    ) -> None:
        self._worker_id_to_type[worker_id] = worker_type
        batch_key = f"{worker_id}/{worker_type}"
        batch_id = self._next_batch_id[batch_key]
        self._next_batch_id[batch_key] += 1
        num_tokens = prompt_len + gen_len
        execution_time = max(0.0, batch_end_time - batch_start_time)
        batch_token_throughput = num_tokens / max(execution_time, 1e-12)
        kv_page_utilization_ratio = (
            kv_used_pages / kv_total_pages if kv_total_pages > 0 else 0.0
        )
        batch_phase_label = _BATCH_PHASE_TO_LABEL.get(batch_phase, "unknown")
        if (
            batch_op_attn_time is not None
            or batch_op_mlp_time is not None
            or batch_op_qkv_time is not None
        ):
            self._include_batch_operation_metrics = True

        self.worker_batch_count[batch_key][
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS
        ].put(num_tokens)
        self.worker_batch_count[batch_key][
            BatchMetricsCountDistribution.BATCH_prompt_len
        ].put(prompt_len)
        self.worker_batch_count[batch_key][
            BatchMetricsCountDistribution.BATCH_gen_len
        ].put(gen_len)
        self.worker_batch_count[batch_key][
            BatchMetricsCountDistribution.BATCH_NUM_UNMASKED_TOKENS
        ].put(num_unmasked_tokens)
        self.worker_batch_count[batch_key][
            BatchMetricsCountDistribution.BATCH_SIZE
        ].put(batch_size)

        self.worker_batch_time[batch_key][
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME
        ].put(execution_time)
        self.worker_batch_time[batch_key][
            BatchMetricsTimeDistribution.BATCH_SAMPLING_TIME
        ].put(sampling_duration)
        self.worker_batch_time[batch_key][
            BatchMetricsTimeDistribution.BATCH_TOKEN_THROUGHPUT
        ].put(batch_token_throughput)
        batch_phase_metric = _BATCH_PHASE_TO_EXECUTION_METRIC.get(batch_phase)
        if batch_phase_metric is not None:
            self.worker_batch_time[batch_key][batch_phase_metric].put(execution_time)
            self.worker_batch_phase_time_totals[batch_key][batch_phase_label] += (
                execution_time
            )

        decode_lengths: list[int] = []
        for update in request_updates:
            effective_phase = getattr(update, "request_phase", batch_phase)
            if effective_phase != sangam_pb2.BATCH_PHASE_DECODE:
                continue
            decode_lengths.append(len(getattr(update, "updated_sequence", [])))
        if decode_lengths:
            self.worker_batch_time_series[batch_key][
                WorkerBatchTimeSeries.DECODE_LENGTH_SUM
            ].put(batch_end_time, sum(decode_lengths))

        decode_length_std: float | None = None
        if len(decode_lengths) >= 2:
            decode_length_std = statistics.pstdev(decode_lengths)
            self.worker_batch_count[batch_key][
                BatchMetricsCountDistribution.BATCH_DECODE_LENGTH_STD
            ].put(decode_length_std)

        self.worker_system_metrics[batch_key][
            WorkerSystemMetricsDistribution.KV_PAGE_UTILIZATION_RATIO
        ].put(kv_page_utilization_ratio)
        last_batch_end = self._last_batch_end_time.get(batch_key)
        inter_batch_delay = (
            max(0.0, batch_start_time - last_batch_end)
            if last_batch_end is not None
            else 0.0
        )
        if last_batch_end is not None:
            self.worker_batch_time[batch_key][
                BatchMetricsTimeDistribution.INTER_BATCH_DELAY
            ].put(inter_batch_delay)
        self._last_batch_end_time[batch_key] = batch_end_time

        if not self.enable_individual_batch_metrics:
            return

        row = {
            "worker_id": worker_id,
            "worker_type": worker_type,
            "batch_id": batch_id,
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS.value: num_tokens,
            BatchMetricsCountDistribution.BATCH_prompt_len.value: prompt_len,
            BatchMetricsCountDistribution.BATCH_gen_len.value: gen_len,
            BatchMetricsCountDistribution.BATCH_NUM_UNMASKED_TOKENS.value: num_unmasked_tokens,
            BatchMetricsCountDistribution.BATCH_SIZE.value: batch_size,
            BatchMetricsCountDistribution.BATCH_DECODE_LENGTH_STD.value: decode_length_std,
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME.value: execution_time,
            BatchMetricsTimeDistribution.BATCH_SAMPLING_TIME.value: sampling_duration,
            BatchMetricsTimeDistribution.BATCH_TOKEN_THROUGHPUT.value: batch_token_throughput,
            BatchMetricsTimeDistribution.INTER_BATCH_DELAY.value: inter_batch_delay,
            "batch_phase": batch_phase_label,
            WorkerSystemMetricsDistribution.KV_PAGE_UTILIZATION_RATIO.value: kv_page_utilization_ratio,
        }
        if self._include_batch_operation_metrics:
            row[_BATCH_OP_ATTN_TIME] = (
                batch_op_attn_time if batch_op_attn_time is not None else 0.0
            )
            row[_BATCH_OP_MLP_TIME] = (
                batch_op_mlp_time if batch_op_mlp_time is not None else 0.0
            )
            row[_BATCH_OP_QKV_TIME] = (
                batch_op_qkv_time if batch_op_qkv_time is not None else 0.0
            )
        self._worker_batch_rows[batch_key].append(row)

    @_check_enabled
    def on_worker_state(
        self,
        worker_id: str,
        worker_type: str,
        state: WorkerStateTimeline,
        timestamp: float,
        waiting_queue_depth: int,
        active_batch_size: int,
        kv_total_pages: int,
        kv_used_pages: int,
        kv_free_pages: int,
    ) -> None:
        self._worker_observability.on_worker_state(
            worker_id=worker_id,
            worker_type=worker_type,
            state=state,
            timestamp=timestamp,
            waiting_queue_depth=waiting_queue_depth,
            active_batch_size=active_batch_size,
            kv_total_pages=kv_total_pages,
            kv_used_pages=kv_used_pages,
            kv_free_pages=kv_free_pages,
        )

    @_check_enabled
    def on_scheduler_queue_depth(
        self,
        queue: SchedulerQueueTimeSeries,
        depth: int,
        timestamp: float,
    ) -> None:
        self.scheduler_queue_time_series[queue].put(timestamp, depth)

    @_check_enabled
    def on_outstanding_prefill_tokens(
        self,
        worker_id: str,
        tokens: int,
        timestamp: float,
    ) -> None:
        self.worker_outstanding_prefill_tokens_time_series[worker_id].put(
            timestamp, tokens
        )

    @_check_enabled
    def on_scheduler_pending_prefill_tokens(
        self,
        tokens: int,
        timestamp: float,
    ) -> None:
        self.scheduler_pending_prefill_tokens_time_series.put(timestamp, tokens)

    @_check_enabled
    def on_worker_deficit_tokens(
        self,
        worker_id: str,
        worker_type: str,
        deficit_tokens: int,
        timestamp: float,
    ) -> None:
        batch_key = f"{worker_id}/{worker_type}"
        self.worker_batch_time_series[batch_key][
            WorkerBatchTimeSeries.DEFICIT_TOKENS
        ].put(timestamp, deficit_tokens)

    def _store_request_metrics(self, plot_path: str) -> None:
        all_request_metrics = list(self.request_time_distributions.values()) + list(
            self.request_histograms.values()
        )
        non_empty = [
            data_series for data_series in all_request_metrics if len(data_series) > 0
        ]
        if non_empty:
            save_merged_dataseries_csv(
                non_empty,
                REQUEST_ID_STR,
                self.output_dir,
                "request_metrics",
                "%.4f",
            )
        for data_series in self.request_histograms.values():
            data_series.plot_histogram(plot_path, data_series.y_name)
        for data_series in self.request_time_distributions.values():
            data_series.plot_cdf(plot_path, data_series.y_name, TIME_STR)
        for sketch in self.request_cdf_sketches.values():
            sketch.plot_cdf(plot_path, sketch.metric_name)

    def _block_worker_attribution_df(self) -> pd.DataFrame | None:
        block_ids = set(self._block_prefill_worker_id) | set(
            self._block_decode_worker_id
        )
        if not block_ids:
            return None
        rows = []
        for block_id in block_ids:
            prefill_worker_id = self._block_prefill_worker_id.get(block_id)
            decode_worker_id = self._block_decode_worker_id.get(block_id)
            rows.append(
                {
                    BLOCK_ID_STR: block_id,
                    "prefill_worker_id": prefill_worker_id,
                    "prefill_worker_type": self._worker_id_to_type.get(
                        prefill_worker_id
                    ),
                    "decode_worker_id": decode_worker_id,
                    "decode_worker_type": self._worker_id_to_type.get(decode_worker_id),
                }
            )
        return pd.DataFrame(rows)

    def _store_block_metrics(self, plot_path: str) -> None:
        non_empty = [
            data_series
            for data_series in self.block_time_distributions.values()
            if len(data_series) > 0
        ]
        if non_empty:
            save_merged_dataseries_csv(
                non_empty,
                BLOCK_ID_STR,
                self.output_dir,
                "block_metrics",
                "%.4f",
                extra_columns_df=self._block_worker_attribution_df(),
            )
        for data_series in self.block_time_distributions.values():
            data_series.plot_cdf(plot_path, data_series.y_name, TIME_STR)

    def _metrics_start_time(self) -> float:
        return float(
            self.completion_time_series[
                CompletionMetricsTimeSeries.REQUEST_ARRIVAL
            ].min_x
        )

    def _store_worker_batch_metrics(self, plot_path: str, start_time: float) -> None:
        os.makedirs(plot_path, exist_ok=True)

        for metric in BatchMetricsCountDistribution:
            plot_worker_metric_cdf(
                output_dir=plot_path,
                metric_name=worker_plot_name(metric.value),
                metric_column=metric.value,
                x_label=COUNT_STR,
                source_map=self.worker_batch_count,
                metric_enum=metric,
            )
        for metric in BatchMetricsTimeDistribution:
            if metric in _BATCH_PHASE_EXECUTION_EXPORT_EXCLUSIONS:
                continue
            plot_worker_metric_cdf(
                output_dir=plot_path,
                metric_name=worker_plot_name(metric.value),
                metric_column=metric.value,
                x_label=TIME_STR,
                source_map=self.worker_batch_time,
                metric_enum=metric,
            )
        for metric in WorkerSystemMetricsDistribution:
            plot_worker_metric_cdf(
                output_dir=plot_path,
                metric_name=worker_plot_name(metric.value),
                metric_column=metric.value,
                x_label=RATIO_STR
                if metric == WorkerSystemMetricsDistribution.KV_PAGE_UTILIZATION_RATIO
                else COUNT_STR,
                source_map=self.worker_system_metrics,
                metric_enum=metric,
            )
        self._store_worker_batch_time_series_metrics(plot_path, start_time)
        self._store_worker_batch_phase_time_totals(plot_path)
        if not self.enable_individual_batch_metrics:
            return

        detailed_rows = []
        for worker_id in sorted(self._worker_batch_rows.keys()):
            detailed_rows.extend(self._worker_batch_rows[worker_id])
        if not detailed_rows:
            return

        detailed_df = pd.DataFrame(detailed_rows).sort_values(
            by=["worker_id", "batch_id"]
        )
        if _BATCH_OP_ATTN_TIME in detailed_df.columns:
            detailed_df[_BATCH_OP_ATTN_TIME] = detailed_df[_BATCH_OP_ATTN_TIME].fillna(
                0.0
            )
        if _BATCH_OP_MLP_TIME in detailed_df.columns:
            detailed_df[_BATCH_OP_MLP_TIME] = detailed_df[_BATCH_OP_MLP_TIME].fillna(
                0.0
            )
        detailed_df.to_csv(
            f"{self.output_dir}/worker_batch_metrics.csv",
            index=False,
            float_format="%.4f",
        )

    def _store_worker_batch_time_series_metrics(
        self, plot_path: str, start_time: float
    ) -> None:
        for metric in WorkerBatchTimeSeries:
            worker_dfs: list[pd.DataFrame] = []
            for worker_id in sorted(self.worker_batch_time_series.keys()):
                data_series = self.worker_batch_time_series[worker_id][metric]
                if len(data_series) == 0:
                    continue
                df = data_series.to_df()
                df[TIME_STR] -= start_time
                df["worker_id"] = worker_id
                worker_dfs.append(downsample_rows(df, _MAX_TIME_SERIES_POINTS))
            if not worker_dfs:
                continue

            metric_df = pd.concat(worker_dfs, ignore_index=True).sort_values(
                by=TIME_STR
            )
            plot_name = worker_plot_name(f"{metric.value}_time_series")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(
                data=metric_df,
                x=TIME_STR,
                y=metric.value,
                hue="worker_id",
                marker="o",
                markersize=2,
                ax=ax,
            )
            ax.set_xlabel(TIME_STR)
            ax.set_ylabel(COUNT_STR)
            ax.set_title(plot_name)
            fig.tight_layout()
            fig.savefig(f"{plot_path}/{plot_name}.png", dpi=150)
            plt.close(fig)
            metric_df.to_csv(f"{plot_path}/{plot_name}.csv", index=False)

    def _store_worker_batch_phase_time_totals(self, plot_path: str) -> None:
        rows: list[dict[str, Union[str, float]]] = []
        for worker_id in sorted(self.worker_batch_phase_time_totals.keys()):
            phase_totals = self.worker_batch_phase_time_totals[worker_id]
            for phase in ("prefill", "decode", "mixed"):
                rows.append(
                    {
                        "worker_id": worker_id,
                        "batch_phase": phase,
                        "execution_time_s": phase_totals.get(phase, 0.0),
                    }
                )
        if not rows:
            return
        totals_df = pd.DataFrame(rows)
        totals_name = "worker_batch_phase_time_totals"
        totals_df.to_csv(
            f"{plot_path}/{totals_name}.csv",
            index=False,
            float_format="%.6f",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=totals_df,
            x="worker_id",
            y="execution_time_s",
            hue="batch_phase",
            ax=ax,
        )
        ax.set_xlabel("worker_id")
        ax.set_ylabel(TIME_STR)
        ax.set_title(totals_name)
        fig.tight_layout()
        fig.savefig(f"{plot_path}/{totals_name}.png", dpi=150)
        plt.close(fig)

    def _store_scheduler_queue_time_series(
        self, plot_path: str, start_time: float
    ) -> None:
        queue_dfs: list[pd.DataFrame] = []
        for metric in SchedulerQueueTimeSeries:
            data_series = self.scheduler_queue_time_series[metric]
            if len(data_series) == 0:
                continue
            df = data_series.to_df()
            df[TIME_STR] -= start_time
            df = df.rename(columns={metric.value: "depth"})
            df["queue"] = metric.value
            queue_dfs.append(downsample_rows(df, _MAX_TIME_SERIES_POINTS))
        if not queue_dfs:
            return

        plot_name = "scheduler_queue_depth_time_series"
        merged_df = pd.concat(queue_dfs, ignore_index=True).sort_values(by=TIME_STR)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=merged_df,
            x=TIME_STR,
            y="depth",
            hue="queue",
            marker="o",
            markersize=2,
            ax=ax,
        )
        ax.set_xlabel(TIME_STR)
        ax.set_ylabel(COUNT_STR)
        ax.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(f"{plot_path}/{plot_name}.png", dpi=150)
        plt.close(fig)
        merged_df.to_csv(
            f"{plot_path}/{plot_name}.csv", index=False, float_format="%.4f"
        )

    def _store_worker_outstanding_prefill_tokens(
        self, plot_path: str, start_time: float
    ) -> None:
        worker_dfs: list[pd.DataFrame] = []
        for worker_id in sorted(
            self.worker_outstanding_prefill_tokens_time_series.keys()
        ):
            data_series = self.worker_outstanding_prefill_tokens_time_series[worker_id]
            if len(data_series) == 0:
                continue
            df = data_series.to_df()
            df[TIME_STR] -= start_time
            df["worker_id"] = worker_id
            worker_dfs.append(downsample_rows(df, _MAX_TIME_SERIES_POINTS))
        if not worker_dfs:
            return

        if len(self.scheduler_pending_prefill_tokens_time_series) > 0:
            scheduler_df = self.scheduler_pending_prefill_tokens_time_series.to_df()
            scheduler_df[TIME_STR] -= start_time
            scheduler_df["worker_id"] = "scheduler_pending"
            worker_dfs.append(downsample_rows(scheduler_df, _MAX_TIME_SERIES_POINTS))

        plot_name = worker_plot_name("outstanding_prefill_tokens_time_series")
        merged_df = pd.concat(worker_dfs, ignore_index=True).sort_values(by=TIME_STR)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=merged_df,
            x=TIME_STR,
            y=WorkerSystemMetricsDistribution.OUTSTANDING_PREFILL_TOKENS.value,
            hue="worker_id",
            marker="o",
            markersize=2,
            ax=ax,
        )
        ax.set_xlabel(TIME_STR)
        ax.set_ylabel(COUNT_STR)
        ax.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(f"{plot_path}/{plot_name}.png", dpi=150)
        plt.close(fig)
        merged_df.to_csv(
            f"{plot_path}/{plot_name}.csv", index=False, float_format="%.4f"
        )

    def _store_completion_metrics(self, plot_path: str) -> None:
        first_arrival = self._metrics_start_time()
        for data_series in self.completion_time_series.values():
            data_series.plot_step(
                plot_path,
                f"{data_series.y_name}_time_series",
                COUNT_STR,
                start_time=first_arrival,
            )

    @_check_enabled
    def plot(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        request_plot_path = f"{self.output_dir}/request_plots"
        worker_plot_path = f"{self.output_dir}/worker_plots"
        os.makedirs(request_plot_path, exist_ok=True)
        os.makedirs(worker_plot_path, exist_ok=True)
        start_time = self._metrics_start_time()

        self._store_request_metrics(request_plot_path)
        self._store_block_metrics(request_plot_path)
        self._store_worker_batch_metrics(worker_plot_path, start_time)
        self._store_scheduler_queue_time_series(worker_plot_path, start_time)
        self._store_worker_outstanding_prefill_tokens(worker_plot_path, start_time)
        self._worker_observability.export(
            output_dir=self.output_dir,
            plot_path=worker_plot_path,
            completion_time_series=self.completion_time_series,
            start_time=start_time,
            scheduler_queue_time_series=self.scheduler_queue_time_series,
        )
        self._store_completion_metrics(request_plot_path)
        logger.info("Metrics exported to %s", self.output_dir)

    @_check_enabled
    def merge(self, other: "MetricsStore") -> None:
        for metric in RequestMetricsTimeDistribution:
            self.request_time_distributions[metric].merge(
                other.request_time_distributions[metric]
            )
        for metric in RequestMetricsHistogram:
            self.request_histograms[metric].merge(other.request_histograms[metric])
        for metric in RequestMetricsCDFSketch:
            self.request_cdf_sketches[metric].merge(other.request_cdf_sketches[metric])
        for metric in BlockMetricsTimeDistribution:
            self.block_time_distributions[metric].merge(
                other.block_time_distributions[metric]
            )
        self._block_prefill_worker_id.update(other._block_prefill_worker_id)
        self._block_decode_worker_id.update(other._block_decode_worker_id)
        self._worker_id_to_type.update(other._worker_id_to_type)
        for metric in CompletionMetricsTimeSeries:
            self.completion_time_series[metric].merge(
                other.completion_time_series[metric]
            )
        for worker_id in other.worker_batch_count:
            for metric in BatchMetricsCountDistribution:
                self.worker_batch_count[worker_id][metric].merge(
                    other.worker_batch_count[worker_id][metric]
                )
            for metric in BatchMetricsTimeDistribution:
                self.worker_batch_time[worker_id][metric].merge(
                    other.worker_batch_time[worker_id][metric]
                )
        for worker_id in other.worker_system_metrics:
            for metric in WorkerSystemMetricsDistribution:
                self.worker_system_metrics[worker_id][metric].merge(
                    other.worker_system_metrics[worker_id][metric]
                )
        for worker_id in other.worker_batch_time_series:
            for metric in WorkerBatchTimeSeries:
                self.worker_batch_time_series[worker_id][metric].merge(
                    other.worker_batch_time_series[worker_id][metric]
                )
        for metric in SchedulerQueueTimeSeries:
            self.scheduler_queue_time_series[metric].merge(
                other.scheduler_queue_time_series[metric]
            )
        for (
            worker_id,
            series,
        ) in other.worker_outstanding_prefill_tokens_time_series.items():
            self.worker_outstanding_prefill_tokens_time_series[worker_id].merge(series)
        self.scheduler_pending_prefill_tokens_time_series.merge(
            other.scheduler_pending_prefill_tokens_time_series
        )
        for worker_id, totals in other.worker_batch_phase_time_totals.items():
            for phase, value in totals.items():
                self.worker_batch_phase_time_totals[worker_id][phase] += value
        for worker_id, rows in other._worker_batch_rows.items():
            self._worker_batch_rows[worker_id].extend(rows)
            if rows:
                max_batch_id = max(int(row["batch_id"]) for row in rows)
                self._next_batch_id[worker_id] = max(
                    self._next_batch_id[worker_id],
                    max_batch_id + 1,
                )
        for request_id, timestamp in other._last_request_visibility_at.items():
            self._last_request_visibility_at[request_id] = max(
                self._last_request_visibility_at.get(request_id, timestamp),
                timestamp,
            )
        self._worker_observability.merge(other._worker_observability)
