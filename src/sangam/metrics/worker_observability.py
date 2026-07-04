"""Worker-state tracking for metrics export."""

import os
from dataclasses import dataclass
from statistics import median
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sangam.metrics.constants import (
    CompletionMetricsTimeSeries,
    WorkerStateTimeline,
    WorkerSystemMetricsDistribution,
)
from sangam.metrics.export_utils import worker_plot_name

TIME_STR = "Time (s)"
COUNT_STR = "Count"
RATIO_STR = "Ratio"


@dataclass
class WorkerStateSnapshot:
    worker_type: str
    state: WorkerStateTimeline
    start_time: float
    waiting_queue_depth: int
    active_batch_size: int
    kv_total_pages: int
    kv_used_pages: int
    kv_free_pages: int


class WorkerObservability:
    def __init__(self) -> None:
        self.current_state: dict[str, WorkerStateSnapshot] = {}
        self.state_rows: list[dict[str, Union[str, int, float]]] = []
        self.drain_totals: dict[str, float] = {}

    def record_drain(self, worker_id: str, duration: float) -> None:
        self.drain_totals[worker_id] = self.drain_totals.get(worker_id, 0.0) + duration

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
        current = self.current_state.get(worker_id)
        if current is not None:
            if (
                current.state == state
                and current.waiting_queue_depth == waiting_queue_depth
                and current.active_batch_size == active_batch_size
                and current.kv_total_pages == kv_total_pages
                and current.kv_used_pages == kv_used_pages
                and current.kv_free_pages == kv_free_pages
            ):
                return
            # Clamp end_time to never go backwards — snapshots from concurrent
            # worker threads (gRPC handler vs _process_thread) can arrive
            # out of timestamp order even after event-queue reordering fixes.
            effective_end = max(timestamp, current.start_time)
            self._append_state_row(worker_id, current, effective_end)

        # Likewise clamp the new start_time so it never precedes the previous
        # interval's end, which would create overlapping intervals.
        new_start = max(
            timestamp, current.start_time if current is not None else timestamp
        )
        self.current_state[worker_id] = WorkerStateSnapshot(
            worker_type=worker_type,
            state=state,
            start_time=new_start,
            waiting_queue_depth=waiting_queue_depth,
            active_batch_size=active_batch_size,
            kv_total_pages=kv_total_pages,
            kv_used_pages=kv_used_pages,
            kv_free_pages=kv_free_pages,
        )

    def _append_state_row(
        self, worker_id: str, snapshot: WorkerStateSnapshot, end_time: float
    ) -> None:
        self.state_rows.append(
            {
                "worker_id": worker_id,
                "worker_type": snapshot.worker_type,
                "state": snapshot.state.value,
                "start_time": snapshot.start_time,
                "end_time": end_time,
                "duration_s": max(0.0, end_time - snapshot.start_time),
                WorkerSystemMetricsDistribution.QUEUE_DEPTH_WAITING.value: snapshot.waiting_queue_depth,
                WorkerSystemMetricsDistribution.QUEUE_DEPTH_ACTIVE.value: snapshot.active_batch_size,
            }
        )

    def _finalize_worker_states(self, completion_time_series) -> None:
        closing_time = completion_time_series[
            CompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].max_x
        if not closing_time:
            closing_time = completion_time_series[
                CompletionMetricsTimeSeries.REQUEST_ARRIVAL
            ].max_x
        for worker_id, snapshot in list(self.current_state.items()):
            self._append_state_row(
                worker_id, snapshot, closing_time or snapshot.start_time
            )
            del self.current_state[worker_id]

    def export(
        self,
        output_dir: str,
        plot_path: str,
        completion_time_series,
        start_time: float,
        scheduler_queue_time_series=None,
    ) -> None:
        self._finalize_worker_states(completion_time_series)
        self._export_worker_states(
            output_dir, plot_path, start_time, scheduler_queue_time_series
        )

    def _export_worker_states(
        self,
        output_dir: str,
        plot_path: str,
        start_time: float,
        scheduler_queue_time_series=None,
    ) -> None:
        if not self.state_rows:
            return

        os.makedirs(plot_path, exist_ok=True)
        timeline_df = pd.DataFrame(self.state_rows).sort_values(
            by=["worker_id", "start_time"]
        )
        export_timeline_df = timeline_df.copy()
        export_timeline_df["start_time"] -= start_time
        export_timeline_df["end_time"] -= start_time
        export_timeline_df.to_csv(
            f"{output_dir}/worker_timeline.csv",
            index=False,
            float_format="%.4f",
        )

        summary_rows = []
        for (worker_id, worker_type), worker_df in timeline_df.groupby(
            ["worker_id", "worker_type"]
        ):
            duration_by_state = worker_df.groupby("state")["duration_s"].sum().to_dict()
            total = float(worker_df["duration_s"].sum())
            idle_gaps = worker_df.loc[
                worker_df["state"] == WorkerStateTimeline.IDLE.value, "duration_s"
            ].tolist()
            summary_rows.append(
                {
                    "worker_id": worker_id,
                    "worker_type": worker_type,
                    "worker_label": f"{worker_id}/{worker_type}",
                    "busy_time_s": duration_by_state.get(
                        WorkerStateTimeline.BUSY.value, 0.0
                    ),
                    "queued_time_s": duration_by_state.get(
                        WorkerStateTimeline.QUEUED.value, 0.0
                    ),
                    "idle_time_s": duration_by_state.get(
                        WorkerStateTimeline.IDLE.value, 0.0
                    ),
                    "draining_time_s": self.drain_totals.get(worker_id, 0.0),
                    "utilization_ratio": (
                        duration_by_state.get(WorkerStateTimeline.BUSY.value, 0.0)
                        / total
                        if total > 0
                        else 0.0
                    ),
                    "num_idle_gaps": len(idle_gaps),
                    "p50_idle_gap_s": median(idle_gaps) if idle_gaps else 0.0,
                    "p95_idle_gap_s": float(pd.Series(idle_gaps).quantile(0.95))
                    if idle_gaps
                    else 0.0,
                    "mean_waiting_queue_depth": float(
                        worker_df[
                            WorkerSystemMetricsDistribution.QUEUE_DEPTH_WAITING.value
                        ].mean()
                    ),
                    "mean_active_batch_size": float(
                        worker_df[
                            WorkerSystemMetricsDistribution.QUEUE_DEPTH_ACTIVE.value
                        ].mean()
                    ),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["worker_id", "worker_type"]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = summary_df.melt(
            id_vars=["worker_id", "worker_type", "worker_label"],
            value_vars=[
                "busy_time_s",
                "queued_time_s",
                "idle_time_s",
                "draining_time_s",
            ],
            var_name="state",
            value_name="duration_s",
        )
        plot_df.to_csv(
            f"{plot_path}/{worker_plot_name('state_time')}.csv",
            index=False,
            float_format="%.4f",
        )
        sns.barplot(data=plot_df, x="worker_label", y="duration_s", hue="state", ax=ax)
        ax.set_title("Worker Time by State")
        ax.set_xlabel("Worker")
        ax.set_ylabel(TIME_STR)
        fig.tight_layout()
        fig.savefig(f"{plot_path}/{worker_plot_name('state_time')}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        summary_df[
            ["worker_id", "worker_type", "worker_label", "utilization_ratio"]
        ].to_csv(
            f"{plot_path}/{worker_plot_name('utilization_ratio')}.csv",
            index=False,
            float_format="%.4f",
        )
        sns.barplot(data=summary_df, x="worker_label", y="utilization_ratio", ax=ax)
        ax.set_title("Worker Utilization Ratio")
        ax.set_xlabel("Worker")
        ax.set_ylabel(RATIO_STR)
        fig.tight_layout()
        fig.savefig(f"{plot_path}/{worker_plot_name('utilization_ratio')}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        queue_depth_df = export_timeline_df[
            [
                "worker_id",
                "worker_type",
                "state",
                "start_time",
                WorkerSystemMetricsDistribution.QUEUE_DEPTH_WAITING.value,
            ]
        ].copy()
        queue_depth_df["worker_label"] = (
            queue_depth_df["worker_id"] + "/" + queue_depth_df["worker_type"]
        )

        if scheduler_queue_time_series is not None:
            depth_col = WorkerSystemMetricsDistribution.QUEUE_DEPTH_WAITING.value
            scheduler_rows: list[pd.DataFrame] = []
            for metric, data_series in scheduler_queue_time_series.items():
                if len(data_series) == 0:
                    continue
                short = metric.value.removeprefix("scheduler_").removesuffix(
                    "_requests"
                )
                label = f"scheduler/{short}"
                df = data_series.to_df()
                df = df.rename(
                    columns={TIME_STR: "start_time", metric.value: depth_col}
                )
                df["start_time"] -= start_time
                df["worker_id"] = label
                df["worker_type"] = "scheduler"
                df["state"] = "scheduler"
                df["worker_label"] = label
                scheduler_rows.append(df[queue_depth_df.columns])
            if scheduler_rows:
                queue_depth_df = pd.concat(
                    [queue_depth_df, *scheduler_rows], ignore_index=True
                )

        queue_depth_df.to_csv(
            f"{plot_path}/{worker_plot_name('queue_depth_time_series')}.csv",
            index=False,
            float_format="%.4f",
        )
        sns.scatterplot(
            data=queue_depth_df,
            x="start_time",
            y=WorkerSystemMetricsDistribution.QUEUE_DEPTH_WAITING.value,
            hue="worker_label",
            style="state",
            ax=ax,
        )
        ax.set_title("Worker Queue Depth Over Time")
        ax.set_xlabel(TIME_STR)
        ax.set_ylabel(COUNT_STR)
        fig.tight_layout()
        fig.savefig(
            f"{plot_path}/{worker_plot_name('queue_depth_time_series')}.png", dpi=150
        )
        plt.close(fig)

    def merge(self, other: "WorkerObservability") -> None:
        self.state_rows.extend(other.state_rows)
        for wid, dur in other.drain_totals.items():
            self.drain_totals[wid] = self.drain_totals.get(wid, 0.0) + dur
