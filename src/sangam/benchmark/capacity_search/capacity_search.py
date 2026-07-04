from __future__ import annotations

import json
import operator
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from sangam.benchmark.capacity_search.config import (
    CapacitySearchConfig,
    JobSpec,
    _sanitize_name,
)
from sangam.benchmark.config import BenchmarkConfig
from sangam.logger import init_logger
from sangam.process_lifecycle import (
    TRIAL_KILL_TIMEOUT_SECONDS,
    TRIAL_TERM_TIMEOUT_SECONDS,
    spawn_process_group,
    stop_process_group,
)

logger = init_logger(__name__)

_OPS = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
}

_REQUEST_METRICS_FILE = "request_metrics.csv"
_WORKER_TIMELINE_FILE = "worker_timeline.csv"
_BENCHMARK_RESULTS_FILE = "benchmark_results.json"
_WORKER_TIMELINE_METRICS = {"queue_depth_waiting"}


def _format_qps(qps: float) -> str:
    return f"{qps:.4f}".rstrip("0").rstrip(".")


def _weighted_quantile(
    values: pd.Series,
    weights: pd.Series,
    quantile: float,
) -> float:
    valid_mask = values.notna() & weights.notna()
    filtered_values = values[valid_mask].astype(float)
    filtered_weights = weights[valid_mask].astype(float)
    positive_mask = filtered_weights > 0
    filtered_values = filtered_values[positive_mask]
    filtered_weights = filtered_weights[positive_mask]
    if filtered_values.empty:
        raise RuntimeError(
            "Cannot compute weighted quantile with no positive-duration rows"
        )

    sorted_df = pd.DataFrame(
        {
            "value": filtered_values,
            "weight": filtered_weights,
        }
    ).sort_values("value", kind="stable")
    cumulative = sorted_df["weight"].cumsum()
    cutoff = quantile * float(sorted_df["weight"].sum())
    return float(sorted_df.loc[cumulative >= cutoff, "value"].iloc[0])


def _evaluate_request_metric_rule(
    metrics_df: pd.DataFrame,
    metric: str,
    quantile: float,
) -> float:
    if metric not in metrics_df.columns:
        raise RuntimeError(
            f"Metric column '{metric}' missing in {_REQUEST_METRICS_FILE}. "
            f"Columns: {list(metrics_df.columns)}"
        )
    return float(metrics_df[metric].quantile(quantile))


def _evaluate_worker_timeline_rule(
    worker_timeline_df: pd.DataFrame,
    metric: str,
    quantile: float,
) -> float:
    required_columns = {"worker_id", "duration_s", metric}
    missing_columns = sorted(required_columns - set(worker_timeline_df.columns))
    if missing_columns:
        raise RuntimeError(
            f"Missing columns {missing_columns} in {_WORKER_TIMELINE_FILE}. "
            f"Columns: {list(worker_timeline_df.columns)}"
        )
    if worker_timeline_df.empty:
        raise RuntimeError(f"{_WORKER_TIMELINE_FILE} is empty")

    per_worker_values = []
    for _, worker_df in worker_timeline_df.groupby("worker_id", sort=False):
        per_worker_values.append(
            _weighted_quantile(
                values=worker_df[metric],
                weights=worker_df["duration_s"],
                quantile=quantile,
            )
        )
    return max(per_worker_values)


@dataclass(frozen=True)
class TrialResult:
    qps: float
    passed: bool | None
    metric_values: dict[str, float]
    run_dir: str
    cached: bool
    timed_out: bool = False


@dataclass(frozen=True)
class JobSearchResult:
    job_name: str
    job_key: str
    max_qps_under_sla: float | None
    best_metric_values: dict[str, float] | None
    num_trials: int
    search_trace: list[dict]
    error: str | None = None


def evaluate_sla(
    metrics_df: pd.DataFrame | None,
    worker_timeline_df: pd.DataFrame | None,
    capacity_config: CapacitySearchConfig,
) -> tuple[bool | None, dict[str, float]]:
    if not capacity_config.sla_rules:
        return None, {}
    metric_values: dict[str, float] = {}
    all_passed = True
    for rule in capacity_config.sla_rules:
        if rule.metric in _WORKER_TIMELINE_METRICS:
            if worker_timeline_df is None:
                raise RuntimeError(
                    f"Missing {_WORKER_TIMELINE_FILE} for SLA metric '{rule.metric}'"
                )
            value = _evaluate_worker_timeline_rule(
                worker_timeline_df=worker_timeline_df,
                metric=rule.metric,
                quantile=rule.quantile,
            )
        else:
            if metrics_df is None:
                raise RuntimeError(
                    f"Missing {_REQUEST_METRICS_FILE} for SLA metric '{rule.metric}'"
                )
            value = _evaluate_request_metric_rule(
                metrics_df=metrics_df,
                metric=rule.metric,
                quantile=rule.quantile,
            )
        metric_key = f"{rule.metric}@q{rule.quantile}"
        metric_values[metric_key] = value
        if not _OPS[rule.op](value, rule.threshold):
            all_passed = False
    return all_passed, metric_values


class CapacitySearch:
    def __init__(
        self,
        capacity_config: CapacitySearchConfig,
        job: JobSpec,
        output_dir: Path,
        max_iterations: int,
        min_search_granularity_pct: float,
        max_qps_cap: float | None,
    ):
        self.capacity_config = capacity_config
        self.job = job
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.min_search_granularity_pct = min_search_granularity_pct
        self.max_qps_cap = max_qps_cap
        self.job_output_dir = (
            output_dir / "jobs" / f"{_sanitize_name(job.name)}_{job.key}"
        )
        self.job_output_dir.mkdir(parents=True, exist_ok=True)

    def _trial_paths(self, qps: float) -> tuple[Path, Path, Path, Path]:
        trial_dir = self.job_output_dir / "runs" / _format_qps(qps)
        metrics_file = trial_dir / _REQUEST_METRICS_FILE
        worker_timeline_file = trial_dir / _WORKER_TIMELINE_FILE
        trial_result_file = trial_dir / "trial_result.json"
        return trial_dir, metrics_file, worker_timeline_file, trial_result_file

    def _required_trial_artifacts(self) -> tuple[bool, bool]:
        needs_request_metrics = any(
            rule.metric not in _WORKER_TIMELINE_METRICS
            for rule in self.capacity_config.sla_rules
        )
        needs_worker_timeline = any(
            rule.metric in _WORKER_TIMELINE_METRICS
            for rule in self.capacity_config.sla_rules
        )
        return needs_request_metrics, needs_worker_timeline

    def _trial_succeeded(self, trial_dir: Path) -> bool:
        results_file = trial_dir / _BENCHMARK_RESULTS_FILE
        if not results_file.exists():
            return False
        try:
            with results_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False
        summary = payload.get("summary") or {}
        num_failed = summary.get("num_failed", 1)
        num_successful = summary.get("num_successful", 0)
        return num_failed == 0 and num_successful > 0

    def _write_trial_result(
        self,
        trial_result_file: Path,
        trial_result: TrialResult,
    ) -> None:
        trial_result_file.parent.mkdir(parents=True, exist_ok=True)
        with trial_result_file.open("w", encoding="utf-8") as file:
            json.dump(asdict(trial_result), file, indent=2)

    def _run_trial_subprocess(
        self,
        benchmark_config: BenchmarkConfig,
        trial_dir: Path,
        qps: float,
    ) -> bool:
        """Run one benchmark trial in a child process group.

        Returns True if the inner benchmark hit its --benchmark-timeout
        (signalled by exit code 2). Raises RuntimeError on any other
        non-zero exit. The benchmark itself is responsible for tearing
        down its server before exiting.
        """
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_pickle = trial_dir / ".trial_config.pkl"
        with config_pickle.open("wb") as file:
            pickle.dump(benchmark_config, file)

        cmd = [
            sys.executable,
            "-m",
            "sangam.benchmark.run_trial_subprocess",
            "--config-pickle",
            str(config_pickle),
        ]

        process = spawn_process_group(
            cmd=cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        try:
            returncode = process.wait()
        except BaseException:
            stop_process_group(
                process=process,
                process_name=f"Trial subprocess (QPS={qps})",
                term_timeout_seconds=TRIAL_TERM_TIMEOUT_SECONDS,
                kill_timeout_seconds=TRIAL_KILL_TIMEOUT_SECONDS,
            )
            raise

        if returncode == 2:
            logger.warning(
                f"Trial at QPS={qps} hit benchmark_timeout. "
                f"Partial results (if any) kept in {trial_dir}"
            )
            return True
        if returncode != 0:
            raise RuntimeError(
                f"Trial subprocess for QPS={qps} exited with code {returncode}"
            )
        return False

    def run_trial(self, qps: float) -> TrialResult:
        qps = round(qps, 4)
        trial_dir, metrics_file, worker_timeline_file, trial_result_file = (
            self._trial_paths(qps)
        )
        needs_request_metrics, needs_worker_timeline = self._required_trial_artifacts()
        cached = self._trial_succeeded(trial_dir)

        if not cached:
            benchmark_config = self.capacity_config.create_benchmark_config(
                job=self.job,
                qps=qps,
                output_dir=str(trial_dir),
            )

            timed_out = self._run_trial_subprocess(
                benchmark_config=benchmark_config,
                trial_dir=trial_dir,
                qps=qps,
            )
            if timed_out:
                return TrialResult(
                    qps=qps,
                    passed=None if not self.capacity_config.sla_rules else False,
                    metric_values={},
                    run_dir=str(trial_dir),
                    cached=False,
                    timed_out=True,
                )

        if needs_request_metrics and not metrics_file.exists():
            raise RuntimeError(f"Missing request metrics after trial: {metrics_file}")
        if needs_worker_timeline and not worker_timeline_file.exists():
            raise RuntimeError(
                f"Missing worker timeline after trial: {worker_timeline_file}"
            )

        metrics_df = pd.read_csv(metrics_file) if needs_request_metrics else None
        worker_timeline_df = (
            pd.read_csv(worker_timeline_file) if needs_worker_timeline else None
        )
        passed, metric_values = evaluate_sla(
            metrics_df,
            worker_timeline_df,
            self.capacity_config,
        )

        trial_result = TrialResult(
            qps=qps,
            passed=passed,
            metric_values=metric_values,
            run_dir=str(trial_dir),
            cached=cached,
            timed_out=False,
        )
        self._write_trial_result(trial_result_file, trial_result)
        return trial_result

    def _converged(self, left: float, right: float) -> bool:
        reference = max(right, left, 1.0)
        threshold = self.min_search_granularity_pct * reference / 100.0
        return abs(right - left) < threshold

    def _append_trace(self, trace: list[dict], trial_result: TrialResult) -> None:
        trace.append(
            {
                "qps": trial_result.qps,
                "passed": trial_result.passed,
                "metric_values": trial_result.metric_values,
                "run_dir": trial_result.run_dir,
                "cached": trial_result.cached,
                "timed_out": trial_result.timed_out,
            }
        )

    def run(self) -> JobSearchResult:
        if self.job.is_linear:
            return self._run_linear()
        return self._run_adaptive()

    def _finalize(
        self,
        trace: list[dict],
        max_qps_under_sla: float | None,
        best_metric_values: dict[str, float] | None,
    ) -> JobSearchResult:
        result = JobSearchResult(
            job_name=self.job.name,
            job_key=self.job.key,
            max_qps_under_sla=max_qps_under_sla,
            best_metric_values=best_metric_values,
            num_trials=len(trace),
            search_trace=trace,
        )
        job_summary_path = self.job_output_dir / "job_summary.json"
        with job_summary_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(result), file, indent=2)

        if max_qps_under_sla is None:
            logger.info(f"Job '{self.job.name}': no QPS found under SLA")
        else:
            logger.info(f"Job '{self.job.name}': max_qps_under_sla={max_qps_under_sla}")
        return result

    def _run_linear(self) -> JobSearchResult:
        assert self.job.qps_list is not None
        logger.info(
            f"Starting linear capacity search for job '{self.job.name}' "
            f"over qps_list={list(self.job.qps_list)}"
        )
        trace: list[dict] = []
        max_qps_under_sla: float | None = None
        best_metric_values: dict[str, float] | None = None

        for i, qps in enumerate(self.job.qps_list):
            trial = self.run_trial(qps)
            self._append_trace(trace, trial)

            # Check for timeout
            if trial.timed_out:
                remaining = len(self.job.qps_list) - i - 1
                logger.warning(
                    f"Capacity search terminated early due to timeout. "
                    f"Skipping {remaining} remaining trial(s)."
                )
                break

            if trial.passed is True and (
                max_qps_under_sla is None or trial.qps > max_qps_under_sla
            ):
                max_qps_under_sla = trial.qps
                best_metric_values = trial.metric_values

        return self._finalize(trace, max_qps_under_sla, best_metric_values)

    def _run_adaptive(self) -> JobSearchResult:
        assert self.job.start_qps is not None
        logger.info(f"Starting capacity search for job '{self.job.name}'")
        trace: list[dict] = []

        max_qps_under_sla: float | None = None
        best_metric_values: dict[str, float] | None = None
        min_qps_over_sla: float | None = None

        trial = self.run_trial(self.job.start_qps)
        self._append_trace(trace, trial)
        if trial.timed_out:
            logger.warning(
                "Capacity search terminated early due to timeout during initial trial."
            )
            return self._finalize(trace, max_qps_under_sla, best_metric_values)
        if trial.passed:
            max_qps_under_sla = trial.qps
            best_metric_values = trial.metric_values
        else:
            min_qps_over_sla = trial.qps

        while (
            max_qps_under_sla is not None
            and min_qps_over_sla is None
            and len(trace) < self.max_iterations
        ):
            candidate = max_qps_under_sla * 2.0
            if self.max_qps_cap is not None:
                candidate = min(candidate, self.max_qps_cap)
            candidate = round(candidate, 4)
            if candidate <= max_qps_under_sla:
                break

            trial = self.run_trial(candidate)
            self._append_trace(trace, trial)

            # Check for timeout
            if trial.timed_out:
                logger.warning(
                    "Capacity search terminated early due to timeout during exponential phase."
                )
                return self._finalize(trace, max_qps_under_sla, best_metric_values)

            if trial.passed:
                max_qps_under_sla = trial.qps
                best_metric_values = trial.metric_values
            else:
                min_qps_over_sla = trial.qps

        if (
            max_qps_under_sla is not None
            and min_qps_over_sla is None
            and self.max_qps_cap is not None
            and max_qps_under_sla >= self.max_qps_cap
        ):
            logger.info(
                f"Job '{self.job.name}': reached max_qps_cap={self.max_qps_cap}"
            )

        left = max_qps_under_sla if max_qps_under_sla is not None else 0.0
        right = min_qps_over_sla if min_qps_over_sla is not None else None

        while (
            right is not None
            and len(trace) < self.max_iterations
            and not self._converged(left, right)
        ):
            candidate = round((left + right) / 2.0, 4)
            if candidate <= left or candidate >= right:
                break
            trial = self.run_trial(candidate)
            self._append_trace(trace, trial)

            # Check for timeout
            if trial.timed_out:
                logger.warning(
                    "Capacity search terminated early due to timeout during binary search phase."
                )
                break

            if trial.passed:
                left = trial.qps
                max_qps_under_sla = trial.qps
                best_metric_values = trial.metric_values
            else:
                right = trial.qps
                min_qps_over_sla = trial.qps

        return self._finalize(trace, max_qps_under_sla, best_metric_values)
