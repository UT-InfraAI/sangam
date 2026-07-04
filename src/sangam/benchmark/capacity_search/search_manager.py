from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from sangam.benchmark.capacity_search.capacity_search import (
    CapacitySearch,
    JobSearchResult,
)
from sangam.benchmark.capacity_search.config import CapacitySearchConfig
from sangam.logger import init_logger

logger = init_logger(__name__)


class SearchManager:
    def __init__(
        self,
        capacity_config: CapacitySearchConfig,
        output_dir: Path,
        max_iterations_override: int | None,
        min_search_granularity_override: float | None,
        max_qps_cap_override: float | None,
    ):
        self.capacity_config = capacity_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = (
            max_iterations_override
            if max_iterations_override is not None
            else capacity_config.search.max_iterations
        )
        self.min_search_granularity_pct = (
            min_search_granularity_override
            if min_search_granularity_override is not None
            else capacity_config.search.min_search_granularity_pct
        )
        self.max_qps_cap = (
            max_qps_cap_override
            if max_qps_cap_override is not None
            else capacity_config.search.max_qps_cap
        )

    def _write_summary_csv(self, results: list[JobSearchResult]) -> None:
        summary_path = self.output_dir / "capacity_search_results.csv"
        fieldnames = [
            "job_name",
            "job_key",
            "max_qps_under_sla",
            "num_trials",
            "error",
        ]
        with summary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "job_name": result.job_name,
                        "job_key": result.job_key,
                        "max_qps_under_sla": result.max_qps_under_sla,
                        "num_trials": result.num_trials,
                        "error": result.error,
                    }
                )

    def run(self) -> list[JobSearchResult]:
        results: list[JobSearchResult] = []
        for job in self.capacity_config.jobs:
            search = CapacitySearch(
                capacity_config=self.capacity_config,
                job=job,
                output_dir=self.output_dir,
                max_iterations=self.max_iterations,
                min_search_granularity_pct=self.min_search_granularity_pct,
                max_qps_cap=self.max_qps_cap,
            )
            try:
                results.append(search.run())
            except Exception as exc:
                logger.exception(f"Job '{job.name}' failed, skipping: {exc}")
                results.append(
                    JobSearchResult(
                        job_name=job.name,
                        job_key=job.key,
                        max_qps_under_sla=None,
                        best_metric_values=None,
                        num_trials=0,
                        search_trace=[],
                        error=str(exc),
                    )
                )

        results_json_path = self.output_dir / "capacity_search_results.json"
        with results_json_path.open("w", encoding="utf-8") as file:
            json.dump([asdict(result) for result in results], file, indent=2)

        self._write_summary_csv(results)
        return results
