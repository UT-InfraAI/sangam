from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import yaml

from sangam.benchmark.config import BenchmarkConfig

_SUPPORTED_OPS = {"<=", "<", ">=", ">", "=="}
_FORBIDDEN_CAPACITY_SEARCH_BENCHMARK_KEYS = {"launch_server"}


def _validate_benchmark_keys(values: dict[str, Any], context: str) -> None:
    valid_fields = {
        field.name for field in dataclass_fields(BenchmarkConfig) if field.init
    }
    unknown = sorted(set(values.keys()) - valid_fields)
    if unknown:
        raise ValueError(f"{context} has unsupported keys: {unknown}")


def _validate_capacity_search_benchmark_keys(
    values: dict[str, Any],
    context: str,
) -> None:
    forbidden = sorted(set(values.keys()) & _FORBIDDEN_CAPACITY_SEARCH_BENCHMARK_KEYS)
    if forbidden:
        raise ValueError(f"{context} does not support keys: {forbidden}")


def _sanitize_name(name: str) -> str:
    """Sanitize job name for use as directory name."""
    import re

    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Collapse consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Truncate to 50 characters
    sanitized = sanitized[:50]
    # Strip trailing underscores again in case truncation exposed them
    sanitized = sanitized.rstrip("_")
    # Handle empty result
    return sanitized if sanitized else "job"


def _stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class SlaRule:
    metric: str
    quantile: float
    threshold: float
    op: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlaRule:
        required = {"metric", "quantile", "threshold", "op"}
        missing = sorted(required - set(data.keys()))
        if missing:
            raise ValueError(f"SLA rule missing required keys: {missing}")
        op = str(data["op"])
        if op not in _SUPPORTED_OPS:
            raise ValueError(f"Unsupported SLA operator: {op}")
        quantile = float(data["quantile"])
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError(f"SLA quantile must be in [0, 1], got {quantile}")
        return cls(
            metric=str(data["metric"]),
            quantile=quantile,
            threshold=float(data["threshold"]),
            op=op,
        )


@dataclass(frozen=True)
class SearchSettings:
    max_iterations: int
    min_search_granularity_pct: float
    max_qps_cap: float | None
    qps_list: tuple[float, ...] | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchSettings:
        max_qps_cap = data.get("max_qps_cap")
        if max_qps_cap is not None:
            max_qps_cap = float(max_qps_cap)
            if max_qps_cap <= 0:
                raise ValueError("search.max_qps_cap must be > 0")

        max_iterations = int(data.get("max_iterations", 20))
        if max_iterations <= 0:
            raise ValueError("search.max_iterations must be > 0")

        min_granularity = float(data.get("min_search_granularity_pct", 2.5))
        if min_granularity <= 0:
            raise ValueError("search.min_search_granularity_pct must be > 0")

        qps_list: tuple[float, ...] | None = None
        raw_qps_list = data.get("qps_list")
        if raw_qps_list is not None:
            if not isinstance(raw_qps_list, list) or not raw_qps_list:
                raise ValueError("search.qps_list must be a non-empty list")
            coerced: list[float] = []
            for entry in raw_qps_list:
                value = float(entry)
                if value <= 0:
                    raise ValueError(
                        f"search.qps_list entries must be > 0, got {value}"
                    )
                coerced.append(value)
            qps_list = tuple(coerced)

        return cls(
            max_iterations=max_iterations,
            min_search_granularity_pct=min_granularity,
            max_qps_cap=max_qps_cap,
            qps_list=qps_list,
        )


@dataclass(frozen=True)
class JobSpec:
    name: str
    start_qps: float | None
    qps_list: tuple[float, ...] | None
    benchmark_overrides: dict[str, Any]
    key: str

    @property
    def is_linear(self) -> bool:
        return self.qps_list is not None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        benchmark_base: dict[str, Any],
        default_qps_list: tuple[float, ...] | None,
    ) -> JobSpec:
        if "name" not in data:
            raise ValueError("job is missing required key: name")
        name = str(data["name"])

        has_start_qps = "start_qps" in data and data["start_qps"] is not None
        has_qps_list = "qps_list" in data and data["qps_list"] is not None
        if has_start_qps and has_qps_list:
            raise ValueError(
                f"job {name}: set exactly one of start_qps or qps_list, not both"
            )
        if not has_start_qps and not has_qps_list and default_qps_list is None:
            raise ValueError(
                f"job {name}: must set start_qps, qps_list, or "
                "search.qps_list at the top level"
            )

        start_qps: float | None = None
        qps_list: tuple[float, ...] | None = None
        if has_start_qps:
            start_qps = float(data["start_qps"])
            if start_qps <= 0:
                raise ValueError(f"job {name}: start_qps must be > 0")
        elif has_qps_list:
            raw_list = data["qps_list"]
            if not isinstance(raw_list, list) or not raw_list:
                raise ValueError(f"job {name}: qps_list must be a non-empty list")
            coerced: list[float] = []
            for entry in raw_list:
                value = float(entry)
                if value <= 0:
                    raise ValueError(
                        f"job {name}: qps_list entries must be > 0, got {value}"
                    )
                coerced.append(value)
            qps_list = tuple(coerced)
        else:
            qps_list = default_qps_list

        overrides = data.get("benchmark_overrides")
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"job {name}: benchmark_overrides must be a mapping")
        _validate_benchmark_keys(overrides, f"job {name}.benchmark_overrides")
        _validate_capacity_search_benchmark_keys(
            overrides,
            f"job {name}.benchmark_overrides",
        )

        key = _stable_hash(
            {
                "name": name,
                "benchmark_base": benchmark_base,
                "benchmark_overrides": overrides,
            }
        )
        return cls(
            name=name,
            start_qps=start_qps,
            qps_list=qps_list,
            benchmark_overrides=overrides,
            key=key,
        )


@dataclass(frozen=True)
class CapacitySearchConfig:
    benchmark_base: dict[str, Any]
    jobs: list[JobSpec]
    sla_rules: list[SlaRule]
    search: SearchSettings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapacitySearchConfig:
        if not isinstance(data, dict):
            raise ValueError("Capacity search config must be a mapping")

        if "benchmark_base" not in data:
            raise ValueError("Config missing required top-level key: benchmark_base")
        if "jobs" not in data:
            raise ValueError("Config missing required top-level key: jobs")

        benchmark_base = data["benchmark_base"]
        if not isinstance(benchmark_base, dict):
            raise ValueError("benchmark_base must be a mapping")
        _validate_benchmark_keys(benchmark_base, "benchmark_base")
        _validate_capacity_search_benchmark_keys(benchmark_base, "benchmark_base")

        search = SearchSettings.from_dict(data.get("search", {}))

        jobs_raw = data["jobs"]
        if not isinstance(jobs_raw, list) or not jobs_raw:
            raise ValueError("jobs must be a non-empty list")
        jobs = [
            JobSpec.from_dict(
                item,
                benchmark_base=benchmark_base,
                default_qps_list=search.qps_list,
            )
            for item in jobs_raw
        ]

        sla_rules: list[SlaRule] = []
        if "sla" in data:
            sla_raw = data["sla"]
            if not isinstance(sla_raw, list) or not sla_raw:
                raise ValueError("sla must be a non-empty list")
            sla_rules = [SlaRule.from_dict(item) for item in sla_raw]

        if not sla_rules and any(not job.is_linear for job in jobs):
            raise ValueError(
                "sla is required when any job uses start_qps (adaptive search)"
            )

        return cls(
            benchmark_base=benchmark_base,
            jobs=jobs,
            sla_rules=sla_rules,
            search=search,
        )

    @classmethod
    def from_yaml_file(cls, config_path: Path) -> CapacitySearchConfig:
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        return cls.from_dict(raw)

    def create_benchmark_config(
        self,
        job: JobSpec,
        qps: float,
        output_dir: str,
    ) -> BenchmarkConfig:
        merged = {
            **self.benchmark_base,
            **job.benchmark_overrides,
            "qps": qps,
            "output_dir": output_dir,
        }
        return BenchmarkConfig(**merged)
