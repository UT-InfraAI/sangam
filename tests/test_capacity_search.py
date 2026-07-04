import json
from pathlib import Path

import pandas as pd
import pytest

from sangam.benchmark.capacity_search.capacity_search import (
    CapacitySearch,
    evaluate_sla,
)
from sangam.benchmark.capacity_search.config import (
    CapacitySearchConfig,
    _sanitize_name,
)
from sangam.benchmark.capacity_search.search_manager import SearchManager


def _patch_run_benchmark(monkeypatch: pytest.MonkeyPatch, fake_run_benchmark) -> None:
    """Patch CapacitySearch._run_trial_subprocess to invoke a fake benchmark.

    Tests pass a fake_run_benchmark(benchmark_config) that produces the
    expected CSV outputs in benchmark_config.output_dir. The patched
    method signals "ran successfully, did not time out" by returning False.
    """

    def fake_run_trial_subprocess(self, benchmark_config, trial_dir, qps) -> bool:
        fake_run_benchmark(benchmark_config)
        return False

    monkeypatch.setattr(
        CapacitySearch,
        "_run_trial_subprocess",
        fake_run_trial_subprocess,
    )


def _write_success_summary(out_dir: Path, num_successful: int = 1) -> None:
    (out_dir / "benchmark_results.json").write_text(
        json.dumps({"summary": {"num_successful": num_successful, "num_failed": 0}})
    )


def _build_config(metric: str = "request_scheduling_delay") -> CapacitySearchConfig:
    return CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "colocated",
                "gpus": "0",
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [
                {
                    "name": "job-a",
                    "start_qps": 2.0,
                    "benchmark_overrides": {"mode": "hybrid"},
                }
            ],
            "sla": [
                {
                    "metric": metric,
                    "quantile": 0.5,
                    "threshold": 2.0,
                    "op": "<=",
                }
            ],
            "search": {
                "max_iterations": 12,
                "min_search_granularity_pct": 1.0,
                "max_qps_cap": 10.0,
            },
        }
    )


def test_config_requires_sla_for_adaptive_jobs() -> None:
    with pytest.raises(
        ValueError,
        match="sla is required when any job uses start_qps",
    ):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "x", "start_qps": 1.0}],
            }
        )


def test_config_allows_missing_sla_for_linear_only_jobs() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {},
            "jobs": [{"name": "x", "qps_list": [1.0, 2.0]}],
        }
    )
    assert config.sla_rules == []


def test_job_key_ignores_qps_values() -> None:
    base = {
        "benchmark_base": {"mode": "colocated"},
        "jobs": [
            {
                "name": "x",
                "qps_list": [1.0, 2.0],
                "benchmark_overrides": {"num_requests": 8},
            }
        ],
    }
    other = {
        **base,
        "jobs": [
            {
                "name": "x",
                "qps_list": [3.0, 4.0, 5.0],
                "benchmark_overrides": {"num_requests": 8},
            }
        ],
    }
    key_a = CapacitySearchConfig.from_dict(base).jobs[0].key
    key_b = CapacitySearchConfig.from_dict(other).jobs[0].key
    assert key_a == key_b


def test_job_key_changes_with_benchmark_base() -> None:
    job = {"name": "x", "qps_list": [1.0]}
    key_a = (
        CapacitySearchConfig.from_dict(
            {"benchmark_base": {"mode": "colocated"}, "jobs": [job]}
        )
        .jobs[0]
        .key
    )
    key_b = (
        CapacitySearchConfig.from_dict(
            {"benchmark_base": {"mode": "hybrid"}, "jobs": [job]}
        )
        .jobs[0]
        .key
    )
    assert key_a != key_b


def test_config_rejects_unknown_benchmark_keys() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {"unknown_field": 123},
                "jobs": [{"name": "x", "start_qps": 1.0}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_config_rejects_removed_benchmark_time_limit_key() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {"time_limit": 123.0},
                "jobs": [{"name": "x", "start_qps": 1.0}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_config_rejects_metrics_output_dir_key() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {"metrics_output_dir": "server_metrics"},
                "jobs": [{"name": "x", "start_qps": 1.0}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_config_accepts_shared_engine_launch_keys() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "hybrid",
                "prefill_gpus": "0",
                "hybrid_colocated_gpus": "1",
                "master_addr": "127.0.0.1",
                "master_port": 29501,
                "max_prefill_tokens_per_batch": 2048,
                "prefill_scheduler_policy": "round_robin",
                "prefill_queue_policy": "fewest_remaining_blocks",
                "enable_operation_metrics": True,
                "op_metrics_layer_id": 10,
                "export_partial_metrics": True,
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [{"name": "x", "start_qps": 1.0}],
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 1.0,
                    "op": "<=",
                }
            ],
        }
    )

    benchmark_config = config.create_benchmark_config(
        job=config.jobs[0],
        qps=1.0,
        output_dir="out",
    )

    assert benchmark_config.master_addr == "127.0.0.1"
    assert benchmark_config.master_port == 29501
    assert benchmark_config.max_prefill_tokens_per_batch == 2048
    assert benchmark_config.prefill_scheduler_policy == "round_robin"
    assert benchmark_config.prefill_queue_policy == "fewest_remaining_blocks"
    assert benchmark_config.enable_operation_metrics is True
    assert benchmark_config.op_metrics_layer_id == 10
    assert benchmark_config.export_partial_metrics is True


def test_config_accepts_and_forwards_hybrid_prefill_overflow() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "hybrid",
                "prefill_gpus": "0",
                "hybrid_colocated_gpus": "1",
                "enable_hybrid_prefill_overflow": True,
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [{"name": "x", "start_qps": 1.0}],
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 1.0,
                    "op": "<=",
                }
            ],
        }
    )

    benchmark_config = config.create_benchmark_config(
        job=config.jobs[0],
        qps=1.0,
        output_dir="out",
    )
    launch_config = benchmark_config.build_engine_launch_config(
        metrics_output_dir="out"
    )

    assert benchmark_config.enable_hybrid_prefill_overflow is True
    assert launch_config.enable_hybrid_prefill_overflow is True


def test_evaluate_sla_pass_and_fail() -> None:
    config = _build_config()
    metrics_df = pd.DataFrame({"request_scheduling_delay": [0.1, 0.3, 0.5]})
    passed, metric_values = evaluate_sla(metrics_df, None, config)
    assert passed
    assert metric_values["request_scheduling_delay@q0.5"] == pytest.approx(0.3)

    bad_df = pd.DataFrame({"request_scheduling_delay": [3.0, 4.0, 5.0]})
    passed, _ = evaluate_sla(bad_df, None, config)
    assert not passed


def test_evaluate_sla_raises_for_missing_column() -> None:
    config = _build_config()
    metrics_df = pd.DataFrame({"request_decode_time": [0.1, 0.2]})
    with pytest.raises(RuntimeError, match="missing in request_metrics.csv"):
        evaluate_sla(metrics_df, None, config)


def test_evaluate_sla_uses_time_weighted_worst_worker_queue_depth() -> None:
    config = _build_config(metric="queue_depth_waiting")
    worker_timeline_df = pd.DataFrame(
        {
            "worker_id": ["decode-0", "decode-0", "decode-1", "decode-1"],
            "duration_s": [9.0, 1.0, 8.0, 2.0],
            "queue_depth_waiting": [5, 1, 1, 6],
        }
    )

    passed, metric_values = evaluate_sla(None, worker_timeline_df, config)

    assert not passed
    assert metric_values["queue_depth_waiting@q0.5"] == pytest.approx(5.0)


def test_evaluate_sla_allows_mixed_request_and_queue_rules() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "colocated",
                "gpus": "0",
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [{"name": "job-a", "start_qps": 2.0}],
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 1.0,
                    "op": "<=",
                },
                {
                    "metric": "queue_depth_waiting",
                    "quantile": 0.95,
                    "threshold": 4.0,
                    "op": "<=",
                },
            ],
        }
    )

    metrics_df = pd.DataFrame({"request_scheduling_delay": [0.2, 0.4, 0.6]})
    worker_timeline_df = pd.DataFrame(
        {
            "worker_id": ["decode-0", "decode-0", "decode-1"],
            "duration_s": [1.0, 9.0, 10.0],
            "queue_depth_waiting": [1, 3, 2],
        }
    )

    passed, metric_values = evaluate_sla(metrics_df, worker_timeline_df, config)

    assert passed
    assert metric_values["request_scheduling_delay@q0.5"] == pytest.approx(0.4)
    assert metric_values["queue_depth_waiting@q0.95"] == pytest.approx(3.0)


def test_evaluate_sla_raises_for_missing_worker_timeline_columns() -> None:
    config = _build_config(metric="queue_depth_waiting")
    worker_timeline_df = pd.DataFrame(
        {
            "worker_id": ["decode-0"],
            "duration_s": [1.0],
        }
    )

    with pytest.raises(RuntimeError, match="Missing columns"):
        evaluate_sla(None, worker_timeline_df, config)


def test_run_trial_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _build_config()
    job = config.jobs[0]
    calls = {"count": 0}

    def fake_run_benchmark(benchmark_config) -> None:
        calls["count"] += 1
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"request_scheduling_delay": [0.2, 0.3, 0.4]}).to_csv(
            out_dir / "request_metrics.csv", index=False
        )
        _write_success_summary(out_dir)

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    search = CapacitySearch(
        capacity_config=config,
        job=job,
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )

    first = search.run_trial(2.0)
    second = search.run_trial(2.0)

    assert not first.cached
    assert second.cached
    assert calls["count"] == 1


def test_run_trial_recomputes_sla_without_rerunning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing_config = _build_config()
    failing_config = CapacitySearchConfig.from_dict(
        {
            **{
                "benchmark_base": passing_config.benchmark_base,
                "jobs": [
                    {
                        "name": passing_config.jobs[0].name,
                        "start_qps": passing_config.jobs[0].start_qps,
                        "benchmark_overrides": passing_config.jobs[
                            0
                        ].benchmark_overrides,
                    }
                ],
                "search": {
                    "max_iterations": passing_config.search.max_iterations,
                    "min_search_granularity_pct": passing_config.search.min_search_granularity_pct,
                    "max_qps_cap": passing_config.search.max_qps_cap,
                },
            },
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 0.25,
                    "op": "<=",
                }
            ],
        }
    )
    calls = {"count": 0}

    def fake_run_benchmark(benchmark_config) -> None:
        calls["count"] += 1
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"request_scheduling_delay": [0.2, 0.3, 0.4]}).to_csv(
            out_dir / "request_metrics.csv", index=False
        )
        _write_success_summary(out_dir)

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    first_search = CapacitySearch(
        capacity_config=passing_config,
        job=passing_config.jobs[0],
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )
    second_search = CapacitySearch(
        capacity_config=failing_config,
        job=failing_config.jobs[0],
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )

    first = first_search.run_trial(2.0)
    second = second_search.run_trial(2.0)

    assert not first.cached
    assert first.passed
    assert second.cached
    assert not second.passed
    assert calls["count"] == 1

    trial_result_path = (
        tmp_path
        / "jobs"
        / f"{_sanitize_name(passing_config.jobs[0].name)}_{passing_config.jobs[0].key}"
        / "runs"
        / "2"
        / "trial_result.json"
    )
    payload = json.loads(trial_result_path.read_text())
    assert not payload["passed"]


def test_search_manager_runs_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config()

    def fake_run_benchmark(benchmark_config) -> None:
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qps = float(benchmark_config.qps)
        if qps <= 4.0:
            values = [0.2, 0.3, 0.4]
        else:
            values = [3.0, 4.0, 5.0]
        pd.DataFrame({"request_scheduling_delay": values}).to_csv(
            out_dir / "request_metrics.csv",
            index=False,
        )

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    manager = SearchManager(
        capacity_config=config,
        output_dir=tmp_path,
        max_iterations_override=None,
        min_search_granularity_override=None,
        max_qps_cap_override=None,
    )
    results = manager.run()

    assert len(results) == 1
    assert results[0].job_name == "job-a"
    assert results[0].max_qps_under_sla == pytest.approx(4.0, abs=1e-6)
    assert (tmp_path / "capacity_search_results.json").exists()
    assert (tmp_path / "capacity_search_results.csv").exists()

    payload = json.loads((tmp_path / "capacity_search_results.json").read_text())
    assert payload[0]["job_name"] == "job-a"


def test_run_trial_raises_for_missing_worker_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(metric="queue_depth_waiting")
    job = config.jobs[0]

    def fake_run_benchmark(benchmark_config) -> None:
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    search = CapacitySearch(
        capacity_config=config,
        job=job,
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )

    with pytest.raises(RuntimeError, match="Missing worker timeline after trial"):
        search.run_trial(2.0)


def test_queue_sla_reuses_worker_timeline_without_rerunning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing_config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "colocated",
                "gpus": "0",
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [
                {
                    "name": "job-a",
                    "start_qps": 2.0,
                    "benchmark_overrides": {"mode": "hybrid"},
                }
            ],
            "sla": [
                {
                    "metric": "queue_depth_waiting",
                    "quantile": 0.5,
                    "threshold": 6.0,
                    "op": "<=",
                }
            ],
            "search": {
                "max_iterations": 12,
                "min_search_granularity_pct": 1.0,
                "max_qps_cap": 10.0,
            },
        }
    )
    failing_config = CapacitySearchConfig.from_dict(
        {
            **{
                "benchmark_base": passing_config.benchmark_base,
                "jobs": [
                    {
                        "name": passing_config.jobs[0].name,
                        "start_qps": passing_config.jobs[0].start_qps,
                        "benchmark_overrides": passing_config.jobs[
                            0
                        ].benchmark_overrides,
                    }
                ],
                "search": {
                    "max_iterations": passing_config.search.max_iterations,
                    "min_search_granularity_pct": passing_config.search.min_search_granularity_pct,
                    "max_qps_cap": passing_config.search.max_qps_cap,
                },
            },
            "sla": [
                {
                    "metric": "queue_depth_waiting",
                    "quantile": 0.5,
                    "threshold": 4.0,
                    "op": "<=",
                }
            ],
        }
    )
    calls = {"count": 0}

    def fake_run_benchmark(benchmark_config) -> None:
        calls["count"] += 1
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"worker_id": "decode-0", "duration_s": 9.0, "queue_depth_waiting": 5},
                {"worker_id": "decode-0", "duration_s": 1.0, "queue_depth_waiting": 1},
            ]
        ).to_csv(out_dir / "worker_timeline.csv", index=False)
        _write_success_summary(out_dir)

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    first_search = CapacitySearch(
        capacity_config=passing_config,
        job=passing_config.jobs[0],
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )
    second_search = CapacitySearch(
        capacity_config=failing_config,
        job=failing_config.jobs[0],
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )

    first = first_search.run_trial(2.0)
    second = second_search.run_trial(2.0)

    assert not first.cached
    assert first.passed
    assert second.cached
    assert not second.passed
    assert calls["count"] == 1


def test_job_rejects_both_start_qps_and_qps_list() -> None:
    with pytest.raises(ValueError, match="exactly one of start_qps or qps_list"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "j", "start_qps": 1.0, "qps_list": [1.0, 2.0]}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_job_rejects_neither_start_qps_nor_qps_list() -> None:
    with pytest.raises(
        ValueError,
        match=r"must set start_qps, qps_list, or search\.qps_list",
    ):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "j"}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_job_inherits_search_qps_list() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {},
            "jobs": [
                {"name": "inherits"},
                {"name": "explicit", "qps_list": [4.0, 6.0]},
            ],
            "search": {"qps_list": [1.0, 2.0, 3.0]},
        }
    )
    inherits = config.jobs[0]
    explicit = config.jobs[1]

    assert inherits.is_linear
    assert inherits.qps_list == (1.0, 2.0, 3.0)
    assert explicit.qps_list == (4.0, 6.0)
    assert config.search.qps_list == (1.0, 2.0, 3.0)


def test_search_qps_list_does_not_override_start_qps() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {},
            "jobs": [{"name": "adaptive", "start_qps": 2.0}],
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 1.0,
                    "op": "<=",
                }
            ],
            "search": {"qps_list": [1.0, 2.0]},
        }
    )
    job = config.jobs[0]
    assert not job.is_linear
    assert job.start_qps == 2.0
    assert job.qps_list is None


def test_search_qps_list_inheritance_keeps_sla_optional() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {},
            "jobs": [{"name": "a"}, {"name": "b"}],
            "search": {"qps_list": [1.0, 2.0]},
        }
    )
    assert config.sla_rules == []
    assert all(job.qps_list == (1.0, 2.0) for job in config.jobs)


def test_search_qps_list_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match=r"search\.qps_list must be a non-empty list"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "j", "start_qps": 1.0}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
                "search": {"qps_list": []},
            }
        )


def test_search_qps_list_rejects_non_positive_entry() -> None:
    with pytest.raises(ValueError, match=r"search\.qps_list entries must be > 0"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "j", "start_qps": 1.0}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
                "search": {"qps_list": [1.0, 0.0, 2.0]},
            }
        )


def test_job_rejects_empty_qps_list() -> None:
    with pytest.raises(ValueError, match="qps_list must be a non-empty list"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "j", "qps_list": []}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_job_rejects_non_positive_qps_list_entry() -> None:
    with pytest.raises(ValueError, match="qps_list entries must be > 0"):
        CapacitySearchConfig.from_dict(
            {
                "benchmark_base": {},
                "jobs": [{"name": "j", "qps_list": [1.0, 0.0, 2.0]}],
                "sla": [
                    {
                        "metric": "request_scheduling_delay",
                        "quantile": 0.5,
                        "threshold": 1.0,
                        "op": "<=",
                    }
                ],
            }
        )


def test_job_accepts_qps_list_only() -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {},
            "jobs": [{"name": "j", "qps_list": [1.0, 2.5, 4.0]}],
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 1.0,
                    "op": "<=",
                }
            ],
        }
    )
    job = config.jobs[0]
    assert job.is_linear
    assert job.start_qps is None
    assert job.qps_list == (1.0, 2.5, 4.0)


def test_linear_search_runs_all_qps_and_picks_largest_passing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "colocated",
                "gpus": "0",
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [
                {
                    "name": "linear-job",
                    "qps_list": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            "sla": [
                {
                    "metric": "request_scheduling_delay",
                    "quantile": 0.5,
                    "threshold": 0.35,
                    "op": "<=",
                }
            ],
        }
    )

    seen_qps: list[float] = []

    def fake_run_benchmark(benchmark_config) -> None:
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qps = float(benchmark_config.qps)
        seen_qps.append(qps)
        # Pass at qps 1.0 and 3.0; fail at 2.0 and 4.0 (non-monotone on purpose
        # to verify SLA does not gate iteration order).
        if qps in (1.0, 3.0):
            values = [0.1, 0.2, 0.3]
        else:
            values = [0.5, 0.6, 0.7]
        pd.DataFrame({"request_scheduling_delay": values}).to_csv(
            out_dir / "request_metrics.csv",
            index=False,
        )

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    search = CapacitySearch(
        capacity_config=config,
        job=config.jobs[0],
        output_dir=tmp_path,
        max_iterations=12,
        min_search_granularity_pct=1.0,
        max_qps_cap=None,
    )
    result = search.run()

    assert seen_qps == [1.0, 2.0, 3.0, 4.0]
    assert result.num_trials == 4
    assert [entry["qps"] for entry in result.search_trace] == [1.0, 2.0, 3.0, 4.0]
    assert result.max_qps_under_sla == pytest.approx(3.0)


def test_linear_search_without_sla_runs_all_qps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = CapacitySearchConfig.from_dict(
        {
            "benchmark_base": {
                "mode": "colocated",
                "gpus": "0",
                "num_requests": 8,
                "length_type": "fixed",
                "prefill_tokens": 64,
                "decode_tokens": 16,
                "interval_type": "poisson",
            },
            "jobs": [
                {
                    "name": "linear-job",
                    "qps_list": [1.0, 2.0, 3.0],
                }
            ],
        }
    )

    seen_qps: list[float] = []

    def fake_run_benchmark(benchmark_config) -> None:
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        seen_qps.append(float(benchmark_config.qps))

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    search = CapacitySearch(
        capacity_config=config,
        job=config.jobs[0],
        output_dir=tmp_path,
        max_iterations=12,
        min_search_granularity_pct=1.0,
        max_qps_cap=None,
    )
    result = search.run()

    assert seen_qps == [1.0, 2.0, 3.0]
    assert result.num_trials == 3
    assert result.max_qps_under_sla is None
    assert result.best_metric_values is None
    assert all(entry["passed"] is None for entry in result.search_trace)
    assert all(entry["metric_values"] == {} for entry in result.search_trace)


def test_search_manager_runs_with_queue_sla(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(metric="queue_depth_waiting")

    def fake_run_benchmark(benchmark_config) -> None:
        out_dir = Path(benchmark_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qps = float(benchmark_config.qps)
        if qps <= 4.0:
            rows = [
                {"worker_id": "decode-0", "duration_s": 8.0, "queue_depth_waiting": 1},
                {"worker_id": "decode-0", "duration_s": 2.0, "queue_depth_waiting": 3},
                {"worker_id": "decode-1", "duration_s": 10.0, "queue_depth_waiting": 2},
            ]
        else:
            rows = [
                {"worker_id": "decode-0", "duration_s": 8.0, "queue_depth_waiting": 5},
                {"worker_id": "decode-0", "duration_s": 2.0, "queue_depth_waiting": 1},
                {"worker_id": "decode-1", "duration_s": 10.0, "queue_depth_waiting": 2},
            ]
        pd.DataFrame(rows).to_csv(out_dir / "worker_timeline.csv", index=False)

    _patch_run_benchmark(monkeypatch, fake_run_benchmark)

    manager = SearchManager(
        capacity_config=config,
        output_dir=tmp_path,
        max_iterations_override=None,
        min_search_granularity_override=None,
        max_qps_cap_override=None,
    )
    results = manager.run()

    assert len(results) == 1
    assert results[0].max_qps_under_sla == pytest.approx(4.0, abs=1e-6)


def test_run_trial_subprocess_timeout_returns_timed_out_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config()

    def fake_run_trial_subprocess(self, benchmark_config, trial_dir, qps) -> bool:
        return True

    monkeypatch.setattr(
        CapacitySearch,
        "_run_trial_subprocess",
        fake_run_trial_subprocess,
    )

    search = CapacitySearch(
        capacity_config=config,
        job=config.jobs[0],
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )

    trial = search.run_trial(2.0)

    assert trial.timed_out
    assert trial.passed is False
    assert trial.metric_values == {}


def test_run_trial_subprocess_treats_returncode_two_as_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The benchmark subprocess signals --benchmark-timeout via exit code 2."""

    from sangam.benchmark.capacity_search import capacity_search as cs_module

    config = _build_config()

    class _FakeProc:
        def wait(self) -> int:
            return 2

    monkeypatch.setattr(
        cs_module,
        "spawn_process_group",
        lambda *, cmd, stdout, stderr, env=None: _FakeProc(),
    )

    search = CapacitySearch(
        capacity_config=config,
        job=config.jobs[0],
        output_dir=tmp_path,
        max_iterations=5,
        min_search_granularity_pct=1.0,
        max_qps_cap=10.0,
    )

    trial = search.run_trial(2.0)

    assert trial.timed_out
    assert trial.passed is False
    assert trial.metric_values == {}


def test_capacity_search_does_not_import_run_benchmark() -> None:
    """run_benchmark must run in a subprocess, not in the controller's
    process. The controller would otherwise leak server subprocesses on
    timeout (the daemon-thread bug)."""

    import sangam.benchmark.capacity_search.capacity_search as cs_module

    assert not hasattr(cs_module, "run_benchmark")
