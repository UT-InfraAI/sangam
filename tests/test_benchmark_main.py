from datetime import datetime

from sangam.benchmark.config import BenchmarkConfig
from sangam.benchmark.main import run_benchmark


def test_run_benchmark_writes_config_and_runs_without_server(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    class DummyRunner:
        def __init__(
            self, config: BenchmarkConfig, backend=None, on_abort=None
        ) -> None:
            assert config.output_dir == str(tmp_path)
            calls.append("runner_init")

        def run(self):
            calls.append("run")
            return ["result"]

        def save_results(self, results) -> None:
            assert results == ["result"]
            calls.append("save_results")

        def export_scheduler_metrics(self) -> None:
            calls.append("export_metrics")

    class DummyServerManager:
        def __init__(self, **kwargs) -> None:
            calls.append("server_init")

        def start(self) -> None:
            calls.append("server_start")

        def wait_for_ready(self) -> None:
            calls.append("server_ready")

        def stop(self) -> None:
            calls.append("server_stop")

    monkeypatch.setattr("sangam.benchmark.main.BenchmarkRunner", DummyRunner)
    monkeypatch.setattr("sangam.benchmark.main.SangamServerManager", DummyServerManager)

    config = BenchmarkConfig(launch_server=False, output_dir=str(tmp_path))
    run_benchmark(config)

    assert (tmp_path / "config.yaml").exists()
    assert calls == ["runner_init", "run", "save_results", "export_metrics"]


def test_run_benchmark_launches_colocated_server_when_requested(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    server_kwargs: dict = {}

    class DummyRunner:
        def __init__(
            self, config: BenchmarkConfig, backend=None, on_abort=None
        ) -> None:
            calls.append("runner_init")

        def run(self):
            calls.append("run")
            return ["result"]

        def save_results(self, results) -> None:
            calls.append("save_results")

        def export_scheduler_metrics(self) -> None:
            calls.append("export_metrics")

    class DummyServerManager:
        def __init__(self, **kwargs) -> None:
            server_kwargs.update(kwargs)
            calls.append("server_init")

        def start(self) -> None:
            calls.append("server_start")

        def wait_for_ready(self) -> None:
            calls.append("server_ready")

        def stop(self) -> None:
            calls.append("server_stop")

    monkeypatch.setattr("sangam.benchmark.main.BenchmarkRunner", DummyRunner)
    monkeypatch.setattr("sangam.benchmark.main.SangamServerManager", DummyServerManager)

    config = BenchmarkConfig(
        launch_server=True,
        mode="colocated",
        gpus="0,1",
        decode_scheduler_policy="round_robin",
        output_dir=str(tmp_path),
    )
    run_benchmark(config)

    launch_config = server_kwargs["launch_config"]
    assert launch_config.mode == "colocated"
    assert launch_config.gpus == "0,1"
    assert launch_config.max_tokens_per_iteration == 4096
    assert launch_config.decode_scheduler_policy == "round_robin"
    assert launch_config.enable_individual_batch_metrics is True
    assert calls == [
        "server_init",
        "server_start",
        "server_ready",
        "runner_init",
        "run",
        "save_results",
        "export_metrics",
        "server_stop",
    ]


def test_run_benchmark_uses_timestamped_subdir_for_benchmark_output(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    output_root = tmp_path / "benchmark_output"
    expected_output_dir = output_root / "20260102_030405"

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 1, 2, 3, 4, 5)

    class DummyRunner:
        def __init__(
            self, config: BenchmarkConfig, backend=None, on_abort=None
        ) -> None:
            assert config.output_dir == str(expected_output_dir)
            calls.append("runner_init")

        def run(self):
            return []

        def save_results(self, results) -> None:
            calls.append("save_results")

        def export_scheduler_metrics(self) -> None:
            calls.append("export_metrics")

    monkeypatch.setattr("sangam.benchmark.main.datetime", _FixedDatetime)
    monkeypatch.setattr("sangam.benchmark.main.BenchmarkRunner", DummyRunner)

    config = BenchmarkConfig(
        launch_server=False,
        output_dir=str(output_root),
        export_partial_metrics=True,
    )
    run_benchmark(config)

    assert expected_output_dir.exists()
    assert (expected_output_dir / "config.yaml").exists()
    assert calls == ["runner_init", "save_results", "export_metrics"]
