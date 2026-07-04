import pytest

from sangam.engine.launch_config import EngineLaunchConfig
from sangam.benchmark.server_managers import SangamServerManager


def _make_manager(mode: str) -> SangamServerManager:
    return SangamServerManager(
        scheduler_address="localhost:50051",
        launch_config=EngineLaunchConfig(
            mode=mode,
            model="GSAI-ML/LLaDA-8B-Instruct",
            gpus="0,1",
            prefill_gpus="2,3",
            hybrid_colocated_gpus="1,3",
            scheduler_port=50051,
            base_worker_port=20100,
            max_batch_size=8,
            max_tokens_per_iteration=1024,
            decode_scheduler_policy="max_free_memory",
            kv_fast_pairs="",
            kv_topology_alpha=0.7,
            kv_page_size=16,
            kv_max_pages=2048,
            metrics_output_dir="./metrics_output",
            disable_metrics=False,
            enable_individual_batch_metrics=False,
        ),
        startup_timeout=120.0,
    )


def test_server_manager_uses_colocated_flags(monkeypatch) -> None:
    seen_cmd: list[str] = []
    seen_stdio: dict[str, object] = {}

    class DummyProcess:
        def poll(self):
            return None

    def _spawn(cmd, stdout, stderr):
        nonlocal seen_cmd
        seen_cmd = cmd
        seen_stdio["stdout"] = stdout
        seen_stdio["stderr"] = stderr
        return DummyProcess()

    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.spawn_process_group",
        _spawn,
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.atexit.register",
        lambda fn: None,
    )

    manager = _make_manager(mode="colocated")
    manager.start()

    assert "--mode" in seen_cmd
    assert "colocated" in seen_cmd
    assert "--gpus" in seen_cmd
    assert "0,1" in seen_cmd
    assert "--prefill-gpus" not in seen_cmd
    assert "--decode-gpus" not in seen_cmd
    assert "--max-tokens-per-iteration" in seen_cmd
    assert "--decode-scheduler-policy" in seen_cmd
    assert "max_free_memory" in seen_cmd
    assert "1024" in seen_cmd
    assert seen_stdio["stdout"] is not None
    assert seen_stdio["stderr"] is not None


def test_server_manager_uses_hybrid_flags(monkeypatch) -> None:
    seen_cmd: list[str] = []

    class DummyProcess:
        def poll(self):
            return None

    def _spawn(cmd, stdout, stderr):
        nonlocal seen_cmd
        seen_cmd = cmd
        return DummyProcess()

    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.spawn_process_group",
        _spawn,
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.atexit.register",
        lambda fn: None,
    )

    manager = _make_manager(mode="hybrid")
    manager.start()

    assert "--mode" in seen_cmd
    assert "hybrid" in seen_cmd
    assert "--prefill-gpus" in seen_cmd
    assert "2,3" in seen_cmd
    assert "--hybrid-colocated-gpus" in seen_cmd
    assert "1,3" in seen_cmd
    assert "--decode-gpus" not in seen_cmd


def test_server_manager_passes_topology_flags(monkeypatch) -> None:
    seen_cmd: list[str] = []

    class DummyProcess:
        def poll(self):
            return None

    def _spawn(cmd, stdout, stderr):
        nonlocal seen_cmd
        seen_cmd = cmd
        return DummyProcess()

    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.spawn_process_group",
        _spawn,
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.atexit.register",
        lambda fn: None,
    )

    manager = _make_manager(mode="hybrid")
    manager.launch_config.decode_scheduler_policy = "topology_guarded_memory"
    manager.launch_config.kv_fast_pairs = "0-1,2-3"
    manager.launch_config.kv_topology_alpha = 0.8
    manager.start()

    assert "--decode-scheduler-policy" in seen_cmd
    assert "topology_guarded_memory" in seen_cmd
    assert "--kv-fast-pairs" in seen_cmd
    assert "0-1,2-3" in seen_cmd
    assert "--kv-topology-alpha" in seen_cmd
    assert "0.8" in seen_cmd


def test_server_manager_uses_enable_individual_batch_metrics_flag(monkeypatch) -> None:
    seen_cmd: list[str] = []

    class DummyProcess:
        def poll(self):
            return None

    def _spawn(cmd, stdout, stderr):
        nonlocal seen_cmd
        seen_cmd = cmd
        return DummyProcess()

    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.spawn_process_group",
        _spawn,
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.atexit.register",
        lambda fn: None,
    )

    manager = _make_manager(mode="hybrid")
    manager.launch_config.enable_individual_batch_metrics = True
    manager.start()

    assert "--enable-individual-batch-metrics" in seen_cmd


def test_stop_uses_process_group_shutdown(monkeypatch) -> None:
    manager = _make_manager(mode="colocated")
    dummy_proc = type("P", (), {"poll": lambda self: None})()
    manager._process = dummy_proc

    captured: dict[str, object] = {}

    def _stop(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.stop_process_group",
        _stop,
    )
    manager.stop()

    assert captured["process"] is dummy_proc
    assert captured["process_name"] == "Server process"


def test_wait_for_ready_requires_expected_colocated_workers(monkeypatch) -> None:
    manager = _make_manager(mode="colocated")
    manager._process = type("P", (), {"poll": lambda self: None})()

    class DummyChannel:
        def close(self):
            return None

    status_responses = [
        type(
            "Status",
            (),
            {
                "num_prefill_workers": 0,
                "num_decode_workers": 0,
                "num_colocated_workers": 1,
            },
        )(),
        type(
            "Status",
            (),
            {
                "num_prefill_workers": 0,
                "num_decode_workers": 0,
                "num_colocated_workers": 2,
            },
        )(),
    ]

    class DummyStub:
        def __init__(self, _channel):
            return None

        def GetSchedulerStatus(self, _req):
            nonlocal status_responses
            return status_responses.pop(0)

    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.grpc.insecure_channel",
        lambda *args, **kwargs: DummyChannel(),
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.grpc.channel_ready_future",
        lambda _channel: type(
            "ReadyFuture", (), {"result": lambda self, timeout: None}
        )(),
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.sangam_pb2_grpc.SchedulerServiceStub",
        DummyStub,
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.time.sleep",
        lambda _: None,
    )

    manager.wait_for_ready()


def test_server_manager_counts_hybrid_workers_correctly() -> None:
    manager = _make_manager(mode="hybrid")

    assert manager._expected_prefill_workers == 2
    assert manager._expected_decode_workers == 0
    assert manager._expected_colocated_workers == 2


def test_wait_for_ready_raises_if_process_exits(monkeypatch) -> None:
    manager = _make_manager(mode="colocated")
    manager._process = type("P", (), {"poll": lambda self: 1, "returncode": 1})()
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.time.monotonic",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "sangam.benchmark.server_managers.sangam_server_manager.time.sleep",
        lambda _: None,
    )
    with pytest.raises(RuntimeError, match="before becoming ready"):
        manager.wait_for_ready()
