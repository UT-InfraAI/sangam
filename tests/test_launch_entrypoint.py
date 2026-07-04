from datetime import datetime

import pytest

from sangam.engine.launch_config import EngineLaunchConfig
from sangam.engine.scheduler_config import (
    ColocatedSchedulerConfig,
    HybridSchedulerConfig,
)
from sangam.entrypoints import launch
from sangam.worker.worker_config import (
    ColocatedWorkerConfig,
    PrefillWorkerConfig,
)


class _DummyProcess:
    def __init__(self, target, args, name, daemon) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.pid = 1000
        self.started = False
        self.terminated = False
        self.killed = False
        self.exitcode = 0

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def is_alive(self) -> bool:
        return False


def _base_config(mode: str) -> EngineLaunchConfig:
    return EngineLaunchConfig(
        mode=mode,
        model="GSAI-ML/LLaDA-8B-Instruct",
        scheduler_port=50051,
        gpus="0,1",
        prefill_gpus="0",
        hybrid_colocated_gpus="1",
        base_worker_port=20100,
        master_addr="localhost",
        master_port=29500,
        max_batch_size=8,
        max_tokens_per_iteration=1024,
        max_prefill_tokens_per_batch=4096,
        kv_page_size=16,
        kv_max_pages=2048,
        metrics_output_dir="./metrics_output",
        disable_metrics=False,
        enable_individual_batch_metrics=False,
        export_partial_metrics=False,
        enable_operation_metrics=False,
        op_metrics_layer_id=None,
        prefill_scheduler_policy="round_robin",
        decode_grouping_slack_ratio=0.10,
        prefill_queue_policy="arrival_order",
        decode_scheduler_policy="max_free_memory",
        kv_fast_pairs="",
        kv_topology_alpha=0.7,
        prefill_overload_threshold=4,
        enable_hybrid_prefill_overflow=False,
        block_length=32,
        mask_id=126336,
        max_gen_len=None,
        colocated_sticky_worker=False,
    )


def _scheduler_config(proc: _DummyProcess):
    """Each scheduler mp.Process is launched with args=(port, scheduler_config)."""
    return proc.args[1]


def _worker_config(proc: _DummyProcess):
    """Each worker mp.Process is launched with args=(worker_config,)."""
    return proc.args[0]


def test_main_launches_colocated_processes(monkeypatch) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    monkeypatch.setattr(launch, "parse_args", lambda: _base_config("colocated"))
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    assert [p.name for p in created] == [
        "colocated-scheduler",
        "colocated-0",
        "colocated-1",
    ]
    assert all(p.started for p in created)
    sched_cfg = _scheduler_config(created[0])
    assert isinstance(sched_cfg, ColocatedSchedulerConfig)
    assert sched_cfg.prefill_scheduler_policy == "round_robin"
    assert sched_cfg.decode_grouping_slack_ratio == 0.10
    assert sched_cfg.block_length == 32
    assert sched_cfg.mask_id == 126336
    assert sched_cfg.max_gen_len is None
    assert sched_cfg.colocated_sticky_worker is False

    colocated_proc = next(proc for proc in created if proc.name == "colocated-0")
    worker_cfg = _worker_config(colocated_proc)
    assert isinstance(worker_cfg, ColocatedWorkerConfig)
    assert worker_cfg.kv_page_size == 16
    assert worker_cfg.kv_max_pages == 2048
    assert worker_cfg.prefill_queue_policy == "arrival_order"
    assert worker_cfg.enable_metrics is True
    assert worker_cfg.enable_operation_metrics is False
    assert worker_cfg.op_metrics_layer_id is None


def test_main_passes_enable_individual_batch_metrics_to_scheduler(monkeypatch) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("colocated")
    cfg.enable_individual_batch_metrics = True
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    assert created[0].name == "colocated-scheduler"
    sched_cfg = _scheduler_config(created[0])
    assert sched_cfg.enable_individual_batch_metrics is True
    assert sched_cfg.prefill_scheduler_policy == "round_robin"
    assert sched_cfg.block_length == 32


def test_main_rejects_unsupported_mode(monkeypatch) -> None:
    cfg = _base_config("colocated")
    cfg.mode = "disaggregated"
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    with pytest.raises(ValueError, match="Unsupported serving mode"):
        launch.main()


def test_terminate_processes_sends_sigterm_first(monkeypatch) -> None:
    class _Proc:
        def __init__(self) -> None:
            self.name = "scheduler"
            self.pid = 2000
            self.terminated = False
            self.killed = False
            self.join_timeouts: list[float] = []
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout=None) -> None:
            self.join_timeouts.append(timeout)
            if self.terminated:
                self._alive = False

        def kill(self) -> None:
            self.killed = True
            self._alive = False

    proc = _Proc()
    cfg = launch.EngineLaunchConfig()
    launch._terminate_processes([proc], cfg)

    assert proc.terminated is True
    assert proc.killed is False
    assert proc.join_timeouts == [cfg.term_timeout_seconds]


def test_parse_args_defaults_decode_policy_to_max_free_memory(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py"])

    cfg = launch.parse_args()

    assert cfg.decode_scheduler_policy == "max_free_memory"


def test_parse_args_defaults_mode_to_colocated(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py"])

    cfg = launch.parse_args()

    assert cfg.mode == "colocated"


def test_parse_args_defaults_prefill_queue_policy_to_arrival_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py"])

    cfg = launch.parse_args()

    assert cfg.prefill_queue_policy == "arrival_order"


def test_parse_args_accepts_max_free_memory_decode_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--decode-scheduler-policy", "max_free_memory"],
    )

    cfg = launch.parse_args()

    assert cfg.decode_scheduler_policy == "max_free_memory"


def test_parse_args_accepts_prefill_queue_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--prefill-queue-policy", "fewest_remaining_blocks"],
    )

    cfg = launch.parse_args()

    assert cfg.prefill_queue_policy == "fewest_remaining_blocks"


def test_parse_args_accepts_least_outstanding_requests_in_colocated_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "colocated",
            "--prefill-scheduler-policy",
            "least_outstanding_requests",
        ],
    )

    cfg = launch.parse_args()

    assert cfg.prefill_scheduler_policy == "least_outstanding_requests"


def test_parse_args_accepts_least_request_length_sum_in_colocated_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "colocated",
            "--prefill-scheduler-policy",
            "least_request_length_sum",
        ],
    )

    cfg = launch.parse_args()

    assert cfg.prefill_scheduler_policy == "least_request_length_sum"


def test_parse_args_accepts_balanced_length_clustering_in_colocated_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "colocated",
            "--prefill-scheduler-policy",
            "balanced_length_clustering",
        ],
    )

    cfg = launch.parse_args()

    assert cfg.prefill_scheduler_policy == "balanced_length_clustering"


def test_parse_args_accepts_decode_grouping_slack_ratio(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--decode-grouping-slack-ratio", "0.25"],
    )

    cfg = launch.parse_args()

    assert cfg.decode_grouping_slack_ratio == 0.25


def test_parse_args_rejects_least_request_length_sum_outside_colocated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "hybrid",
            "--prefill-scheduler-policy",
            "least_request_length_sum",
        ],
    )

    with pytest.raises(
        ValueError,
        match="--prefill-scheduler-policy=least_request_length_sum is supported only in colocated mode",
    ):
        launch.parse_args()


def test_parse_args_rejects_balanced_length_clustering_outside_colocated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "hybrid",
            "--prefill-scheduler-policy",
            "balanced_length_clustering",
        ],
    )

    with pytest.raises(
        ValueError,
        match="--prefill-scheduler-policy=balanced_length_clustering is supported only in colocated mode",
    ):
        launch.parse_args()


def test_parse_args_rejects_negative_decode_grouping_slack_ratio(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--decode-grouping-slack-ratio", "-0.1"],
    )

    with pytest.raises(
        ValueError,
        match="--decode-grouping-slack-ratio must be non-negative",
    ):
        launch.parse_args()


def test_parse_args_accepts_topology_guarded_memory_decode_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "hybrid",
            "--decode-scheduler-policy",
            "topology_guarded_memory",
            "--kv-fast-pairs",
            "0-1,2-3",
        ],
    )

    cfg = launch.parse_args()

    assert cfg.decode_scheduler_policy == "topology_guarded_memory"


def test_parse_args_accepts_kv_topology_alpha(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--kv-topology-alpha", "0.8"],
    )

    cfg = launch.parse_args()

    assert cfg.kv_topology_alpha == 0.8


def test_parse_args_rejects_prefill_overflow_outside_hybrid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--mode", "colocated", "--enable-hybrid-prefill-overflow"],
    )

    with pytest.raises(
        ValueError,
        match="--enable-hybrid-prefill-overflow is supported only in hybrid mode",
    ):
        launch.parse_args()


def test_parse_args_accepts_prefill_overflow_in_hybrid(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "hybrid",
            "--enable-hybrid-prefill-overflow",
            "--prefill-overload-threshold",
            "12",
        ],
    )

    cfg = launch.parse_args()

    assert cfg.enable_hybrid_prefill_overflow is True
    assert cfg.prefill_overload_threshold == 12


def test_parse_args_defaults_cuda_graphs_on(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py"])

    cfg = launch.parse_args()

    assert cfg.enable_cuda_graphs is True


def test_parse_args_disable_cuda_graphs(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py", "--no-enable-cuda-graphs"])

    cfg = launch.parse_args()

    assert cfg.enable_cuda_graphs is False


def test_parse_args_defaults_overflow_on_in_hybrid(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py", "--mode", "hybrid"])

    cfg = launch.parse_args()

    assert cfg.enable_hybrid_prefill_overflow is True


def test_parse_args_defaults_overflow_off_in_colocated(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py", "--mode", "colocated"])

    cfg = launch.parse_args()

    assert cfg.enable_hybrid_prefill_overflow is False


def test_parse_args_disable_overflow_in_hybrid(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--mode", "hybrid", "--no-enable-hybrid-prefill-overflow"],
    )

    cfg = launch.parse_args()

    assert cfg.enable_hybrid_prefill_overflow is False


def test_parse_args_auto_selects_kv_max_pages_from_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "sangam.model.model_loader.read_default_kv_max_pages",
        lambda _: 5632,
    )
    monkeypatch.setattr("sys.argv", ["launch.py"])

    cfg = launch.parse_args()

    assert cfg.kv_max_pages == 5632


def test_parse_args_kv_max_pages_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["launch.py", "--kv-max-pages", "1234"])

    cfg = launch.parse_args()

    assert cfg.kv_max_pages == 1234


def test_parse_args_requires_fast_pairs_for_topology_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "hybrid",
            "--decode-scheduler-policy",
            "topology_guarded_memory",
            "--kv-fast-pairs",
            "",
        ],
    )

    with pytest.raises(
        ValueError,
        match="--kv-fast-pairs is required when --decode-scheduler-policy=topology_guarded_memory",
    ):
        launch.parse_args()


def test_parse_args_allows_topology_policy_in_colocated_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--mode",
            "colocated",
            "--decode-scheduler-policy",
            "topology_guarded_memory",
        ],
    )

    cfg = launch.parse_args()

    assert cfg.mode == "colocated"
    assert cfg.decode_scheduler_policy == "topology_guarded_memory"


def test_main_passes_prefill_scheduler_policy_to_schedulers(monkeypatch) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("colocated")
    cfg.prefill_scheduler_policy = "least_outstanding_prefill_tokens"
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    assert created[0].name == "colocated-scheduler"
    assert (
        _scheduler_config(created[0]).prefill_scheduler_policy
        == "least_outstanding_prefill_tokens"
    )


def test_main_passes_least_outstanding_requests_to_colocated_scheduler(
    monkeypatch,
) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("colocated")
    cfg.prefill_scheduler_policy = "least_outstanding_requests"
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    assert created[0].name == "colocated-scheduler"
    assert (
        _scheduler_config(created[0]).prefill_scheduler_policy
        == "least_outstanding_requests"
    )


def test_main_passes_least_request_length_sum_to_colocated_scheduler(
    monkeypatch,
) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("colocated")
    cfg.prefill_scheduler_policy = "least_request_length_sum"
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    assert created[0].name == "colocated-scheduler"
    assert (
        _scheduler_config(created[0]).prefill_scheduler_policy
        == "least_request_length_sum"
    )


def test_main_passes_decode_grouping_slack_ratio_to_colocated_scheduler(
    monkeypatch,
) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("colocated")
    cfg.prefill_scheduler_policy = "balanced_length_clustering"
    cfg.decode_grouping_slack_ratio = 0.25
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    assert created[0].name == "colocated-scheduler"
    sched_cfg = _scheduler_config(created[0])
    assert sched_cfg.prefill_scheduler_policy == "balanced_length_clustering"
    assert sched_cfg.decode_grouping_slack_ratio == 0.25


def test_main_passes_prefill_queue_policy_to_workers(monkeypatch) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("hybrid")
    cfg.prefill_queue_policy = "fewest_remaining_blocks"
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    prefill_proc = next(proc for proc in created if proc.name == "prefill-0")
    colocated_proc = next(proc for proc in created if proc.name == "colocated-0")
    assert (
        _worker_config(prefill_proc).prefill_queue_policy == "fewest_remaining_blocks"
    )
    assert (
        _worker_config(colocated_proc).prefill_queue_policy == "fewest_remaining_blocks"
    )


def test_main_passes_kv_page_size_to_hybrid_workers(monkeypatch) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("hybrid")
    cfg.kv_page_size = 32
    cfg.kv_max_pages = 4096
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    prefill_proc = next(proc for proc in created if proc.name == "prefill-0")
    prefill_cfg = _worker_config(prefill_proc)
    assert isinstance(prefill_cfg, PrefillWorkerConfig)
    assert prefill_cfg.kv_page_size == 32
    assert prefill_cfg.kv_max_pages == 4096
    assert prefill_cfg.max_prefill_tokens_per_batch == 4096
    assert prefill_cfg.prefill_queue_policy == "arrival_order"


def test_main_launches_hybrid_processes(monkeypatch) -> None:
    created: list[_DummyProcess] = []

    def _process_factory(*, target, args, name, daemon):
        proc = _DummyProcess(target=target, args=args, name=name, daemon=daemon)
        created.append(proc)
        return proc

    cfg = _base_config("hybrid")
    monkeypatch.setattr(launch, "parse_args", lambda: cfg)
    monkeypatch.setattr(launch.mp, "Process", _process_factory)
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    launch.main()

    sched_proc = next(proc for proc in created if proc.name == "hybrid-scheduler")
    sched_cfg = _scheduler_config(sched_proc)
    assert isinstance(sched_cfg, HybridSchedulerConfig)
    assert sched_cfg.prefill_overload_threshold == cfg.prefill_overload_threshold
    assert sched_cfg.enable_prefill_overflow == cfg.enable_hybrid_prefill_overflow

    colocated_proc = next(proc for proc in created if proc.name == "colocated-0")
    colocated_cfg = _worker_config(colocated_proc)
    assert isinstance(colocated_cfg, ColocatedWorkerConfig)
    # Hybrid colocated workers must be set up to receive KV from prefill.
    assert colocated_cfg.enable_kv_receive is True


def test_terminate_processes_escalates_to_kill() -> None:
    class _HangingProcess:
        def __init__(self, name: str) -> None:
            self.name = name
            self.terminated = False
            self.killed = False
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout=None) -> None:
            return None

        def kill(self) -> None:
            self.killed = True
            self._alive = False

    proc = _HangingProcess("worker")
    launch._terminate_processes([proc], launch.EngineLaunchConfig())

    assert proc.terminated is True
    assert proc.killed is True


def test_wait_for_processes_returns_nonzero_exit(monkeypatch) -> None:
    class _ExitedProcess:
        def __init__(self, name: str, exitcode: int | None) -> None:
            self.name = name
            self.exitcode = exitcode

        def join(self, timeout=None) -> None:
            return None

    monkeypatch.setattr(launch.time, "sleep", lambda _: None)

    exit_code = launch._wait_for_processes(
        [_ExitedProcess("worker-0", 1), _ExitedProcess("worker-1", None)],
        launch.EngineLaunchConfig(),
    )

    assert exit_code == 1


def test_resolve_metrics_output_dir_adds_timestamp_for_benchmark_output(
    monkeypatch,
    tmp_path,
) -> None:
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(launch, "datetime", _FixedDatetime)

    output_root = tmp_path / "benchmark_output"
    resolved = launch._resolve_metrics_output_dir(str(output_root))

    assert resolved == str(output_root / "20260102_030405")


def test_resolve_metrics_output_dir_keeps_non_benchmark_output(tmp_path) -> None:
    output_dir = tmp_path / "custom_metrics_dir"
    resolved = launch._resolve_metrics_output_dir(str(output_dir))
    assert resolved == str(output_dir)
