"""Benchmark harness entry point.

Usage:
    # Auto-launch server (default), benchmark, teardown:
    python -m sangam.benchmark.main \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --mode hybrid --prefill-gpus 0,1 --hybrid-colocated-gpus 2,3 \
        --num-requests 200 \
        --interval-type poisson --qps 2.0 \
        --length-type fixed --prefill-tokens 256 --decode-tokens 64 \
        --gen-length 64 --block-length 32

    # Connect to existing server:
    python -m sangam.benchmark.main \
        --no-launch-server \
        --scheduler-address localhost:50051 \
        --num-requests 200 ...

    # Benchmark Fast-dLLM's basic single-process server. `fastdllm_serve`
    # must be importable; the simplest setup (no install) is to prepend
    # Fast-dLLM/v1 to PYTHONPATH so both this process and the spawned
    # server subprocess pick it up:
    PYTHONPATH=$HOME/Fast-dLLM/v1 python -m sangam.benchmark.main \
        --backend fast-dllm \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --mode colocated --gpus 0 \
        --gen-length 64 --block-length 32 --max-batch-size 4 \
        --fast-dllm-default-threshold 0.9
"""

import os
import signal
import sys
from datetime import datetime

import yaml

from sangam.benchmark.backends import (
    SangamBackend,
    FastDllmBackend,
)
from sangam.benchmark.benchmark_runner import BenchmarkRunner
from sangam.benchmark.config import BenchmarkConfig
from sangam.benchmark.server_managers import (
    SangamServerManager,
    FastDllmServerManager,
)
from sangam.logger import init_logger
from sangam.process_lifecycle import install_termination_handler

logger = init_logger(__name__)


def _resolve_output_dir(output_dir: str) -> str:
    normalized = os.path.normpath(output_dir)
    if os.path.basename(normalized) != "benchmark_output":
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, timestamp)


def _build_server(config: BenchmarkConfig):
    if config.backend == "sangam":
        return SangamServerManager(
            scheduler_address=config.scheduler_address,
            launch_config=config.build_engine_launch_config(
                metrics_output_dir=config.output_dir
            ),
            startup_timeout=config.server_startup_timeout,
        )
    if config.backend == "fast-dllm":
        return FastDllmServerManager(
            scheduler_address=config.scheduler_address,
            launch_args=config.build_fast_dllm_launch_args(),
            cuda_visible_devices=config.fast_dllm_gpu_csv(),
            block_length=config.block_length,
            startup_timeout=config.server_startup_timeout,
            max_grpc_message_length=config.max_grpc_message_length,
        )
    raise ValueError(f"Unknown backend: {config.backend}")


def _build_backend(config: BenchmarkConfig):
    if config.backend == "sangam":
        return SangamBackend(config)
    if config.backend == "fast-dllm":
        return FastDllmBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")


class _BenchmarkTimeout(Exception):
    """Raised by SIGALRM when --benchmark-timeout is exceeded."""


def _arm_benchmark_deadline(timeout: float | None) -> bool:
    if timeout is None or timeout <= 0:
        return False

    def _handler(signum, frame):
        raise _BenchmarkTimeout(f"Benchmark exceeded timeout of {timeout}s")

    signal.signal(signal.SIGALRM, _handler)
    # alarm() only accepts integer seconds; round up so we never fire early.
    signal.alarm(max(1, int(timeout + 0.5)))
    return True


def run_benchmark(config: BenchmarkConfig) -> None:
    config.output_dir = _resolve_output_dir(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    config_path = os.path.join(config.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    server_ref: list = [None]
    timed_out = False
    succeeded = False
    armed = _arm_benchmark_deadline(config.benchmark_timeout)

    def _abort_stop_server() -> None:
        srv = server_ref[0]
        if srv is None:
            return
        # Clear first so the outer finally does not double-stop.
        server_ref[0] = None
        srv.stop()

    runner: BenchmarkRunner | None = None
    try:
        if config.launch_server:
            server_ref[0] = _build_server(config)
            server_ref[0].start()
            server_ref[0].wait_for_ready()

        logger.info(
            f"Starting benchmark (backend={config.backend}, output: {config.output_dir})"
        )
        backend = _build_backend(config)
        runner = BenchmarkRunner(config, backend, on_abort=_abort_stop_server)
        results = runner.run()
        runner.save_results(results)
        succeeded = bool(results)
    except _BenchmarkTimeout as e:
        timed_out = True
        logger.warning(
            f"Benchmark terminated due to timeout: {e}. "
            f"Pass --export-partial-metrics to flush partial metrics on teardown."
        )
    finally:
        if runner is not None and (succeeded or config.export_partial_metrics):
            try:
                runner.export_scheduler_metrics()
            except Exception as e:
                logger.warning(f"Failed to export scheduler metrics: {e}")
        elif runner is not None:
            logger.warning(
                "Skipping metrics export after benchmark failure "
                "(pass --export-partial-metrics to flush partial metrics)."
            )
        if armed:
            signal.alarm(0)
        if server_ref[0] is not None:
            server_ref[0].stop()
    if timed_out:
        raise SystemExit(2)


def main() -> None:
    config = BenchmarkConfig.create_from_cli_args()
    install_termination_handler()
    try:
        run_benchmark(config)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
