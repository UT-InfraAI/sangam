"""Unified launch entry point for colocated and hybrid modes.

Uses multiprocessing to start all processes on a single node.
"""

import argparse
import multiprocessing as mp
import os
import time
from datetime import datetime

from sangam.config_utils import scheduler_address_from_port
from sangam.engine.launch_config import (
    EngineLaunchConfig,
    add_engine_launch_args,
    build_colocated_scheduler_config,
    build_colocated_worker_config,
    build_hybrid_scheduler_config,
    build_prefill_worker_config,
    engine_launch_config_from_namespace,
)
from sangam.engine.scheduler_config import (
    ColocatedSchedulerConfig,
    HybridSchedulerConfig,
)
from sangam.logger import init_logger
from sangam.process_lifecycle import install_termination_handler
from sangam.worker.worker_config import (
    ColocatedWorkerConfig,
    PrefillWorkerConfig,
)

logger = init_logger(__name__)


def _resolve_metrics_output_dir(metrics_output_dir: str) -> str:
    normalized = os.path.normpath(metrics_output_dir)
    if os.path.basename(normalized) != "benchmark_output":
        return metrics_output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(metrics_output_dir, timestamp)


def _run_colocated_scheduler(port: int, config: ColocatedSchedulerConfig) -> None:
    from sangam.engine.colocated_scheduler import serve_colocated_scheduler

    serve_colocated_scheduler(port, config)


def _run_prefill_worker(config: PrefillWorkerConfig) -> None:
    from sangam.worker.prefill_worker import serve_prefill_worker

    serve_prefill_worker(config)


def _run_hybrid_scheduler(port: int, config: HybridSchedulerConfig) -> None:
    from sangam.engine.hybrid_scheduler import serve_hybrid_scheduler

    serve_hybrid_scheduler(port, config)


def _run_colocated_worker(config: ColocatedWorkerConfig) -> None:
    from sangam.worker.colocated_worker import serve_colocated_worker

    serve_colocated_worker(config)


def _parse_gpu_ids(csv_gpu_ids: str, arg_name: str) -> list[int]:
    gpu_ids = [item.strip() for item in csv_gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError(f"{arg_name} must specify at least one GPU id")
    return [int(g) for g in gpu_ids]


def parse_args() -> EngineLaunchConfig:
    parser = argparse.ArgumentParser(description="Launch sangam engine")
    add_engine_launch_args(parser, include_metrics_output_dir=True)
    args = parser.parse_args()
    return engine_launch_config_from_namespace(args)


def _launch_colocated(cfg: EngineLaunchConfig) -> list[mp.Process]:
    gpus = _parse_gpu_ids(cfg.gpus, "--gpus")
    world_size = len(gpus)

    scheduler_address = scheduler_address_from_port(cfg.scheduler_port, cfg.master_addr)
    logger.info(f"Launching colocated mode with {world_size} workers on GPUs {gpus}")
    logger.info(f"torch.distributed world_size={world_size}")

    processes: list[mp.Process] = []

    scheduler_config = build_colocated_scheduler_config(cfg)
    p_sched = mp.Process(
        target=_run_colocated_scheduler,
        args=(cfg.scheduler_port, scheduler_config),
        name="colocated-scheduler",
        daemon=True,
    )
    p_sched.start()
    processes.append(p_sched)

    time.sleep(2)

    for i, gpu_id in enumerate(gpus):
        worker_id = f"colocated-{i}"
        port = cfg.base_worker_port + i
        worker_config = build_colocated_worker_config(
            cfg,
            worker_id=worker_id,
            gpu_id=gpu_id,
            dist_rank=i,
            world_size=world_size,
            port=port,
            scheduler_address=scheduler_address,
            enable_kv_receive=False,
        )
        p = mp.Process(
            target=_run_colocated_worker,
            args=(worker_config,),
            name=worker_id,
            daemon=True,
        )
        p.start()
        processes.append(p)
    return processes


def _launch_hybrid(cfg: EngineLaunchConfig) -> list[mp.Process]:
    prefill_gpus = _parse_gpu_ids(cfg.prefill_gpus, "--prefill-gpus")
    colocated_gpus = _parse_gpu_ids(
        cfg.hybrid_colocated_gpus, "--hybrid-colocated-gpus"
    )
    num_prefill = len(prefill_gpus)
    num_colocated = len(colocated_gpus)
    world_size = num_prefill + num_colocated

    scheduler_address = scheduler_address_from_port(cfg.scheduler_port, cfg.master_addr)

    logger.info(
        f"Launching hybrid mode: {num_prefill} prefill workers (GPUs {prefill_gpus}), "
        f"{num_colocated} colocated workers (GPUs {colocated_gpus})"
    )
    logger.info(f"torch.distributed world_size={world_size}")

    processes: list[mp.Process] = []

    # 1. Start hybrid scheduler (CPU-only)
    scheduler_config = build_hybrid_scheduler_config(cfg)
    p_sched = mp.Process(
        target=_run_hybrid_scheduler,
        args=(cfg.scheduler_port, scheduler_config),
        name="hybrid-scheduler",
        daemon=True,
    )
    p_sched.start()
    processes.append(p_sched)

    time.sleep(2)

    # 2. Start prefill workers (ranks 0..P-1)
    for i, gpu_id in enumerate(prefill_gpus):
        worker_id = f"prefill-{i}"
        dist_rank = i
        port = cfg.base_worker_port + i
        prefill_config = build_prefill_worker_config(
            cfg,
            worker_id=worker_id,
            gpu_id=gpu_id,
            dist_rank=dist_rank,
            world_size=world_size,
            port=port,
            scheduler_address=scheduler_address,
        )

        p = mp.Process(
            target=_run_prefill_worker,
            args=(prefill_config,),
            name=worker_id,
            daemon=True,
        )
        p.start()
        processes.append(p)

    # 3. Start colocated workers (ranks P..P+C-1)
    for i, gpu_id in enumerate(colocated_gpus):
        worker_id = f"colocated-{i}"
        dist_rank = num_prefill + i
        port = cfg.base_worker_port + num_prefill + i
        colocated_config = build_colocated_worker_config(
            cfg,
            worker_id=worker_id,
            gpu_id=gpu_id,
            dist_rank=dist_rank,
            world_size=world_size,
            port=port,
            scheduler_address=scheduler_address,
            enable_kv_receive=True,
        )

        p = mp.Process(
            target=_run_colocated_worker,
            args=(colocated_config,),
            name=worker_id,
            daemon=True,
        )
        p.start()
        processes.append(p)

    return processes


def _terminate_processes(processes: list[mp.Process], cfg: EngineLaunchConfig) -> None:
    for proc in processes:
        if proc.is_alive():
            proc.terminate()

    for proc in processes:
        if proc.is_alive():
            proc.join(timeout=cfg.term_timeout_seconds)

    for proc in processes:
        if proc.is_alive():
            logger.warning("Process %s did not exit after SIGTERM; killing.", proc.name)
            proc.kill()
            proc.join(timeout=cfg.kill_timeout_seconds)


def _wait_for_processes(processes: list[mp.Process], cfg: EngineLaunchConfig) -> int:
    while True:
        for proc in processes:
            proc.join(timeout=0)
            if proc.exitcode is None:
                continue
            if proc.exitcode != 0:
                logger.error(
                    "Process %s exited with code %s; shutting down engine.",
                    proc.name,
                    proc.exitcode,
                )
                return proc.exitcode
        if all(proc.exitcode is not None for proc in processes):
            return 0
        time.sleep(cfg.poll_interval)


def main() -> None:
    cfg = parse_args()
    cfg.metrics_output_dir = _resolve_metrics_output_dir(cfg.metrics_output_dir)
    logger.info("Launch config: %s", cfg)
    if cfg.mode == "colocated":
        processes = _launch_colocated(cfg)
    elif cfg.mode == "hybrid":
        processes = _launch_hybrid(cfg)
    else:
        raise ValueError(f"Unsupported serving mode: {cfg.mode!r}")
    install_termination_handler()

    logger.info("All processes launched. Waiting for workers to register...")

    # Wait for all processes
    exit_code = 0
    try:
        exit_code = _wait_for_processes(processes, cfg)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal; stopping processes gracefully...")
    finally:
        _terminate_processes(processes, cfg)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
