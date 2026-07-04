"""Manage a sangam server subprocess for automated benchmarking."""

from __future__ import annotations

import atexit
import subprocess
import sys
import time

import grpc

from sangam.engine.launch_config import EngineLaunchConfig
from sangam.grpc_utils import grpc_message_length_options
from sangam.logger import init_logger
from sangam.process_lifecycle import (
    SERVER_KILL_TIMEOUT_SECONDS,
    SERVER_TERM_TIMEOUT_SECONDS,
    spawn_process_group,
    stop_process_group,
)
from sangam.proto import sangam_pb2, sangam_pb2_grpc

logger = init_logger(__name__)


class SangamServerManager:
    """Starts and stops a sangam server as a subprocess."""

    def __init__(
        self,
        scheduler_address: str,
        launch_config: EngineLaunchConfig,
        startup_timeout: float,
    ) -> None:
        self.scheduler_address = scheduler_address
        self.launch_config = launch_config
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._atexit_registered = False
        (
            self._expected_prefill_workers,
            self._expected_decode_workers,
            self._expected_colocated_workers,
        ) = self._expected_worker_counts()

    def _parse_gpu_count(self, gpu_csv: str) -> int:
        return len([gpu.strip() for gpu in gpu_csv.split(",") if gpu.strip()])

    def _expected_worker_counts(self) -> tuple[int, int, int]:
        if self.launch_config.mode == "colocated":
            return (0, 0, self._parse_gpu_count(self.launch_config.gpus))
        if self.launch_config.mode == "hybrid":
            return (
                self._parse_gpu_count(self.launch_config.prefill_gpus),
                0,
                self._parse_gpu_count(self.launch_config.hybrid_colocated_gpus),
            )
        raise ValueError(f"Unsupported serving mode: {self.launch_config.mode!r}")

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "sangam.entrypoints.launch",
            *self.launch_config.to_cli_args(),
        ]

        logger.debug(f"Starting server: {' '.join(cmd)}")
        self._process = spawn_process_group(
            cmd=cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

    def wait_for_ready(self) -> None:
        logger.info(
            f"Waiting for scheduler at {self.scheduler_address} "
            f"(timeout={self.startup_timeout}s)..."
        )
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            # Check that subprocess is still alive
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {self._process.returncode} "
                    "before becoming ready"
                )

            try:
                channel = grpc.insecure_channel(
                    self.scheduler_address,
                    options=grpc_message_length_options(
                        self.launch_config.max_grpc_message_length
                    ),
                )
                grpc.channel_ready_future(channel).result(timeout=5)
                stub = sangam_pb2_grpc.SchedulerServiceStub(channel)
                status = stub.GetSchedulerStatus(sangam_pb2.GetSchedulerStatusRequest())
                prefill_ready = (
                    status.num_prefill_workers >= self._expected_prefill_workers
                )
                decode_ready = (
                    status.num_decode_workers >= self._expected_decode_workers
                )
                colocated_ready = (
                    status.num_colocated_workers >= self._expected_colocated_workers
                )
                if prefill_ready and decode_ready and colocated_ready:
                    channel.close()
                    logger.info(
                        "Scheduler and workers are ready "
                        f"(prefill={status.num_prefill_workers}, "
                        f"decode={status.num_decode_workers}, "
                        f"colocated={status.num_colocated_workers})"
                    )
                    return
                channel.close()
                time.sleep(0.5)
            except grpc.FutureTimeoutError:
                pass
            except Exception:
                time.sleep(2)

        raise TimeoutError(
            f"Server did not become ready within {self.startup_timeout}s"
        )

    def stop(self) -> None:
        logger.debug("Stopping server...")
        stop_process_group(
            process=self._process,
            process_name="Server process",
            term_timeout_seconds=SERVER_TERM_TIMEOUT_SECONDS,
            kill_timeout_seconds=SERVER_KILL_TIMEOUT_SECONDS,
        )
        logger.info("Server stopped")
