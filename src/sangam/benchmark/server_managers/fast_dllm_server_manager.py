"""Manage a fastdllm_serve gRPC server subprocess for automated benchmarking."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time

import grpc

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


class FastDllmServerManager:
    """Starts and stops a fastdllm_serve server as a subprocess.

    fastdllm_serve uses the same `sangam.proto` wire format as sangam
    and implements `Submit` / `Poll` / `Generate`. The benchmark client
    drives generation through the blocking `Generate` RPC, but readiness
    is still probed via a cheap `Submit` so the probe does not wait for
    actual generation to finish. fastdllm_serve only accepts
    `unmasking_strategy='conf_threshold'`. Once the probe `Submit`
    returns a request_id the model has finished loading (server.py
    starts the gRPC listener only after `backend.load(...)` completes).
    """

    def __init__(
        self,
        scheduler_address: str,
        launch_args: list[str],
        cuda_visible_devices: str,
        block_length: int,
        startup_timeout: float,
        max_grpc_message_length: int,
    ) -> None:
        self.scheduler_address = scheduler_address
        self.launch_args = launch_args
        self.cuda_visible_devices = cuda_visible_devices
        self.block_length = block_length
        self.startup_timeout = startup_timeout
        self.max_grpc_message_length = max_grpc_message_length
        self._process: subprocess.Popen | None = None
        self._atexit_registered = False

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "fastdllm_serve",
            *self.launch_args,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices}

        logger.info(
            f"Starting fastdllm_serve "
            f"(CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}): "
            f"{' '.join(cmd)}"
        )
        self._process = spawn_process_group(
            cmd=cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

    def wait_for_ready(self) -> None:
        logger.info(
            f"Waiting for fastdllm_serve at {self.scheduler_address} "
            f"(timeout={self.startup_timeout}s)..."
        )
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"fastdllm_serve process exited with code "
                    f"{self._process.returncode} before becoming ready"
                )

            try:
                channel = grpc.insecure_channel(
                    self.scheduler_address,
                    options=grpc_message_length_options(self.max_grpc_message_length),
                )
                grpc.channel_ready_future(channel).result(timeout=5)
                stub = sangam_pb2_grpc.SchedulerServiceStub(channel)

                probe = sangam_pb2.GenerateRequest(
                    prompt_token_ids=[1],
                    gen_length=self.block_length,
                )
                probe.sampling_parameters.temperature = 0.0
                probe.sampling_parameters.unmasking_strategy = "conf_threshold"
                probe.sampling_parameters.confidence_threshold = 0.9
                stub.Submit(probe, timeout=self.startup_timeout)
                channel.close()
                logger.info("fastdllm_serve ready (probe Submit succeeded)")
                return
            except grpc.FutureTimeoutError:
                pass
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    raise RuntimeError(
                        "fastdllm_serve readiness probe returned UNIMPLEMENTED; "
                        "the server is not exposing SchedulerService.Submit/Poll."
                    )
                if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                    raise RuntimeError(
                        "fastdllm_serve readiness probe rejected with "
                        f"INVALID_ARGUMENT: {e.details()}. "
                        "This usually means the proto contract drifted."
                    )
                time.sleep(2)
            except Exception:
                time.sleep(2)

        raise TimeoutError(
            f"fastdllm_serve did not become ready within {self.startup_timeout}s"
        )

    def stop(self) -> None:
        logger.info("Stopping fastdllm_serve...")
        stop_process_group(
            process=self._process,
            process_name="fastdllm_serve process",
            term_timeout_seconds=SERVER_TERM_TIMEOUT_SECONDS,
            kill_timeout_seconds=SERVER_KILL_TIMEOUT_SECONDS,
        )
        logger.info("fastdllm_serve stopped")
