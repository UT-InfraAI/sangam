"""Abstract base class for prefill and decode workers.

Consolidates the shared startup sequence (CUDA device, NCCL init, model
loading, gRPC server, scheduler registration) so that concrete workers
only implement their servicer logic.
"""

import abc

import grpc
import torch

from sangam.config_utils import worker_callback_address
from sangam.engine.grpc_server import (
    bind_insecure_port_or_raise,
    create_server,
    suppress_grpc_shutdown_thread_noise,
)
from sangam.logger import init_logger
from sangam.model.model_loader import load_model, init_distributed
from sangam.process_lifecycle import install_termination_handler
from sangam.proto import sangam_pb2, sangam_pb2_grpc
from sangam.types import WorkerType
from sangam.worker.worker_config import BaseWorkerConfig

logger = init_logger(__name__)


def create_scheduler_callback_stub(
    scheduler_address: str,
    max_message_length: int,
) -> tuple[grpc.Channel, sangam_pb2_grpc.SchedulerWorkerServiceStub]:
    """Create a scheduler callback stub with worker-sized gRPC message limits.

    `scheduler_address` is the client-facing address; this helper translates
    it to the worker-callback port so worker traffic uses the dedicated pool
    on the scheduler.
    """
    channel = grpc.insecure_channel(
        worker_callback_address(scheduler_address),
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    return channel, sangam_pb2_grpc.SchedulerWorkerServiceStub(channel)


def _stop_server_quietly(server: grpc.Server) -> None:
    try:
        server.stop(grace=0)
    except KeyboardInterrupt:
        return
    except ValueError as exc:
        if "server must be started and not shutting down" not in str(exc):
            raise


class BaseWorker(abc.ABC):
    """Shared boilerplate for GPU-based gRPC workers."""

    def __init__(self, config: BaseWorkerConfig) -> None:
        self._config = config
        self.worker_id = config.worker_id
        self.gpu_id = config.gpu_id
        self.dist_rank = config.dist_rank
        self.world_size = config.world_size
        self.port = config.port
        self.model_name = config.model_name
        self.scheduler_address = config.scheduler_address
        self.master_addr = config.master_addr
        self.master_port = config.master_port
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._server: grpc.Server | None = None

    # -- abstract hooks --------------------------------------------------

    @abc.abstractmethod
    def _get_worker_type(self) -> WorkerType:
        """Return the worker type enum for scheduler registration."""

    @abc.abstractmethod
    def _create_servicer(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> object:
        """Create and return the gRPC servicer instance."""

    @abc.abstractmethod
    def _add_servicer_to_server(self, servicer: object, server: grpc.Server) -> None:
        """Add the servicer to the gRPC server."""

    def _registration_address(self) -> str:
        """Return the scheduler registration address for this worker."""
        return f"localhost:{self.port}"

    def _registration_extra_fields(self) -> dict[str, int]:
        """Return any extra scheduler registration fields for this worker."""
        return {}

    # -- shared startup sequence -----------------------------------------

    def serve(self) -> None:
        """Run the full worker lifecycle: init, serve, register, block."""
        install_termination_handler()
        suppress_grpc_shutdown_thread_noise()
        # 1. Set CUDA device
        torch.cuda.set_device(self.gpu_id)
        device = torch.device(f"cuda:{self.gpu_id}")
        self._device = device

        # 2. Initialize NCCL process group
        init_distributed(
            self.dist_rank,
            self.world_size,
            self.master_addr,
            self.master_port,
            device_id=self.gpu_id,
        )

        # 3. Load model
        model = load_model(
            self.model_name, gpu_id=self.gpu_id, dtype=self._config.kv_dtype
        )
        self._model = model

        # 4. Start gRPC server
        servicer = self._create_servicer(model, device)
        server = create_server(
            max_workers=16,
            max_message_length=self._config.max_grpc_message_length,
        )
        self._server = server
        self._add_servicer_to_server(servicer, server)
        bind_insecure_port_or_raise(
            server=server,
            port=self.port,
            service_name=f"{self._get_worker_type().value} worker",
        )
        server.start()
        logger.debug(
            f"{self._get_worker_type().value} worker serving on port {self.port}"
        )

        # 5. Register with scheduler
        self._register_with_scheduler()

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                _stop_server_quietly(server)

                # Clean shutdown of GPU work queue if servicer supports it.
                if hasattr(servicer, "shutdown"):
                    servicer.shutdown()  # type: ignore[attr-defined]
            except KeyboardInterrupt:
                pass

    def _register_with_scheduler(
        self,
        *,
        worker_type: WorkerType | None = None,
        extra_fields: dict[str, int] | None = None,
        is_conversion: bool = False,
    ) -> bool:
        """Register this worker with the central scheduler."""
        channel = grpc.insecure_channel(worker_callback_address(self.scheduler_address))
        stub = sangam_pb2_grpc.SchedulerWorkerServiceStub(channel)
        request_fields = {
            "worker_id": self.worker_id,
            "worker_type": (worker_type or self._get_worker_type()).value,
            "address": self._registration_address(),
            "dist_rank": self.dist_rank,
            "gpu_id": self.gpu_id,
            "is_conversion": is_conversion,
        }
        request_fields.update(extra_fields or self._registration_extra_fields())
        resp = stub.RegisterWorker(sangam_pb2.RegisterWorkerRequest(**request_fields))
        if resp.success:
            logger.debug(f"Registered with scheduler at {self.scheduler_address}")
        else:
            logger.error("Failed to register with scheduler")
        channel.close()
        return bool(resp.success)
