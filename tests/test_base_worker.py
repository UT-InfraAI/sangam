from types import SimpleNamespace

import torch

import sangam.worker.base_worker as base_worker_module
from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH
from sangam.worker.base_worker import BaseWorker
from sangam.types import WorkerType
from sangam.worker.worker_config import BaseWorkerConfig


def _make_base_worker_config(**overrides) -> BaseWorkerConfig:
    defaults = dict(
        worker_id="pw-0",
        gpu_id=3,
        dist_rank=1,
        world_size=4,
        port=20100,
        model_name="dummy-model",
        scheduler_address="localhost:50051",
        master_addr="localhost",
        master_port=29500,
        enable_metrics=False,
        enable_operation_metrics=False,
        op_metrics_layer_id=None,
        kv_page_size=16,
        kv_max_pages=128,
        kv_dtype=torch.bfloat16,
        max_grpc_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH,
        poll_interval=0.1,
    )
    defaults.update(overrides)
    return BaseWorkerConfig(**defaults)


class _DummyWorker(BaseWorker):
    def _get_worker_type(self) -> WorkerType:
        return WorkerType.PREFILL

    def _create_servicer(self, model, device) -> object:
        return self._servicer_factory(
            model=model,
            device=device,
            worker_id=self.worker_id,
            dist_rank=self.dist_rank,
            scheduler_address=self.scheduler_address,
        )

    def _add_servicer_to_server(self, servicer: object, server) -> None:
        self._add_servicer_impl(servicer, server)


class _ConfiguredDummyWorker(_DummyWorker):
    def _registration_address(self) -> str:
        return "worker.internal:20200"

    def _registration_extra_fields(self) -> dict[str, int]:
        return {
            "max_pages": 123,
            "page_size": 16,
        }


def test_base_worker_serve_runs_startup_sequence(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class _FakeServer:
        def start(self) -> None:
            events.append(("server.start", None))

        def wait_for_termination(self) -> None:
            raise KeyboardInterrupt

        def stop(self, grace: int) -> None:
            events.append(("server.stop", grace))

    class _FakeChannel:
        def close(self) -> None:
            events.append(("channel.close", None))

    class _FakeSchedulerStub:
        def RegisterWorker(self, request):
            events.append(("scheduler.register", request))
            return SimpleNamespace(success=True)

    def _make_servicer(**kwargs):
        events.append(("create_servicer", kwargs["worker_id"]))
        return SimpleNamespace(
            shutdown=lambda: events.append(("servicer.shutdown", None))
        )

    worker = _DummyWorker(_make_base_worker_config())
    worker._servicer_factory = _make_servicer
    worker._add_servicer_impl = lambda servicer, server: events.append(
        ("add_servicer", None)
    )

    monkeypatch.setattr(
        base_worker_module.torch.cuda,
        "set_device",
        lambda gpu_id: events.append(("cuda.set_device", gpu_id)),
    )
    monkeypatch.setattr(
        base_worker_module,
        "init_distributed",
        lambda rank, world_size, master_addr, master_port, device_id: events.append(
            (
                "init_distributed",
                (rank, world_size, master_addr, master_port, device_id),
            )
        ),
    )
    monkeypatch.setattr(
        base_worker_module,
        "load_model",
        lambda model_name, gpu_id, dtype: (
            events.append(("load_model", (model_name, gpu_id))) or "model"
        ),
    )
    monkeypatch.setattr(
        base_worker_module,
        "create_server",
        lambda max_workers, max_message_length: (
            events.append(("create_server", max_workers)) or _FakeServer()
        ),
    )
    monkeypatch.setattr(
        base_worker_module,
        "bind_insecure_port_or_raise",
        lambda server, port, service_name: (
            events.append(("bind", (port, service_name))) or f"[::]:{port}"
        ),
    )
    monkeypatch.setattr(
        base_worker_module.grpc,
        "insecure_channel",
        lambda address: events.append(("insecure_channel", address)) or _FakeChannel(),
    )
    monkeypatch.setattr(
        base_worker_module.sangam_pb2_grpc,
        "SchedulerWorkerServiceStub",
        lambda channel: _FakeSchedulerStub(),
    )

    worker.serve()

    event_names = [name for name, _ in events]
    assert event_names == [
        "cuda.set_device",
        "init_distributed",
        "load_model",
        "create_servicer",
        "create_server",
        "add_servicer",
        "bind",
        "server.start",
        "insecure_channel",
        "scheduler.register",
        "channel.close",
        "server.stop",
        "servicer.shutdown",
    ]

    register_request = next(
        payload for name, payload in events if name == "scheduler.register"
    )
    assert register_request.worker_id == "pw-0"
    assert register_request.worker_type == WorkerType.PREFILL.value
    assert register_request.address == "localhost:20100"
    assert register_request.gpu_id == 3


def test_base_worker_register_with_scheduler_uses_registration_hooks(
    monkeypatch,
) -> None:
    events: list[tuple[str, object]] = []

    class _FakeChannel:
        def close(self) -> None:
            events.append(("channel.close", None))

    class _FakeSchedulerStub:
        def RegisterWorker(self, request):
            events.append(("scheduler.register", request))
            return SimpleNamespace(success=True)

    worker = _ConfiguredDummyWorker(
        _make_base_worker_config(worker_id="w-1", gpu_id=0, dist_rank=2)
    )

    monkeypatch.setattr(
        base_worker_module.grpc,
        "insecure_channel",
        lambda address: events.append(("insecure_channel", address)) or _FakeChannel(),
    )
    monkeypatch.setattr(
        base_worker_module.sangam_pb2_grpc,
        "SchedulerWorkerServiceStub",
        lambda channel: _FakeSchedulerStub(),
    )

    worker._register_with_scheduler()

    register_request = next(
        payload for name, payload in events if name == "scheduler.register"
    )
    assert register_request.worker_id == "w-1"
    assert register_request.worker_type == WorkerType.PREFILL.value
    assert register_request.address == "worker.internal:20200"
    assert register_request.dist_rank == 2
    assert register_request.gpu_id == 0
    assert register_request.max_pages == 123
    assert register_request.page_size == 16


def test_create_scheduler_callback_stub_uses_message_limits(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    fake_channel = object()
    fake_stub = object()

    monkeypatch.setattr(
        base_worker_module.grpc,
        "insecure_channel",
        lambda address, options=None: (
            recorded.update({"address": address, "options": options}) or fake_channel
        ),
    )
    monkeypatch.setattr(
        base_worker_module.sangam_pb2_grpc,
        "SchedulerWorkerServiceStub",
        lambda channel: fake_stub,
    )

    channel, stub = base_worker_module.create_scheduler_callback_stub(
        "localhost:50051", max_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH
    )

    assert channel is fake_channel
    assert stub is fake_stub
    # Worker callbacks live on scheduler_port + 1 (separate gRPC server / pool).
    assert recorded["address"] == "localhost:50052"
    assert recorded["options"] == [
        ("grpc.max_send_message_length", DEFAULT_MAX_GRPC_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", DEFAULT_MAX_GRPC_MESSAGE_LENGTH),
    ]


def test_base_worker_serve_ignores_server_stop_during_grpc_shutdown(
    monkeypatch,
) -> None:
    events: list[tuple[str, object]] = []

    class _FakeServer:
        def start(self) -> None:
            events.append(("server.start", None))

        def wait_for_termination(self) -> None:
            raise KeyboardInterrupt

        def stop(self, grace: int) -> None:
            events.append(("server.stop", grace))
            raise ValueError("server must be started and not shutting down")

    class _FakeChannel:
        def close(self) -> None:
            events.append(("channel.close", None))

    class _FakeSchedulerStub:
        def RegisterWorker(self, request):
            events.append(("scheduler.register", request))
            return SimpleNamespace(success=True)

    worker = _DummyWorker(_make_base_worker_config())
    worker._servicer_factory = lambda **kwargs: SimpleNamespace(
        shutdown=lambda: events.append(("servicer.shutdown", None))
    )
    worker._add_servicer_impl = lambda servicer, server: events.append(
        ("add_servicer", None)
    )

    monkeypatch.setattr(
        base_worker_module.torch.cuda, "set_device", lambda gpu_id: None
    )
    monkeypatch.setattr(
        base_worker_module,
        "init_distributed",
        lambda rank, world_size, master_addr, master_port, device_id: None,
    )
    monkeypatch.setattr(
        base_worker_module,
        "load_model",
        lambda model_name, gpu_id, dtype: "model",
    )
    monkeypatch.setattr(
        base_worker_module,
        "create_server",
        lambda max_workers, max_message_length: _FakeServer(),
    )
    monkeypatch.setattr(
        base_worker_module,
        "bind_insecure_port_or_raise",
        lambda server, port, service_name: f"[::]:{port}",
    )
    monkeypatch.setattr(
        base_worker_module.grpc,
        "insecure_channel",
        lambda address: _FakeChannel(),
    )
    monkeypatch.setattr(
        base_worker_module.sangam_pb2_grpc,
        "SchedulerWorkerServiceStub",
        lambda channel: _FakeSchedulerStub(),
    )

    worker.serve()

    assert ("server.stop", 0) in events
    assert ("servicer.shutdown", None) in events


def test_base_worker_serve_ignores_keyboard_interrupt_during_server_stop(
    monkeypatch,
) -> None:
    events: list[tuple[str, object]] = []

    class _FakeServer:
        def start(self) -> None:
            events.append(("server.start", None))

        def wait_for_termination(self) -> None:
            raise KeyboardInterrupt

        def stop(self, grace: int) -> None:
            events.append(("server.stop", grace))
            raise KeyboardInterrupt

    class _FakeChannel:
        def close(self) -> None:
            events.append(("channel.close", None))

    class _FakeSchedulerStub:
        def RegisterWorker(self, request):
            events.append(("scheduler.register", request))
            return SimpleNamespace(success=True)

    worker = _DummyWorker(_make_base_worker_config())
    worker._servicer_factory = lambda **kwargs: SimpleNamespace(
        shutdown=lambda: events.append(("servicer.shutdown", None))
    )
    worker._add_servicer_impl = lambda servicer, server: events.append(
        ("add_servicer", None)
    )

    monkeypatch.setattr(
        base_worker_module.torch.cuda, "set_device", lambda gpu_id: None
    )
    monkeypatch.setattr(
        base_worker_module,
        "init_distributed",
        lambda rank, world_size, master_addr, master_port, device_id: None,
    )
    monkeypatch.setattr(
        base_worker_module,
        "load_model",
        lambda model_name, gpu_id, dtype: "model",
    )
    monkeypatch.setattr(
        base_worker_module,
        "create_server",
        lambda max_workers, max_message_length: _FakeServer(),
    )
    monkeypatch.setattr(
        base_worker_module,
        "bind_insecure_port_or_raise",
        lambda server, port, service_name: f"[::]:{port}",
    )
    monkeypatch.setattr(
        base_worker_module.grpc,
        "insecure_channel",
        lambda address: _FakeChannel(),
    )
    monkeypatch.setattr(
        base_worker_module.sangam_pb2_grpc,
        "SchedulerWorkerServiceStub",
        lambda channel: _FakeSchedulerStub(),
    )

    worker.serve()

    assert ("server.stop", 0) in events
    assert ("servicer.shutdown", None) in events
