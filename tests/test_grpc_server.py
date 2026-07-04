from __future__ import annotations

import threading
from types import SimpleNamespace

from sangam.engine import grpc_server
from sangam.grpc_utils import DEFAULT_MAX_GRPC_MESSAGE_LENGTH


def test_create_server_sets_expected_grpc_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_grpc_server(executor, options):
        captured["executor"] = executor
        captured["options"] = options
        return object()

    monkeypatch.setattr(grpc_server.grpc, "server", _fake_grpc_server)

    grpc_server.create_server(
        max_workers=7, max_message_length=DEFAULT_MAX_GRPC_MESSAGE_LENGTH
    )

    executor = captured["executor"]
    assert getattr(executor, "_max_workers") == 7
    assert captured["options"] == [
        ("grpc.max_send_message_length", DEFAULT_MAX_GRPC_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", DEFAULT_MAX_GRPC_MESSAGE_LENGTH),
        ("grpc.so_reuseport", 0),
    ]


def test_bind_insecure_port_or_raise_returns_listen_addr() -> None:
    class _FakeServer:
        def __init__(self) -> None:
            self.listen_addr = ""

        def add_insecure_port(self, listen_addr: str) -> int:
            self.listen_addr = listen_addr
            return 50051

    server = _FakeServer()
    listen_addr = grpc_server.bind_insecure_port_or_raise(
        server=server,
        port=50051,
        service_name="scheduler",
    )
    assert listen_addr == "[::]:50051"
    assert server.listen_addr == "[::]:50051"


def test_bind_insecure_port_or_raise_fails_fast() -> None:
    class _FakeServer:
        def add_insecure_port(self, listen_addr: str) -> int:
            return 0

    server = _FakeServer()
    try:
        grpc_server.bind_insecure_port_or_raise(
            server=server,
            port=50051,
            service_name="scheduler",
        )
        assert False, "Expected RuntimeError when bind fails"
    except RuntimeError as exc:
        assert "scheduler failed to bind to port 50051" in str(exc)


def test_suppress_grpc_shutdown_thread_noise_swallows_known_error(monkeypatch) -> None:
    prior_calls: list[object] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: prior_calls.append(args))

    grpc_server.suppress_grpc_shutdown_thread_noise()

    exc = ValueError("server must be started and not shutting down")
    args = SimpleNamespace(
        exc_type=ValueError, exc_value=exc, exc_traceback=None, thread=None
    )
    threading.excepthook(args)

    assert prior_calls == [], "known gRPC shutdown ValueError should be suppressed"


def test_suppress_grpc_shutdown_thread_noise_forwards_other_errors(monkeypatch) -> None:
    prior_calls: list[object] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: prior_calls.append(args))

    grpc_server.suppress_grpc_shutdown_thread_noise()

    exc = ValueError("some unrelated error")
    args = SimpleNamespace(
        exc_type=ValueError, exc_value=exc, exc_traceback=None, thread=None
    )
    threading.excepthook(args)

    assert len(prior_calls) == 1
    assert prior_calls[0] is args
