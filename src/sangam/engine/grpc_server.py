"""Shared helpers for robust gRPC server startup."""

from __future__ import annotations

import threading
from concurrent import futures

import grpc


def suppress_grpc_shutdown_thread_noise() -> None:
    """Suppress ValueError from gRPC internal serve threads during shutdown.

    gRPC's thread pool loops calling request_call(). After server.stop(), those
    threads may fire one last request_call() before observing the shutdown state,
    raising:

        ValueError: server must be started and not shutting down

    Python prints these as "Exception in thread" noise. This hook silences that
    specific error; all other thread exceptions are forwarded to the prior hook.
    Call once per process before server.start().
    """
    _prev = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if (
            args.exc_type is ValueError
            and args.exc_value is not None
            and "server must be started and not shutting down" in str(args.exc_value)
        ):
            return
        _prev(args)

    threading.excepthook = _hook


def create_server(max_workers: int, max_message_length: int) -> grpc.Server:
    """Create a gRPC server with shared transport options.

    `grpc.so_reuseport=0` enforces exclusive bind semantics so duplicate
    schedulers/workers fail fast instead of silently sharing a port.
    """
    return grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            ("grpc.so_reuseport", 0),
        ],
    )


def bind_insecure_port_or_raise(
    server: grpc.Server, port: int, service_name: str
) -> str:
    """Bind a server to `port` or raise with a clear startup error."""
    listen_addr = f"[::]:{port}"
    bound_port = server.add_insecure_port(listen_addr)
    if bound_port == 0:
        raise RuntimeError(
            f"{service_name} failed to bind to port {port}. "
            "Another process is likely already using this port."
        )
    return listen_addr
