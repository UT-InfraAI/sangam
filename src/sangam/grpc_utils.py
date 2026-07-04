"""Shared gRPC channel/server option helpers."""

from __future__ import annotations

DEFAULT_MAX_GRPC_MESSAGE_LENGTH = 64 * 1024 * 1024


def grpc_message_length_options(max_message_length: int) -> list[tuple[str, int]]:
    return [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]
