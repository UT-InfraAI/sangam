"""Shared configuration helpers used by CLI entrypoints."""


def scheduler_address_from_port(scheduler_port: int, host: str) -> str:
    return f"{host}:{scheduler_port}"


def worker_callback_address(scheduler_address: str) -> str:
    """Translate a `host:port` client address to the worker-callback address.

    The scheduler binds two gRPC servers: one on `port` for client RPCs
    (Generate / Submit / Poll) and one on `port + 1` for worker callbacks
    (RegisterWorker / ReportBatchMetrics / ReportKVTransfer). Workers use
    this helper to derive their callback target from the address they were
    started with.
    """
    host, _, port = scheduler_address.rpartition(":")
    if not host or not port:
        raise ValueError(
            f"scheduler_address must be in 'host:port' form, got {scheduler_address!r}"
        )
    return f"{host}:{int(port) + 1}"
