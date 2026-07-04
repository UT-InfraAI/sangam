"""Shared subprocess lifecycle helpers for long-lived process trees."""

from __future__ import annotations

import os
import signal
import subprocess

from sangam.logger import init_logger

logger = init_logger(__name__)

# Used by callers that wrap an entire server subprocess group
# (benchmark/server_managers/*). Distinct from
# EngineLaunchConfig.{term,kill}_timeout_seconds, which govern in-process
# engine shutdown.
SERVER_TERM_TIMEOUT_SECONDS = 15.0
SERVER_KILL_TIMEOUT_SECONDS = 5.0

# Capacity-search wraps a benchmark subprocess that itself wraps a server, so
# its budget must exceed SERVER_TERM_TIMEOUT_SECONDS to give the inner server
# manager time to teardown before the trial is killed.
TRIAL_TERM_TIMEOUT_SECONDS = 2 * SERVER_TERM_TIMEOUT_SECONDS
TRIAL_KILL_TIMEOUT_SECONDS = SERVER_KILL_TIMEOUT_SECONDS


def spawn_process_group(
    *,
    cmd: list[str],
    stdout,
    stderr,
    env: dict | None = None,
) -> subprocess.Popen:
    """Spawn a subprocess in a new session/process-group."""
    return subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        env=env,
    )


def stop_process_group(
    *,
    process: subprocess.Popen | None,
    process_name: str,
    term_timeout_seconds: float,
    kill_timeout_seconds: float,
) -> None:
    """Stop a subprocess process-group with SIGTERM, escalating to SIGKILL."""
    if process is None or process.poll() is not None:
        return

    def _signal_group(sig: int) -> None:
        if process.poll() is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            return
        os.killpg(pgid, sig)

    try:
        _signal_group(signal.SIGTERM)
        process.wait(timeout=term_timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            f"{process_name} did not exit after SIGTERM; escalating to SIGKILL."
        )
    except ProcessLookupError:
        return

    try:
        _signal_group(signal.SIGKILL)
        process.wait(timeout=kill_timeout_seconds)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        logger.error(f"{process_name} did not exit after SIGKILL.")


def install_termination_handler() -> None:
    """Make SIGTERM/SIGINT raise KeyboardInterrupt on the main thread.

    Children rely on existing `try/except KeyboardInterrupt` cleanup around
    blocking calls (e.g. gRPC `wait_for_termination`) for graceful shutdown.
    """

    def _raise_kbi(_signum: int, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_kbi)
    signal.signal(signal.SIGINT, _raise_kbi)
