"""Serializes GPU-bound operations behind a single worker thread.

gRPC handler threads enqueue work items and block until the dedicated
GPU thread processes them, ensuring only one CUDA/NCCL operation runs
at a time per worker.
"""

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from sangam.logger import init_logger

logger = init_logger(__name__)


@dataclass
class GpuWorkItem:
    """A unit of work to be executed on the GPU thread."""

    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    result: Any = None
    exception: Exception | None = None
    done_event: threading.Event = field(default_factory=threading.Event)


_SENTINEL = object()


class GpuWorkQueue:
    """Single-threaded GPU operation executor.

    Usage::

        q = GpuWorkQueue()
        q.start()
        result = q.submit(some_gpu_function, arg1, arg2)
        q.shutdown()
    """

    def __init__(self, device: torch.device | int | None = None) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._device = self._normalize_device(device)

    @staticmethod
    def _normalize_device(device: torch.device | int | None) -> torch.device | None:
        if device is None:
            return None
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        return device

    def start(self) -> None:
        """Start the GPU worker thread. Call once after construction."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gpu-worker"
        )
        self._thread.start()

    def _run(self) -> None:
        """Pull items off the queue and execute them sequentially."""
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break

            if isinstance(item, GpuWorkItem):
                try:
                    item.result = item.fn(*item.args, **item.kwargs)
                except Exception as e:
                    item.exception = e
                finally:
                    item.done_event.set()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Enqueue a GPU operation and block until it completes.

        Returns the result of ``fn(*args, **kwargs)``.
        Re-raises any exception from *fn* in the caller thread.
        Raises RuntimeError if called after shutdown().
        """
        item = self.submit_async(fn, *args, **kwargs)
        # wait() is outside the lock — holding it would deadlock concurrent submit() callers
        item.done_event.wait()
        if item.exception is not None:
            raise item.exception
        return item.result

    def submit_async(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> GpuWorkItem:
        """Enqueue a GPU operation and return the work item without blocking.

        Callers wait on ``item.done_event`` (with an optional timeout) and
        read ``item.result`` / ``item.exception`` themselves. Used when the
        caller needs to bound the wait or coordinate with other futures.
        Raises RuntimeError if called after shutdown().
        """
        item = GpuWorkItem(fn=fn, args=args, kwargs=kwargs)
        with self._lock:
            if self._closed:
                raise RuntimeError("GpuWorkQueue has been shut down")
            self._queue.put(item)
        return item

    def shutdown(self) -> None:
        """Signal the GPU thread to exit and wait for it."""
        with self._lock:
            self._closed = True
            self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
