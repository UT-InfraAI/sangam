"""Tests for GpuWorkQueue shutdown race safety."""

import threading

import pytest
import torch

from sangam.metrics.utils.cuda_timer import DurationTimer
from sangam.worker.gpu_queue import GpuWorkQueue


def test_normal_submit_and_shutdown_work():
    q = GpuWorkQueue()
    q.start()
    result = q.submit(lambda x: x * 2, 21)
    assert result == 42
    q.shutdown()


def test_submit_after_shutdown_raises():
    """submit() after shutdown() must raise RuntimeError immediately, not hang."""
    q = GpuWorkQueue()
    q.start()
    q.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        q.submit(lambda: None)


def test_queue_thread_binds_configured_cuda_device(monkeypatch) -> None:
    set_device_calls = []

    monkeypatch.setattr(
        "sangam.worker.gpu_queue.torch.cuda.set_device",
        lambda device: set_device_calls.append(device),
    )

    q = GpuWorkQueue(device=torch.device("cuda:3"))
    q.start()
    assert q.submit(lambda: "ok") == "ok"
    q.shutdown()

    assert set_device_calls == [torch.device("cuda:3")]


def test_submit_concurrent_with_shutdown_raises_or_completes():
    """Concurrent submit and shutdown: must either complete or raise, never hang."""
    TIMEOUT = 5.0

    for _ in range(50):
        q = GpuWorkQueue()
        q.start()

        barrier = threading.Barrier(2)
        result_holder: list = []
        exception_holder: list = []

        def do_submit():
            barrier.wait()
            try:
                result_holder.append(q.submit(lambda: "ok"))
            except RuntimeError:
                exception_holder.append("raised")

        def do_shutdown():
            barrier.wait()
            q.shutdown()

        t1 = threading.Thread(target=do_submit)
        t2 = threading.Thread(target=do_shutdown)
        t1.start()
        t2.start()
        t1.join(timeout=TIMEOUT)
        t2.join(timeout=TIMEOUT)

        assert not t1.is_alive(), "submit() hung"
        assert not t2.is_alive(), "shutdown() hung"
        # Exactly one of: completed successfully or raised RuntimeError
        assert len(result_holder) + len(exception_holder) == 1


def test_duration_timer_uses_explicit_cuda_device_for_events(monkeypatch) -> None:
    device_entries = []

    class _DummyDeviceContext:
        def __init__(self, device) -> None:
            self._device = device

        def __enter__(self):
            device_entries.append(self._device)
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyEvent:
        def __init__(self, enable_timing: bool = True) -> None:
            self.enable_timing = enable_timing

        def record(self) -> None:
            return None

        def synchronize(self) -> None:
            return None

        def elapsed_time(self, other) -> float:
            return 12.5

    monkeypatch.setattr(
        "sangam.metrics.utils.cuda_timer.torch.cuda.device",
        lambda device: _DummyDeviceContext(device),
    )
    monkeypatch.setattr(
        "sangam.metrics.utils.cuda_timer.torch.cuda.Event",
        _DummyEvent,
    )

    timer = DurationTimer(
        "worker_sampling_decode",
        use_cuda=True,
        device=torch.device("cuda:2"),
    )
    with timer:
        pass

    assert device_entries == [torch.device("cuda:2"), torch.device("cuda:2")]
    assert timer.elapsed_s == pytest.approx(0.0125)


def test_receive_kv_cache_returns_failure_when_recv_queue_shut_down():
    """ReceiveKVCache returns success=False when the recv queue is already shut down."""
    from types import SimpleNamespace

    from sangam.worker.colocated_worker import ColocatedWorkerServicer

    servicer = ColocatedWorkerServicer.__new__(ColocatedWorkerServicer)

    recv_queue = GpuWorkQueue()
    recv_queue.start()
    recv_queue.shutdown()
    servicer._recv_queues = {0: recv_queue}
    servicer._recv_stream_pool = {0: None}

    context = SimpleNamespace(
        set_code=lambda code: None,
        set_details=lambda details: None,
    )

    done = threading.Event()
    result_holder: list = []

    def call():
        resp = servicer.ReceiveKVCache(SimpleNamespace(request_id="req-x"), context)
        result_holder.append(resp)
        done.set()

    t = threading.Thread(target=call)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "ReceiveKVCache hung after recv queue shutdown"
    assert len(result_holder) == 1
    assert result_holder[0].success is False
