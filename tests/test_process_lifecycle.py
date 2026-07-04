import signal
import subprocess

import sangam.process_lifecycle as process_lifecycle


class _FakeProc:
    def __init__(
        self, pid: int, poll_sequence: list[None | int], wait_outcomes: list[object]
    ):
        self.pid = pid
        self._poll_sequence = poll_sequence
        self._wait_outcomes = wait_outcomes

    def poll(self) -> None | int:
        if len(self._poll_sequence) > 1:
            return self._poll_sequence.pop(0)
        return self._poll_sequence[0]

    def wait(self, timeout: float):
        outcome = self._wait_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_stop_process_group_noop_if_exited(monkeypatch):
    proc = _FakeProc(pid=123, poll_sequence=[0], wait_outcomes=[])
    called = {"killpg": 0}
    monkeypatch.setattr(
        process_lifecycle.os,
        "killpg",
        lambda pgid, sig: called.__setitem__("killpg", 1),
    )
    process_lifecycle.stop_process_group(
        process=proc,
        process_name="proc",
        term_timeout_seconds=5.0,
        kill_timeout_seconds=5.0,
    )
    assert called["killpg"] == 0


def test_stop_process_group_exits_on_sigterm(monkeypatch):
    proc = _FakeProc(
        pid=123,
        poll_sequence=[None],
        wait_outcomes=[None],
    )
    monkeypatch.setattr(process_lifecycle.os, "getpgid", lambda pid: 77)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        process_lifecycle.os,
        "killpg",
        lambda pgid, sig: sent.append((pgid, sig)),
    )

    process_lifecycle.stop_process_group(
        process=proc,
        process_name="proc",
        term_timeout_seconds=5.0,
        kill_timeout_seconds=5.0,
    )

    assert sent == [(77, signal.SIGTERM)]


def test_stop_process_group_escalates_to_sigkill(monkeypatch):
    proc = _FakeProc(
        pid=123,
        poll_sequence=[None, None, None, 0],
        wait_outcomes=[
            subprocess.TimeoutExpired(cmd="launch", timeout=5),
            None,
        ],
    )
    monkeypatch.setattr(process_lifecycle.os, "getpgid", lambda pid: 77)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        process_lifecycle.os,
        "killpg",
        lambda pgid, sig: sent.append((pgid, sig)),
    )

    process_lifecycle.stop_process_group(
        process=proc,
        process_name="proc",
        term_timeout_seconds=5.0,
        kill_timeout_seconds=5.0,
    )

    assert sent == [
        (77, signal.SIGTERM),
        (77, signal.SIGKILL),
    ]


def test_install_termination_handler_raises_keyboard_interrupt(monkeypatch):
    captured: dict[int, object] = {}

    def fake_signal(sig, handler):
        captured[sig] = handler
        return None

    monkeypatch.setattr(process_lifecycle.signal, "signal", fake_signal)

    process_lifecycle.install_termination_handler()

    assert set(captured.keys()) == {signal.SIGTERM, signal.SIGINT}
    for handler in captured.values():
        try:
            handler(signal.SIGTERM, None)
        except KeyboardInterrupt:
            continue
        raise AssertionError("handler should have raised KeyboardInterrupt")
