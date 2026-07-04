import time
import torch

from torch.profiler import record_function
from sangam.metrics.utils.timer_stats_store import TimerStatsStore
from sangam.metrics.utils.profile_methods import ProfileMethod


class CudaTimer:
    def __init__(
        self,
        name: str,
        aggregation_fn=sum,
        filter_str=None,
        disabled: bool = False,
        device: torch.device | int | None = None,
    ):
        self._disabled = disabled
        if self._disabled:
            return

        self._name = name
        self._timer_stats_store = TimerStatsStore()
        self._aggregation_fn = aggregation_fn
        self._filter_str = filter_str
        self._device = self._normalize_device(device)

        # For Kineto profiling
        if self._timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=self.handle_trace,
            )
        else:
            self.profiler = None
        # For CUDA Event profiling
        self.start_event = None
        self.end_event = None
        # For Perf Counter profiling
        self.start_time = None
        self.end_time = None
        self._last_elapsed_ms = 0.0

    @staticmethod
    def _normalize_device(device: torch.device | int | None) -> torch.device | None:
        if device is None:
            return None
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        return device

    def _current_device(self) -> torch.device | None:
        if self._device is None:
            return None
        return self._device

    def _synchronize(self) -> None:
        device = self._current_device()
        if device is not None:
            torch.cuda.synchronize(device=device)
            return
        torch.cuda.synchronize()

    def __enter__(self):
        if self._disabled:
            return

        if self._timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:
            self.profiler_function_context = record_function(self._name)
            self.profiler_function_context.__enter__()
        elif self._timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:
            if self._device is not None:
                with torch.cuda.device(self._device):
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.start_event.record()
            else:
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
        elif self._timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler.__enter__()
        elif self._timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:
            self._synchronize()
            self.start_time = time.perf_counter()
        else:
            raise ValueError(
                f"Unknown profile method {self._timer_stats_store.profile_method}"
            )
        return self

    def handle_trace(self, trace):
        events = trace.events()

        if self._filter_str:
            events = [e for e in events if e.name.startswith(self._filter_str)]

        total_cuda_time = self._aggregation_fn([e.cuda_time_total for e in events])
        self._timer_stats_store.record_time(
            self._name, total_cuda_time * 1e-3
        )  # convert to ms

    def __exit__(self, *args):
        if self._disabled:
            return

        if self._timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:
            self.profiler_function_context.__exit__(*args)
        elif self._timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:
            if self._device is not None:
                with torch.cuda.device(self._device):
                    self.end_event = torch.cuda.Event(enable_timing=True)
                    self.end_event.record()
                    self.end_event.synchronize()
            else:
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.end_event.record()
                self.end_event.synchronize()
            self._last_elapsed_ms = self.start_event.elapsed_time(self.end_event)
            self._timer_stats_store.record_time(
                self._name, [self.start_event, self.end_event]
            )
        elif self._timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler.__exit__(*args)
        elif self._timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:
            self._synchronize()
            self.end_time = time.perf_counter()
            self._last_elapsed_ms = (self.end_time - self.start_time) * 1e3
            self._timer_stats_store.record_time(
                self._name, (self.end_time - self.start_time) * 1e3
            )  # convert to ms
        else:
            raise ValueError(
                f"Unknown profile method {self._timer_stats_store.profile_method}"
            )

    @property
    def elapsed_ms(self) -> float:
        if self._disabled:
            return 0.0
        return self._last_elapsed_ms


class DurationTimer:
    def __init__(
        self,
        name: str,
        use_cuda: bool,
        device: torch.device | int | None = None,
    ):
        self._use_cuda = use_cuda
        self._cuda_timer = CudaTimer(name, device=device) if use_cuda else None
        self._start_time: float | None = None
        self.elapsed_s = 0.0

    def __enter__(self):
        if self._cuda_timer is not None:
            self._cuda_timer.__enter__()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._cuda_timer is not None:
            self._cuda_timer.__exit__(*args)
            self.elapsed_s = self._cuda_timer.elapsed_ms / 1e3
            return
        if self._start_time is not None:
            self.elapsed_s = max(0.0, time.perf_counter() - self._start_time)
