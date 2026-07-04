import numpy as np

from sangam.metrics.utils.profile_methods import ProfileMethod
from sangam.metrics.utils.singleton import Singleton


class TimerStatsStore(metaclass=Singleton):
    def __init__(self, profile_method: str = ProfileMethod.CUDA_EVENT.value):
        self._profile_method = ProfileMethod[profile_method.upper()]
        self._timing_stats = {}

    @property
    def profile_method(self):
        return self._profile_method

    def record_time(self, name: str, time):
        if name not in self._timing_stats:
            self._timing_stats[name] = []

        self._timing_stats[name].append(time)

    def clear_stats(self):
        self._timing_stats = {}

    def get_stats(self):
        stats = {}
        for name, times in self._timing_stats.items():
            times = [
                (time if isinstance(time, float) else time[0].elapsed_time(time[1]))
                for time in times
            ]

            stats[name] = {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }

        return stats
