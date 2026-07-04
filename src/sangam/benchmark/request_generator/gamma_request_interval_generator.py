import random

from sangam.benchmark.config import GammaRequestIntervalGeneratorConfig
from sangam.benchmark.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class GammaRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, config: GammaRequestIntervalGeneratorConfig):
        super().__init__(config)

        cv = self.config.cv
        self.qps = self.config.qps
        self.gamma_shape = 1.0 / (cv**2)

    def get_next_inter_request_time(self) -> float:
        gamma_scale = 1.0 / (self.qps * self.gamma_shape)
        return random.gammavariate(self.gamma_shape, gamma_scale)
