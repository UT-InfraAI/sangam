from typing import List

from sangam.benchmark.config import SyntheticRequestGeneratorConfig
from sangam.benchmark.entities import Request
from sangam.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)
from sangam.benchmark.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from sangam.benchmark.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from sangam.benchmark.utils.random import set_seeds


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: SyntheticRequestGeneratorConfig):
        super().__init__(config)

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config,
        )
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
        )

    def _generate_next_request(self, last_arrived_at: float) -> Request:
        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        (
            prefill_tokens,
            decode_tokens,
        ) = self.request_length_generator.get_next_num_tokens()

        if prefill_tokens is None or decode_tokens is None:
            return None

        prompt_payload = {}
        payload_getter = getattr(
            self.request_length_generator,
            "get_next_request_payload",
            None,
        )
        if payload_getter is not None:
            payload = payload_getter()
            if payload is not None:
                prompt_payload = payload

        return Request(
            arrived_at=arrived_at,
            prompt_len=int(prefill_tokens),
            gen_len=int(decode_tokens),
            **prompt_payload,
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # first priority is duration
        if self.config.duration is not None:
            while current_time < self.config.duration:
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        elif self.config.num_requests is not None:
            for _ in range(self.config.num_requests):
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        else:
            # trace-driven: generate until exhausted
            while True:
                request = self._generate_next_request(current_time)
                if request is None:
                    break
                current_time = request.arrived_at
                requests.append(request)

        return requests

    def generate_requests(self) -> List[Request]:
        assert self.config.num_requests or self.config.duration

        set_seeds(self.config.seed)

        requests = self._generate_requests()

        # sort requests by arrival time
        requests.sort(key=lambda x: x.arrived_at)
        # remove any requests that arrived after the time limit
        if self.config.duration is not None:
            requests = [
                request
                for request in requests
                if request.arrived_at < self.config.duration
            ]

        return requests
