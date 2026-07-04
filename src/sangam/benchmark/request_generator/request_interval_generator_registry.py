from sangam.benchmark.config import RequestIntervalGeneratorType
from sangam.benchmark.request_generator.base_registry import BaseRegistry
from sangam.benchmark.request_generator.gamma_request_interval_generator import (
    GammaRequestIntervalGenerator,
)
from sangam.benchmark.request_generator.poisson_request_interval_generator import (
    PoissonRequestIntervalGenerator,
)
from sangam.benchmark.request_generator.static_request_interval_generator import (
    StaticRequestIntervalGenerator,
)
from sangam.benchmark.request_generator.trace_request_interval_generator import (
    TraceRequestIntervalGenerator,
)


class RequestIntervalGeneratorRegistry(BaseRegistry):
    pass


RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.POISSON, PoissonRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.GAMMA, GammaRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.STATIC, StaticRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.TRACE, TraceRequestIntervalGenerator
)
