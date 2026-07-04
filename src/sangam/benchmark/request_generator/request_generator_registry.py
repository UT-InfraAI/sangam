from sangam.benchmark.config import RequestGeneratorType
from sangam.benchmark.request_generator.base_registry import BaseRegistry
from sangam.benchmark.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from sangam.benchmark.request_generator.trace_request_generator import (
    TraceRequestGenerator,
)


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
