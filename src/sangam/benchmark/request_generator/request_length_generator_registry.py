from sangam.benchmark.config import RequestLengthGeneratorType
from sangam.benchmark.request_generator.base_registry import BaseRegistry
from sangam.benchmark.request_generator.fixed_request_length_generator import (
    FixedRequestLengthGenerator,
)
from sangam.benchmark.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from sangam.benchmark.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from sangam.benchmark.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)


class RequestLengthGeneratorRegistry(BaseRegistry):
    pass


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.FIXED, FixedRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
