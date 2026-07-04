import enum


class ProfileMethod(enum.Enum):
    CUDA_EVENT = "cuda_event"
    KINETO = "kineto"
    PERF_COUNTER = "perf_counter"
    RECORD_FUNCTION = "record_function"
