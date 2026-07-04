from enum import Enum
from typing import Any


class BaseRegistry:
    """Simple registry mapping enum keys to implementation classes."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: Enum, implementation_class: Any) -> None:
        if key in cls._registry:
            return
        cls._registry[key] = implementation_class

    @classmethod
    def get(cls, key: Enum, *args, **kwargs) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered in {cls.__name__}")
        return cls._registry[key](*args, **kwargs)
