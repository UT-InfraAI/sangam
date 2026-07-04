from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sangam.proto import sangam_pb2


@dataclass
class SamplingParameters:
    temperature: float = 0.0
    unmasking_strategy: str = "random"
    confidence_threshold: float | None = None
    fixed_unmask_quota: int | None = None
    dynamic_unmask_factor: float | None = None

    def __post_init__(self) -> None:
        if not self.unmasking_strategy:
            self.unmasking_strategy = "random"
        self.validate()

    @classmethod
    def default(cls) -> "SamplingParameters":
        return cls()

    @classmethod
    def from_proto(
        cls, proto: "sangam_pb2.SamplingParameters | None"
    ) -> "SamplingParameters":
        if proto is None:
            return cls.default()
        return cls(
            temperature=proto.temperature,
            unmasking_strategy=proto.unmasking_strategy,
            confidence_threshold=(
                proto.confidence_threshold
                if proto.HasField("confidence_threshold")
                else None
            ),
            fixed_unmask_quota=(
                proto.fixed_unmask_quota
                if proto.HasField("fixed_unmask_quota")
                else None
            ),
            dynamic_unmask_factor=(
                proto.dynamic_unmask_factor
                if proto.HasField("dynamic_unmask_factor")
                else None
            ),
        )

    def to_proto(self) -> "sangam_pb2.SamplingParameters":
        from sangam.proto import sangam_pb2

        proto = sangam_pb2.SamplingParameters(
            temperature=self.temperature,
            unmasking_strategy=self.unmasking_strategy,
        )
        if self.confidence_threshold is not None:
            proto.confidence_threshold = self.confidence_threshold
        if self.fixed_unmask_quota is not None:
            proto.fixed_unmask_quota = self.fixed_unmask_quota
        if self.dynamic_unmask_factor is not None:
            proto.dynamic_unmask_factor = self.dynamic_unmask_factor
        return proto

    def validate(self) -> None:
        allowed = {"random", "conf_threshold", "conf_quota", "conf_dynamic"}
        if self.unmasking_strategy not in allowed:
            raise ValueError(f"Unknown unmasking_strategy '{self.unmasking_strategy}'")
        if (
            self.unmasking_strategy == "conf_threshold"
            and self.confidence_threshold is None
        ):
            raise ValueError(
                "confidence_threshold is required when unmasking_strategy='conf_threshold'"
            )
        if self.unmasking_strategy == "conf_quota" and self.fixed_unmask_quota is None:
            raise ValueError(
                "fixed_unmask_quota is required when unmasking_strategy='conf_quota'"
            )
        if (
            self.unmasking_strategy == "conf_dynamic"
            and self.dynamic_unmask_factor is None
        ):
            raise ValueError(
                "dynamic_unmask_factor is required when unmasking_strategy='conf_dynamic'"
            )
        if self.fixed_unmask_quota is not None and self.fixed_unmask_quota < 0:
            raise ValueError("fixed_unmask_quota must be non-negative")
