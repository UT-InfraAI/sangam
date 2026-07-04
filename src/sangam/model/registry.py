"""Architecture-keyed model registry.

Maps the `architectures[0]` field of a HuggingFace `config.json` to the
(config_class, model_class) pair that sangam uses to serve the checkpoint.
"""

from __future__ import annotations

from sangam.model.dream import DreamConfig, DreamModel
from sangam.model.llada import LLaDAConfig, LLaDAModelLM


_REGISTRY: dict[str, tuple[type, type]] = {
    "LLaDAModelLM": (LLaDAConfig, LLaDAModelLM),
    "DreamModel": (DreamConfig, DreamModel),
}


def resolve_model_classes(architectures: list[str]) -> tuple[type, type]:
    if not architectures:
        raise ValueError("config.json has no `architectures` entry")
    arch = architectures[0]
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unsupported architecture {arch!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[arch]
