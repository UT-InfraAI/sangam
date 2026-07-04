"""Shared model loading and distributed initialization utilities."""

import json
import os

import torch
from huggingface_hub import hf_hub_download

from sangam.kv_cache.kv_cache_transfer import init_process_group
from sangam.logger import init_logger
from sangam.model.registry import resolve_model_classes

logger = init_logger(__name__)


def _read_hf_config(model_name: str) -> dict:
    """Read config.json for a HF hub id or a local checkpoint directory."""
    if os.path.isdir(model_name):
        path = os.path.join(model_name, "config.json")
    else:
        path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(path, "r") as f:
        return json.load(f)


def read_mask_token_id(model_name: str) -> int:
    """Read the diffusion mask token id from a model's HF config.json."""
    config_dict = _read_hf_config(model_name)
    if "mask_token_id" not in config_dict:
        raise ValueError(
            f"Model {model_name!r} config.json has no `mask_token_id`; "
            "sangam only supports diffusion LMs that expose one."
        )
    return int(config_dict["mask_token_id"])


# Per-model-family KV cache page budgets. Keyed on the checkpoint's HF
# `architectures[0]` field, consistent with `sangam.model.registry`.
_DEFAULT_KV_MAX_PAGES_BY_ARCH = {
    "DreamModel": 49152,
    "LLaDAModelLM": 5632,
}


def read_default_kv_max_pages(model_name: str) -> int:
    """Resolve the per-model default KV cache page budget from HF config.json."""
    config_dict = _read_hf_config(model_name)
    architectures = config_dict.get("architectures", [])
    if not architectures:
        raise ValueError(f"Model {model_name!r} config.json has no `architectures`")
    arch = architectures[0]
    if arch not in _DEFAULT_KV_MAX_PAGES_BY_ARCH:
        raise ValueError(
            f"No default kv_max_pages for architecture {arch!r}; "
            f"pass --kv-max-pages explicitly. Known: "
            f"{sorted(_DEFAULT_KV_MAX_PAGES_BY_ARCH)}"
        )
    return _DEFAULT_KV_MAX_PAGES_BY_ARCH[arch]


def load_model(
    model_name: str,
    gpu_id: int,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Load a registered diffusion LM onto the specified GPU.

    Dispatches via the model registry keyed on the checkpoint's HF
    `architectures[0]` field. Returns the model in eval mode with inference
    optimizations (fused QKV and fused MLP gate/up when supported by the
    architecture).
    """
    device = torch.device(f"cuda:{gpu_id}")
    config_dict = _read_hf_config(model_name)
    _, model_cls = resolve_model_classes(config_dict.get("architectures", []))
    logger.debug(
        f"Loading model {model_name} (arch={model_cls.__name__}) on {device} (dtype={dtype})"
    )

    model = model_cls.from_pretrained(model_name, dtype=dtype).to(device).eval()
    model.fuse_qkv()
    if hasattr(model, "fuse_ff_gate_up"):
        model.fuse_ff_gate_up()

    logger.info(
        f"Loaded model {model_name} (arch={model_cls.__name__}) on {device} (dtype={dtype})"
    )
    return model


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    device_id: int | None,
) -> None:
    """Initialize torch.distributed for NCCL KV cache transfers.

    Pass ``device_id`` (the local CUDA device index) so PyTorch eagerly
    builds the per-device NCCL communicator at process-group init. Without
    it, the comm is created lazily on first use, and concurrent first-use
    across send/recv queues can hit ``ProcessGroupNCCL`` paths that look up
    the comm before init completes (``Communicators not populated in
    cache!``).
    """
    init_process_group(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device_id=device_id,
    )
