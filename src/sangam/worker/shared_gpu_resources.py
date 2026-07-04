"""Shared long-lived GPU allocations for worker backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sangam.kv_cache.paged_kv_cache import PagedKVPool


@dataclass
class WorkerSharedGpuResources:
    """Reusable GPU allocations that can outlive a specific backend role."""

    kv_pool: PagedKVPool
    flashinfer_workspace: torch.Tensor
    prefill_transfer_stream: torch.cuda.Stream | None
    decode_receive_stream: torch.cuda.Stream | None


def create_worker_shared_gpu_resources(
    model: torch.nn.Module,
    device: torch.device,
    kv_page_size: int,
    kv_max_pages: int,
    kv_dtype: torch.dtype,
    *,
    zero_init: bool,
) -> WorkerSharedGpuResources:
    """Allocate the long-lived paged KV state for a worker process."""
    if hasattr(model, "model") and hasattr(model.model, "config"):
        config = model.model.config
        num_layers = config.n_layers
        num_kv_heads = config.effective_n_kv_heads
        head_dim = config.d_model // config.n_heads
    else:
        num_layers = model.num_layers
        num_kv_heads = model.num_kv_heads
        head_dim = model.head_dim

    return WorkerSharedGpuResources(
        kv_pool=PagedKVPool(
            num_layers=num_layers,
            max_pages=kv_max_pages,
            page_size=kv_page_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=kv_dtype,
            zero_init=zero_init,
        ),
        flashinfer_workspace=torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        ),
        prefill_transfer_stream=(
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        ),
        decode_receive_stream=(
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        ),
    )
