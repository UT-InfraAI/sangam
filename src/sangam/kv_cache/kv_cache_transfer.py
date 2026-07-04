"""GPU-to-GPU KV cache transfer using torch.distributed (NCCL backend).

All worker processes (prefill + decode) join a single process group at startup.
KV cache and logits tensors are transferred via point-to-point NCCL send/recv,
which uses NVLink/PCIe on same-node and InfiniBand/RoCE cross-node — same API.
"""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from sangam.logger import init_logger

logger = init_logger(__name__)


def init_process_group(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
    device_id: int | None = None,
) -> None:
    """Initialize the NCCL process group for all workers."""
    import os

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    kwargs = {"backend": "nccl", "rank": rank, "world_size": world_size}
    if device_id is not None:
        kwargs["device_id"] = device_id
    dist.init_process_group(**kwargs)
    logger.info(f"Initialized NCCL process group: rank={rank}, world_size={world_size}")


def send_kv_layer_async(
    key: torch.Tensor,
    value: torch.Tensor,
    dst_rank: int,
    stream: torch.cuda.Stream | None = None,
) -> list[dist.Work]:
    """Asynchronously send one KV layer (K then V) to a destination rank."""
    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx:
        work_k = dist.isend(key.contiguous(), dst=dst_rank)
        work_v = dist.isend(value.contiguous(), dst=dst_rank)
    return [work_k, work_v]


def send_paged_kv_layer_async(
    kv_layer: torch.Tensor,
    page_ids: list[int],
    dst_rank: int,
    stream: torch.cuda.Stream | None = None,
) -> list[dist.Work]:
    """Asynchronously send one paged KV layer as a single packed page bundle."""
    if not page_ids:
        return []

    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx:
        page_idx = torch.tensor(page_ids, dtype=torch.long, device=kv_layer.device)
        packed_pages = kv_layer.index_select(0, page_idx).contiguous()
        return [dist.isend(packed_pages, dst=dst_rank)]


def recv_kv_layer(
    src_rank: int,
    device: torch.device,
    batch_size: int,
    num_kv_heads: int,
    seq_length: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Receive one KV layer (K then V) from a source rank."""
    shape = (batch_size, num_kv_heads, seq_length, head_dim)
    k = torch.empty(shape, dtype=dtype, device=device)
    v = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(k, src=src_rank)
    dist.recv(v, src=src_rank)
    return k, v


def recv_paged_kv_layer(
    kv_layer: torch.Tensor,
    page_ids: list[int],
    src_rank: int,
) -> None:
    """Receive one packed paged KV layer and scatter it into destination page slots."""
    if not page_ids:
        return

    page_idx = torch.tensor(page_ids, dtype=torch.long, device=kv_layer.device)
    recv_buffer = torch.empty(
        (len(page_ids), *kv_layer.shape[1:]),
        dtype=kv_layer.dtype,
        device=kv_layer.device,
    )
    dist.recv(recv_buffer, src=src_rank)
    kv_layer.index_copy_(0, page_idx, recv_buffer)


def recv_paged_kv_layer_drain(
    template: torch.Tensor,
    num_pages: int,
    src_rank: int,
) -> None:
    """Receive a packed paged KV layer into a throwaway buffer and discard.

    Used when the receiver has rejected the transfer (e.g. KV pool exhausted)
    but the matching NCCL send is already in flight on the sender. Without
    draining, the send would block forever on send_stream.synchronize(),
    jamming the per-destination GPU send queue.
    """
    if num_pages <= 0:
        return
    recv_buffer = torch.empty(
        (num_pages, *template.shape[1:]),
        dtype=template.dtype,
        device=template.device,
    )
    dist.recv(recv_buffer, src=src_rank)


def send_tensor(tensor: torch.Tensor, dst_rank: int) -> None:
    """Send a single tensor (e.g. logits) to a destination rank."""
    dist.send(tensor.contiguous(), dst=dst_rank)


def recv_tensor(
    src_rank: int,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Receive a single tensor from a source rank."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank)
    return tensor
