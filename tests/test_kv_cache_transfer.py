from types import SimpleNamespace

import torch

from sangam.kv_cache.kv_cache_transfer import (
    recv_paged_kv_layer,
    send_paged_kv_layer_async,
)


def test_send_paged_kv_layer_async_sends_single_packed_tensor(monkeypatch) -> None:
    sent: dict[str, object] = {}
    kv_layer = torch.arange(5 * 2 * 3 * 2 * 4, dtype=torch.float32).reshape(
        5, 2, 3, 2, 4
    )

    def _fake_isend(tensor: torch.Tensor, dst: int) -> object:
        sent["tensor"] = tensor.clone()
        sent["dst"] = dst
        return SimpleNamespace(dst=dst)

    monkeypatch.setattr("torch.distributed.isend", _fake_isend)

    works = send_paged_kv_layer_async(kv_layer=kv_layer, page_ids=[3, 1], dst_rank=7)

    assert len(works) == 1
    assert sent["dst"] == 7
    assert torch.equal(sent["tensor"], kv_layer[[3, 1]])


def test_recv_paged_kv_layer_receives_single_packed_tensor_and_scatter_copies(
    monkeypatch,
) -> None:
    kv_layer = torch.zeros((5, 2, 3, 2, 4), dtype=torch.float32)
    packed = torch.arange(2 * 2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 2, 3, 2, 4)
    recv_shapes: list[tuple[int, ...]] = []

    def _fake_recv(tensor: torch.Tensor, src: int) -> None:
        recv_shapes.append(tuple(tensor.shape))
        tensor.copy_(packed)

    monkeypatch.setattr("torch.distributed.recv", _fake_recv)

    recv_paged_kv_layer(kv_layer=kv_layer, page_ids=[4, 0], src_rank=2)

    assert recv_shapes == [(2, 2, 3, 2, 4)]
    assert torch.equal(kv_layer[4], packed[0])
    assert torch.equal(kv_layer[0], packed[1])
    assert torch.equal(kv_layer[1], torch.zeros_like(kv_layer[1]))
