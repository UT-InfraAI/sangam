"""Helpers for topology-aware decode scheduling."""


def parse_kv_fast_pairs(kv_fast_pairs: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if not kv_fast_pairs.strip():
        return pairs

    for raw_pair in kv_fast_pairs.split(","):
        pair = raw_pair.strip()
        if not pair:
            raise ValueError(
                f"Invalid --kv-fast-pairs entry {raw_pair!r}: empty pair token"
            )
        left, sep, right = pair.partition("-")
        if sep != "-" or not left.strip() or not right.strip():
            raise ValueError(
                f"Invalid --kv-fast-pairs entry {pair!r}: expected GPU pairs like 0-1"
            )
        try:
            gpu_a = int(left.strip())
            gpu_b = int(right.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid --kv-fast-pairs entry {pair!r}: GPU ids must be integers"
            ) from exc
        if gpu_a == gpu_b:
            raise ValueError(
                f"Invalid --kv-fast-pairs entry {pair!r}: self-pairs are not allowed"
            )
        pairs.add((min(gpu_a, gpu_b), max(gpu_a, gpu_b)))
    return pairs


def is_fast_pair(
    fast_pairs: set[tuple[int, int]], prefill_gpu_id: int, decode_gpu_id: int
) -> bool:
    return (min(prefill_gpu_id, decode_gpu_id), max(prefill_gpu_id, decode_gpu_id)) in (
        fast_pairs
    )
