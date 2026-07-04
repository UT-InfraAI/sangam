# Serving

## Modes

### Colocated

Scheduler + workers run in colocated mode, with each worker performing both prefill and decode.

```bash
uv run python -m sangam.entrypoints.launch \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode colocated \
    --gpus 0
```

### Hybrid (Conditional Disaggregation)

Dedicated prefill workers and colocated workers which initially only process decodes. When prefill workers are overloaded, `--enable-hybrid-prefill-overflow` lets prefills fall through to colocated workers for local prefill+decode.

```bash
uv run python -m sangam.entrypoints.launch \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode hybrid \
    --prefill-gpus 0 \
    --hybrid-colocated-gpus 1 \
```
Overflow is off by default, so hybrid mode runs in a pure disaggregated manner unless you pass `--enable-hybrid-prefill-overflow`.


## Invocation Notes

A list of all CLI arguments with helptexts can be found using the command:

```bash
uv run python -m sangam.entrypoints.launch --help
```

## Key CLI Arguments

### Core

| Argument | Default | Description |
|---|---|---|
| `--mode` | `colocated` | Serving mode: `colocated` or `hybrid` |
| `--model` | `GSAI-ML/LLaDA-8B-Instruct` | HuggingFace model name |
| `--scheduler-port` | `50051` | gRPC port for the scheduler |
| `--gpus` | `0,1` | Comma-separated GPU IDs for colocated workers |
| `--prefill-gpus` | `0,1` | Comma-separated GPU IDs for prefill workers (hybrid) |
| `--hybrid-colocated-gpus` | `1,2,3` | Comma-separated GPU IDs for colocated workers (hybrid) |
| `--base-worker-port` | `20100` | Starting port for worker gRPC servers |
| `--master-addr` | `localhost` | `torch.distributed` master address |
| `--master-port` | `29500` | `torch.distributed` master port |
| `--max-batch-size` | `128` | Max decode steps to batch per forward pass |
| `--block-length` | `32` | Tokens per generation block for every request handled by this server |
| `--max-gen-len` | `None` | Fixed prompt+mask length; when set, must be a positive multiple of `--block-length` |
| `--mask-id` | `None` | Token ID for masked positions; falls back to the HF config `mask_token_id` |
| `--colocated-sticky-worker` | `false` | Pin each request to its first-assigned worker |

### Scheduling & topology

| Argument | Default | Description |
|---|---|---|
| `--max-tokens-per-iteration` | `4096` | Colocated mode token budget per scheduler iteration (decode + optional prefill admission) |
| `--max-prefill-tokens-per-batch` | `4096` | Max total token count across all requests in a single prefill forward pass |
| `--prefill-scheduler-policy` | `least_outstanding_prefill_tokens` | Prefill assignment policy: `round_robin`, `least_outstanding_prefill_tokens`, or colocated-only `least_outstanding_requests`, `least_request_length_sum`, `balanced_length_clustering` |
| `--prefill-queue-policy` | `arrival_order` | Per-worker prefill queue ordering: `arrival_order` or `fewest_remaining_blocks` |
| `--decode-scheduler-policy` | `max_free_memory` | Decode assignment policy: `round_robin`, `max_free_memory`, `balanced_length_clustering`, or topology-aware `topology_guarded_memory` in hybrid mode |
| `--decode-grouping-slack-ratio` | `0.10` | Slack ratio for decode `balanced_length_clustering`; workers within `min_projected_sum * (1 + slack)` remain eligible before clustering preference is applied |
| `--kv-fast-pairs` | `0-1,2-3,4-5,6-7` | Undirected fast GPU link pairs for topology-aware decode routing in hybrid mode, for example `0-1,2-3` |
| `--kv-topology-alpha` | `0.0` | Prefer a fast-link decode worker only when its free KV pages are at least `alpha * mem_best.free_pages` |
| `--enable-hybrid-prefill-overflow` / `--no-enable-hybrid-prefill-overflow` | on in hybrid mode | Hybrid-only: allow overloaded prefill workers to overflow to colocated workers for local prefill+decode. On by default in hybrid mode; pass `--no-enable-hybrid-prefill-overflow` to queue such requests as pending instead |
| `--prefill-overload-threshold` | `16384` | Hybrid-only: outstanding prefill tokens per worker above which requests overflow to colocated workers |

### KV cache

| Argument | Default | Description |
|---|---|---|
| `--kv-page-size` | `16` | Tokens per KV cache page |
| `--kv-max-pages` | auto (per model) | Max KV cache pages per decode worker. Auto-selected from the model's HF architecture when omitted (Dream `49152`, LLaDA `5632`); pass an explicit value to override |

### CUDA graphs

| Argument | Default | Description |
|---|---|---|
| `--enable-cuda-graphs` / `--no-enable-cuda-graphs` | on | Capture and replay CUDA graphs for decode-only batches. On by default; pass `--no-enable-cuda-graphs` to disable |
| `--cuda-graph-batch-sizes` | `1,2,4,8,16,24,32,40,48,56,64` | Comma-separated decode batch sizes to capture, capped at `--max-batch-size` |

### Metrics

| Argument | Default | Description |
|---|---|---|
| `--metrics-output-dir` | `benchmark_output/<timestamp>` | Directory for CSV and plot output |
| `--disable-metrics` | `false` | Disable metrics collection |
| `--enable-individual-batch-metrics` | `false` | Enable per-batch raw CSV export (`worker_batch_metrics.csv`) |

Additional operation-level metrics flags (`--enable-operation-metrics`, `--op-metrics-layer-id`, `--export-partial-metrics`) are covered in the metrics doc.

## Generation Parameters

| Parameter | Description |
|---|---|
| `gen_length` | Total tokens to generate; must be divisible by the server `--block-length` |
| `temperature` | Sampling temperature (0.0 = greedy via Gumbel max) |
| `unmasking_strategy` | `random`, `conf_threshold`, `conf_quota`, or `conf_dynamic` |
| `confidence_threshold` | Required when `unmasking_strategy=conf_threshold` |
| `fixed_unmask_quota` | Required when `unmasking_strategy=conf_quota` |
| `dynamic_unmask_factor` | Required when `unmasking_strategy=conf_dynamic` |
| `request_seed` | Optional stable sampling seed (a top-level request field, not part of `sampling_parameters`); if omitted, the scheduler derives a deterministic fallback from request contents and sampling parameters |

## Request Submission

Use [scripts/submit_request.py](../scripts/submit_request.py) as the minimal client. A request consists of:

- `prompt_token_ids`: fully tokenized prompt payload
- `gen_length`: number of completion tokens to generate; it must be divisible by the server `--block-length`
- `request_seed`: optional top-level request field (not nested under `sampling_parameters`) carrying a per-request sampling seed; identical explicit seeds reproduce the same nonzero-temperature sampling across schedulers
- `sampling_parameters`: nested sampling config carrying `temperature`, `unmasking_strategy`, and any strategy-specific optional fields

If `request_seed` is omitted, the scheduler hashes the request payload, sampling
parameters, and server block length to derive a deterministic fallback seed
before the first worker enqueue. That means duplicate implicit requests
intentionally share randomness within a given server configuration.

Polling returns the full current sequence plus a string status. The scheduler currently uses these statuses:

- `PENDING`
- `PREFILLING`
- `WAITING_DECODE`
- `DECODING`
- `WAITING_NEXT_BLOCK`
- `COMPLETED`
- `ERROR`
- `NOT_FOUND` for an unknown request id

Treat `COMPLETED` and `ERROR` as terminal states.

## Troubleshooting

If launch succeeds but requests do not complete, check these first:

- Python/CUDA environment: this project requires Python 3.13, CUDA 12.8, and compatible GPU drivers
- Dependency resolution: `uv sync` must be able to use the configured FlashInfer package source from `pyproject.toml`
- Ports: ensure `--scheduler-port` and `--base-worker-port` are free and do not overlap with other local runs
- Worker topology: hybrid mode needs both `--prefill-gpus` and `--hybrid-colocated-gpus`; colocated mode needs `--gpus`
- Scheduler readiness: if workers have not registered yet, requests can remain pending
- CUDA/NCCL issues: a worker process can start but fail to make progress if inter-GPU communication is misconfigured
- Terminal status: inspect `ERROR` responses from `Poll`; `NOT_FOUND` means the request id is unknown to the scheduler
