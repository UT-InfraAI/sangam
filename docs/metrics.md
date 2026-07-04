# Metrics emitted by sangam

sangam records per-request, per-block, per-batch, per-worker, and per-scheduler metrics during a run and dumps them as CSV files and plots when the server shuts
down. This document describes the output structure and every metric emitted.

## How metrics are produced

All metrics flow through the `MetricsStore` singleton (`src/sangam/metrics/metrics_store.py`). Schedulers and workers call its `on_*` hooks during the run; `MetricsStore.plot()` writes everything to disk at shutdown. Behavior is controlled by CLI flags:

- `--metrics-output-dir` (default `benchmark_output`): destination directory.
- `--disable-metrics`: turn collection off entirely.
- `--enable-individual-batch-metrics` (default on): also emit one row per batch to `worker_batch_metrics.csv`; without it, only aggregated batch CDFs are kept.
- `--enable-operation-metrics` (default off): additionally profile per-operation (attention / MLP / QKV) times, sampled on a single layer and scaled to the full
  model.

## Output directory layout

```
{metrics_output_dir}/
├── request_metrics.csv          # one row per request (time distributions + histograms)
├── block_metrics.csv            # one row per block (block-level times)
├── worker_batch_metrics.csv     # one row per batch (only with --enable-individual-batch-metrics)
├── worker_timeline.csv          # one row per worker state interval
├── request_plots/               # request/block CDFs, histograms, completion time series (.csv + .png)
└── worker_plots/                # per-worker batch CDFs, time series, worker state/utilization (.csv + .png)
```

CSV conventions:

- `request_metrics.csv` / `block_metrics.csv` merge all metrics on the `Request Id` / `Block Id` key, one row per request / block, values to 4 decimals. Block ids
  use the form `{request_id}_b{block_index}`.
- CDF CSVs (in `request_plots/` and `worker_plots/`) have a `cdf` column (quantile fraction, 0..1) plus the metric column; worker CDFs add a `worker_id` column. CDFs are tracked with DDSketch (`relative_accuracy=0.001`).
- Time-series CSVs have a `Time (s)` column (rebased so `t=0` is the first request arrival) plus the metric column and a `worker_id` / `queue` label. Series are downsampled to at most 1001 points.


## Request metrics

Emitted per request to `request_metrics.csv`; each also gets a CDF in `request_plots/`.

| Metric | Meaning |
| --- | --- |
| `request_e2e_time` | End-to-end latency, `complete_time - submit_time`. |
| `request_e2e_time_normalized` | `request_e2e_time` divided by generated tokens. |
| `request_e2e_time_excl_first_block_queue` | E2E latency minus the first block's prefill queue wait (excludes initial admission delay). |
| `request_e2e_time_excl_first_block_queue_normalized` | Above, divided by generated tokens. |
| `request_first_block_queue` | Queue wait of the first block before prefill (admission delay). |
| `request_scheduling_delay` | Total queue wait across all blocks (prefill + decode). |
| `request_scheduling_delay_prefill` | Queue wait attributed to prefill phases. |
| `request_scheduling_delay_decode` | Queue wait attributed to decode phases. |
| `request_execution_time` | `request_e2e_time` minus total queue wait. |
| `request_execution_time_normalized` | `request_execution_time` divided by generated tokens. |
| `request_prefill_time` | Total prefill execution time across blocks. |
| `request_kv_transfer_time` | Total KV-cache transfer time across blocks. |
| `request_kv_transfer_time_nonoverlapped` | KV transfer time not overlapped with prefill (the part that adds to latency). |
| `request_decode_time` | Total decode execution time across blocks. |
| `request_decode_time_normalized` | `request_decode_time` divided by generated tokens. |
| `request_unaccounted_time` | E2E time minus the sum of prefill, decode, queue wait, and non-overlapped KV transfer. |
| `request_inter_arrival_delay` | Time between consecutive request arrivals. |
| `request_num_prompt_tokens` | Prompt (prefill) tokens in the request. |
| `request_num_gen_tokens` | Generated (decode) tokens in the request. |
| `request_num_blocks` | Number of blocks the request is generated in. |
| `request_num_forward_passes` | Total forward passes (prefill + decode) for the request. |
| `request_time_between_tokens` | Inter-token latency observed by the user. When a forward pass unmasks several tokens at once, the gap is attributed to the first and the rest are recorded as 0. |
| `request_tokens_unmasked_per_forward_pass` | Number of tokens unmasked per decode forward pass. |
| `request_arrival` / `request_completion` (`*_time_series` in `request_plots/`) | Cumulative count of arrived and completed requests over time (step plot); the completion curve's slope gives request throughput. |

## Block metrics

Per block in `block_metrics.csv`, with CDFs in `request_plots/`.

| Metric | Meaning |
| --- | --- |
| `block_prefill_time` | Prefill execution duration of the block. |
| `block_kv_transfer_time_nonoverlapped` | Non-overlapped KV-transfer duration of the block. |
| `block_decode_time` | Decode execution duration of the block. |
| `block_total_time` | Total wall-clock duration of the block from start to finish. |

## Batch metrics

Aggregated per worker (`worker_id/worker_type`) as CDFs in `worker_plots/`; also per batch in `worker_batch_metrics.csv` when individual batch metrics are enabled.

| Metric | Meaning |
| --- | --- |
| `batch_num_tokens` | Total tokens in the batch (`prompt_len + gen_len`). |
| `batch_prompt_len` | Prompt tokens in the batch. |
| `batch_gen_len` | Generated tokens in the batch. |
| `batch_num_unmasked_tokens` | Tokens unmasked by this batch. |
| `batch_size` | Number of requests in the batch. |
| `batch_decode_length_std` | Std. dev. of decode sequence lengths in the batch (recorded only when the batch has at least two decode requests). |
| `batch_execution_time` | Batch execution time, `batch_end - batch_start`. |
| `batch_sampling_time` | Time spent in token sampling for the batch. |
| `batch_token_throughput` | `batch_num_tokens / batch_execution_time`. |
| `inter_batch_delay` | Idle gap between the previous batch's end and this batch's start. |

### Operation metrics

With `--enable-operation-metrics`, each batch row in `worker_batch_metrics.csv` gains `batch_op_attn_time`, `batch_op_mlp_time`, and `batch_op_qkv_time`: the attention, MLP, and QKV-projection times. These are sampled on a single (typically middle) layer and scaled to the full model.

## Worker metrics

Per worker as CDFs in `worker_plots/`.

| Metric | Meaning |
| --- | --- |
| `queue_depth_waiting` | Number of requests waiting in the worker's queue. |
| `queue_depth_active` | Number of requests in the active batch. |
| `outstanding_prefill_tokens` | Backlog of prefill tokens assigned to the worker (also exported as a time series, see below). |
| `kv_page_utilization_ratio` | `kv_used_pages / kv_total_pages` for the worker. |
| `decode_length_sum` (`worker_decode_length_sum_time_series`) | Sum of decode sequence lengths per batch over time, per worker. |
| `deficit_tokens` (`worker_deficit_tokens_time_series`) | Decode token deficit over time, per worker. |
| `outstanding_prefill_tokens` (`worker_outstanding_prefill_tokens_time_series`) | Outstanding prefill-token backlog per worker over time; a `scheduler_pending` line adds the scheduler's pending prefill tokens. |
| `scheduler_pending_requests` / `scheduler_decode_ready_requests` (`scheduler_queue_depth_time_series`) | Scheduler queue depths over time: requests pending and requests ready for decode. |


## Worker state timeline and summaries

`on_worker_state` records intervals during which a worker is in a given state (`idle`, `queued`, `busy`; `draining` time is tracked separately).

- `worker_timeline.csv`: one row per state interval, with `worker_id`, `worker_type`, `state`, `start_time`, `end_time`, `duration_s`, `queue_depth_waiting`, and `queue_depth_active`.
- `worker_plots/worker_state_time.csv`: per-worker total time in each state (`busy_time_s`, `queued_time_s`, `idle_time_s`, `draining_time_s`).
- `worker_plots/worker_utilization_ratio.csv`: per-worker `utilization_ratio` (`busy_time / total_time`).
- `worker_plots/worker_queue_depth_time_series.csv`: queue depth at each state transition per worker, plus scheduler queue lines.

The per-worker summary additionally computes `num_idle_gaps`, `p50_idle_gap_s`, `p95_idle_gap_s`, `mean_waiting_queue_depth`, and `mean_active_batch_size`.
