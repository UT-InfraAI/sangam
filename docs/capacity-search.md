# Capacity Search (QPS under SLA)

Capacity search finds the highest request rate (QPS) a configuration can sustain while meeting one or more latency/queue SLAs. It drives the benchmark harness repeatedly at different QPS values, evaluates each trial against the SLA, and reports the maximum passing QPS per job.

```bash
uv run python -m sangam.benchmark.capacity_search.main \
    --config-path path/to/capacity_search.yaml \
    --output-dir capacity_search_output/
```

CLI flags `--max-iterations`, `--min-search-granularity-pct`, and `--max-qps-cap` override the matching `search` settings in the config.

## Config

```yaml
benchmark_base:          # shared BenchmarkConfig fields, merged into every trial
  mode: hybrid
  prefill_gpus: "0"
  hybrid_colocated_gpus: "1"
  num_requests: 200
  interval_type: poisson
  length_type: fixed
  prefill_tokens: 256
  decode_tokens: 64

jobs:                    # one search per job
  - name: hybrid_fixed_short
    start_qps: 2.0       # adaptive search; or use qps_list for a linear sweep

sla:                     # all rules must pass for a trial to count as passing
  - metric: request_scheduling_delay
    quantile: 0.5
    threshold: 2.0
    op: "<="

search:
  max_iterations: 20             # max trials per job (default 20)
  min_search_granularity_pct: 2.5  # bisection stops once bounds are this close (default 2.5)
  max_qps_cap: 64.0              # optional upper bound on the QPS searched
```

`benchmark_base` and `benchmark_overrides` accept any `BenchmarkConfig` field. `launch_server` is rejected: capacity search always launches its own server per trial.

## Jobs: adaptive vs. linear

Each job sets exactly one mode.

- **Adaptive (`start_qps`)**: bisection that first grows QPS exponentially (doubling from `start_qps`) until it finds a failing rate, then binary-searches between the highest passing and lowest failing QPS until it converges (`min_search_granularity_pct`), hits `max_iterations`, or reaches `max_qps_cap`. Requires `sla`.
- **Linear (`qps_list`)**: runs each listed QPS in order and records the highest passing one. `sla` is optional here; without it, every QPS still runs but `passed` and `max_qps_under_sla` are reported as `null`.

A `qps_list` common to several linear jobs can be hoisted to `search.qps_list`; jobs that set neither `start_qps` nor `qps_list` inherit it, and a per-job setting always wins.

```yaml
search:
  qps_list: [7.0, 9.0]

jobs:
  - name: shared_grid_a
    benchmark_overrides: { mode: colocated }
  - name: custom_grid
    qps_list: [4.0, 6.0]
    benchmark_overrides: { mode: hybrid }
```

## SLA rules

Each rule evaluates a metric quantile against a threshold with one of `<=`, `<`, `>=`, `>`, `==`. A trial passes only if every rule passes.

- **Request metrics** (e.g. `request_scheduling_delay`) are read from each trial's `request_metrics.csv`; the rule uses the plain quantile across requests.
- **Queue metrics** (`queue_depth_waiting`) are read from `worker_timeline.csv`, aggregated as a `duration_s`-weighted quantile per worker, and evaluated against the worst (max) worker.

```yaml
sla:
  - metric: queue_depth_waiting
    quantile: 0.95
    threshold: 4
    op: "<="
```

## Trials, timeouts, and caching

Each trial runs as its own benchmark subprocess and inherits `benchmark_timeout` from `benchmark_base` (default 900s; `0` disables it). On timeout the subprocess exits with code 2; the trial is recorded as timed out and the job's search stops early. A completed trial is cached on disk: re-running the same config and QPS reuses the prior benchmark and only re-evaluates the SLA.

## Outputs

Written under `--output-dir`:

- `capacity_search_results.json`: full results for all jobs, including the per-trial `search_trace`.
- `capacity_search_results.csv`: one row per job (`job_name`, `job_key`, `max_qps_under_sla`, `num_trials`, `error`).
- `jobs/<name>_<key>/job_summary.json`: per-job result.
- `jobs/<name>_<key>/runs/<qps>/`: per-trial benchmark artifacts plus `trial_result.json`.

A job that raises is logged and skipped with its error recorded in the results; remaining jobs still run.
