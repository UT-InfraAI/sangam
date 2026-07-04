# arxiv-summarization trace

`prepare_trace.py` generates the arxiv length trace used by the benchmarks. It tokenizes the [arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) dataset (each document is one request: `article` is the prompt, `abstract` the completion) with the LLaDA instruct tokenizer and writes per-request token counts to CSV.

## Run

First clone the dataset, then run the script:

```bash
git clone https://huggingface.co/datasets/ccdv/arxiv-summarization ~/arxiv-summarization
uv run python scripts/traces/arxiv/prepare_trace.py
```

This writes the trace to `data/traces/arxiv_summarization_llada2_tokenizer_filtered_4k.csv`, the path the benchmark examples expect.

The script reads the dataset from `~/arxiv-summarization/{document,section}/{train,validation,test}-*.parquet`. Override the location with `--dataset-root`, `--config`, and `--split`. Run with `--help` to see all flags.
