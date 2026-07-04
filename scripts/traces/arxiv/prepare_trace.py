"""Compute arxiv-summarization request lengths using the LLaDA tokenizer.

Each row in the dataset becomes one request whose prompt is the `article`
field and whose completion is the `abstract` field. Token counts are
computed with the configured tokenizer (LLaDA by default).
"""

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, TypedDict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

DEFAULT_LLADA_INSTRUCT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_DATASET_ROOT = Path.home() / "arxiv-summarization"

FILTER_MODE_ARXIV_8K = "arxiv_8k"
FILTER_MODE_ARXIV_4K = "arxiv_4k"

_WORKER_TOKENIZER: PreTrainedTokenizerBase = None
_WORKER_IS_INSTRUCT = False


class RequestPayload(TypedDict):
    session_id: int
    request_id_in_session: int
    messages: List[Dict[str, str]]
    prompt_len: int
    gen_len: int


def _build_messages(article: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "Summarize the following arxiv paper.\n\n" + article,
        }
    ]


def _render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    is_instruct: bool,
) -> str:
    if not is_instruct:
        return messages[0]["content"]

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer appears to be an instruct model, but tokenizer.chat_template "
            "is not set. Use an instruct tokenizer with a chat template or pass a "
            "non-instruct tokenizer/model."
        )
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _init_worker(
    tokenizer_name: str,
    trust_remote_code: bool,
    is_instruct: bool,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_IS_INSTRUCT
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    _WORKER_IS_INSTRUCT = is_instruct


def _keep_row(
    prompt_len: int,
    gen_len: int,
    disable_filtering: bool,
    filter_mode: str,
) -> bool:
    if disable_filtering:
        return True
    if filter_mode == FILTER_MODE_ARXIV_8K:
        if gen_len > 512:
            return False
        if prompt_len + gen_len > 8192:
            return False
        return True
    if filter_mode == FILTER_MODE_ARXIV_4K:
        if gen_len > 512:
            return False
        if prompt_len + gen_len > 4096:
            return False
        return True
    raise ValueError(f"Unsupported filter mode: {filter_mode}")


def _process_batch(
    session_ids: List[int],
    articles: List[str],
    abstracts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    is_instruct: bool,
    disable_filtering: bool,
    filter_mode: str,
) -> List[RequestPayload]:
    messages_list = [_build_messages(article) for article in articles]
    prompts = [_render_prompt(tokenizer, msgs, is_instruct) for msgs in messages_list]
    prompt_token_ids = tokenizer(prompts).input_ids
    completion_token_ids = tokenizer(abstracts).input_ids

    rows: List[RequestPayload] = []
    for i, session_id in enumerate(session_ids):
        prompt_len = len(prompt_token_ids[i])
        gen_len = len(completion_token_ids[i])
        if not _keep_row(prompt_len, gen_len, disable_filtering, filter_mode):
            continue
        rows.append(
            {
                "session_id": session_id,
                "request_id_in_session": 0,
                "messages": messages_list[i],
                "prompt_len": prompt_len,
                "gen_len": gen_len,
            }
        )
    return rows


def _process_batch_worker(
    task: tuple[List[int], List[str], List[str], bool, str],
) -> List[RequestPayload]:
    (
        session_ids,
        articles,
        abstracts,
        disable_filtering,
        filter_mode,
    ) = task
    return _process_batch(
        session_ids,
        articles,
        abstracts,
        _WORKER_TOKENIZER,
        _WORKER_IS_INSTRUCT,
        disable_filtering,
        filter_mode,
    )


def _load_dataset(
    dataset_root: Path,
    config: str,
    split: str,
) -> pd.DataFrame:
    split_dir = dataset_root / config
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Expected config dir at {split_dir}")
    files = sorted(split_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files matching {split}-*.parquet in {split_dir}"
        )
    frames = [pd.read_parquet(path, columns=["article", "abstract"]) for path in files]
    return pd.concat(frames, ignore_index=True)


def load_and_prepare_requests(
    dataset_root: Path,
    config: str,
    split: str,
    tokenizer_name: str,
    trust_remote_code: bool,
    num_documents: int | None,
    sample_seed: int,
    num_workers: int,
    worker_batch_size: int,
    disable_filtering: bool,
    filter_mode: str,
) -> List[RequestPayload]:
    df = _load_dataset(dataset_root, config, split)

    if num_documents is not None:
        if num_documents < 1:
            raise ValueError("--num-documents must be >= 1.")
        if num_documents < len(df):
            rng = np.random.default_rng(sample_seed)
            sampled = sorted(rng.choice(len(df), size=num_documents, replace=False))
            df = df.iloc[sampled].reset_index(drop=True)

    is_instruct = "instruct" in tokenizer_name.lower()

    tasks: List[tuple[List[int], List[str], List[str], bool, str]] = []
    for start in range(0, len(df), worker_batch_size):
        end = min(start + worker_batch_size, len(df))
        session_ids = list(range(start, end))
        articles = df["article"].iloc[start:end].astype(str).tolist()
        abstracts = df["abstract"].iloc[start:end].astype(str).tolist()
        tasks.append(
            (
                session_ids,
                articles,
                abstracts,
                disable_filtering,
                filter_mode,
            )
        )

    rows: List[RequestPayload] = []
    if num_workers == 1:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        for (
            session_ids,
            articles,
            abstracts,
            disable_filtering_task,
            filter_mode_task,
        ) in tqdm(tasks, total=len(tasks), desc="Tokenizing documents"):
            rows.extend(
                _process_batch(
                    session_ids,
                    articles,
                    abstracts,
                    tokenizer,
                    is_instruct,
                    disable_filtering_task,
                    filter_mode_task,
                )
            )
    else:
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(tokenizer_name, trust_remote_code, is_instruct),
        ) as pool:
            for batch_rows in tqdm(
                pool.imap(_process_batch_worker, tasks, chunksize=1),
                total=len(tasks),
                desc="Tokenizing documents",
            ):
                rows.extend(batch_rows)

    return rows


def print_summary(rows: List[RequestPayload], mode: str) -> None:
    prompt_lens = np.array([row["prompt_len"] for row in rows])
    gen_lens = np.array([row["gen_len"] for row in rows])
    total_lens = prompt_lens + gen_lens

    def describe(name: str, values: np.ndarray) -> None:
        print(f"{name}:")
        print(f"  min={int(np.min(values))}")
        print(f"  max={int(np.max(values))}")
        print(f"  mean={float(np.mean(values)):.2f}")
        if mode == "full":
            p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
            print(f"  p50={float(p50):.2f}")
            print(f"  p90={float(p90):.2f}")
            print(f"  p95={float(p95):.2f}")
            print(f"  p99={float(p99):.2f}")

    print(f"Document count: {len(rows)}")
    describe("Prompt lengths", prompt_lens)
    describe("Output lengths", gen_lens)
    describe("Total lengths", total_lens)


def write_csv(
    output_csv: Path,
    rows: List[RequestPayload],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "session_id",
                "request_id_in_session",
                "messages",
                "prompt_len",
                "gen_len",
                "total_len",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["session_id"],
                    row["request_id_in_session"],
                    json.dumps(row["messages"], ensure_ascii=False),
                    row["prompt_len"],
                    row["gen_len"],
                    row["prompt_len"] + row["gen_len"],
                ]
            )


def main(args: argparse.Namespace) -> None:
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")
    if args.worker_batch_size < 1:
        raise ValueError("--worker-batch-size must be >= 1.")

    if args.tokenizer is None:
        if args.model is None:
            raise ValueError("Either --tokenizer or --model must be provided.")
        args.tokenizer = args.model

    rows = load_and_prepare_requests(
        dataset_root=Path(args.dataset_root).expanduser(),
        config=args.config,
        split=args.split,
        tokenizer_name=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        num_documents=args.num_documents,
        sample_seed=args.sample_seed,
        num_workers=args.num_workers,
        worker_batch_size=args.worker_batch_size,
        disable_filtering=args.disable_filtering,
        filter_mode=args.filter_mode,
    )

    if not rows:
        if args.disable_filtering:
            raise ValueError("No documents found.")
        raise ValueError(f"No documents passed the '{args.filter_mode}' filter.")

    if args.summary != "none":
        print_summary(rows, mode=args.summary)
    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print(f"Wrote per-document lengths to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute arxiv-summarization prompt/gen lengths using the "
        "LLaDA tokenizer."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the local arxiv-summarization dataset root.",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["document", "section"],
        default="document",
        help="Which dataset config to read (document or section).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Which split to read.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLADA_INSTRUCT_MODEL,
        help="Model name/path. Used as tokenizer if --tokenizer is not provided.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or path of tokenizer.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/traces/arxiv_summarization_llada2_tokenizer_filtered_4k.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=[FILTER_MODE_ARXIV_8K, FILTER_MODE_ARXIV_4K],
        default=FILTER_MODE_ARXIV_4K,
        help="Filtering preset: arxiv_8k keeps requests with gen_len <= 512 "
        "and prompt_len + gen_len <= 8192; arxiv_4k keeps requests with "
        "gen_len <= 512 and prompt_len + gen_len <= 4096.",
    )
    parser.add_argument(
        "--disable-filtering",
        action="store_true",
        help="Disable all filtering, regardless of --filter-mode.",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=None,
        help="Randomly sample up to this many documents before tokenizing.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed used for random document sampling.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        choices=["none", "basic", "full"],
        default="full",
        help="Summary mode.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from Hugging Face.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes for tokenization.",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=64,
        help="Documents per tokenizer batch (and per worker task).",
    )
    main(parser.parse_args())
