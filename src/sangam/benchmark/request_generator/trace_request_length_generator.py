import json
import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd

from sangam.benchmark.config import TraceRequestLengthGeneratorConfig
from sangam.benchmark.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)

logger = logging.getLogger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, config: TraceRequestLengthGeneratorConfig):
        super().__init__(config)

        self.trace_df = pd.read_csv(config.trace_file)

        # scale prefill and decode tokens
        self.trace_df["prompt_len"] = (
            self.trace_df["prompt_len"] * config.prefill_scale_factor
        )
        self.trace_df["gen_len"] = self.trace_df["gen_len"] * config.decode_scale_factor

        # make sure all the prefill and decode counts are integers
        self.trace_df["prompt_len"] = self.trace_df["prompt_len"].astype(int)
        self.trace_df["gen_len"] = self.trace_df["gen_len"].astype(int)

        # make sure the total does not exceed the max tokens
        if config.max_tokens is not None:
            total_tokens = self.trace_df["prompt_len"] + self.trace_df["gen_len"]
            diff_tokens = total_tokens - config.max_tokens
            diff_tokens = diff_tokens.clip(lower=0)

            prefill_tokens_ratio = self.trace_df["prompt_len"] / total_tokens
            decode_tokens_ratio = self.trace_df["gen_len"] / total_tokens

            self.trace_df["prompt_len"] -= (
                np.ceil(diff_tokens * prefill_tokens_ratio)
            ).astype(int)

            self.trace_df["gen_len"] -= (
                np.ceil(diff_tokens * decode_tokens_ratio)
            ).astype(int)

        # make sure that there is at least one prefill and decode token
        self.trace_df["prompt_len"] = self.trace_df["prompt_len"].clip(lower=1)
        self.trace_df["gen_len"] = self.trace_df["gen_len"].clip(lower=1)

        if config.max_tokens is not None:
            assert all(
                self.trace_df["prompt_len"] + self.trace_df["gen_len"]
                <= self.config.max_tokens
            )
        assert all(self.trace_df["prompt_len"] > 0)
        assert all(self.trace_df["gen_len"] > 0)

        pd_ratio = self.trace_df["prompt_len"] / self.trace_df["gen_len"]
        logger.info(
            f"Loaded request length trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        pd_distribution = pd_ratio.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        logger.debug(f"Prompt/decode token ratio stats\n: {pd_distribution}")

        if self.config.shuffle:
            # randomly shuffle the df based on the seed
            self.trace_df = self.trace_df.sample(frac=1, random_state=self.config.seed)
        self.next_request_idx = 0

    def get_next_num_tokens(self) -> Tuple[float, float]:
        if self.next_request_idx >= len(self.trace_df):
            return None, None

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row["prompt_len"],
            row["gen_len"],
        )

    def get_next_request_payload(self) -> dict[str, Any] | None:
        if self.next_request_idx == 0:
            return None

        row = self.trace_df.iloc[self.next_request_idx - 1]
        if "messages" not in row.index:
            return None

        payload: dict[str, Any] = {
            "messages": json.loads(row["messages"]),
        }
        if "session_id" in row.index and "request_id_in_session" in row.index:
            payload["external_id"] = (
                f"{int(row['session_id'])}-{int(row['request_id_in_session'])}"
            )
        return payload
