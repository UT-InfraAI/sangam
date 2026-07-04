import logging
from typing import List

import pandas as pd

from sangam.benchmark.config import TraceRequestGeneratorConfig
from sangam.benchmark.entities import Request
from sangam.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)

logger = logging.getLogger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    """Reads a trace CSV with arrival time, prompt and completion token counts."""

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        self.trace_df = pd.read_csv(config.trace_file)
        # restrict to a specific date
        self.trace_df = self.trace_df[self.trace_df["Date"] == config.date]

        # scale prefill and decode tokens
        self.trace_df["PromptTokenCount"] = (
            self.trace_df["PromptTokenCount"] * config.prefill_scale_factor
        )
        self.trace_df["CompletionTokenCount"] = (
            self.trace_df["CompletionTokenCount"] * config.decode_scale_factor
        )

        # make sure all counts are integers
        self.trace_df["PromptTokenCount"] = self.trace_df["PromptTokenCount"].astype(
            int
        )
        self.trace_df["CompletionTokenCount"] = self.trace_df[
            "CompletionTokenCount"
        ].astype(int)

        # at least one token each
        self.trace_df["PromptTokenCount"] = self.trace_df["PromptTokenCount"].clip(
            lower=1
        )
        self.trace_df["CompletionTokenCount"] = self.trace_df[
            "CompletionTokenCount"
        ].clip(lower=1)

        # enforce max_tokens
        total_tokens = (
            self.trace_df["PromptTokenCount"] + self.trace_df["CompletionTokenCount"]
        )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        self.trace_df["PromptTokenCount"] = (
            self.trace_df["PromptTokenCount"] - diff_tokens
        )

        assert all(
            self.trace_df["PromptTokenCount"] + self.trace_df["CompletionTokenCount"]
            <= config.max_tokens
        )

        # rescale time
        self.trace_df["Time"] = self.trace_df["Time"] * config.time_scale_factor

        pd_ratio = (
            self.trace_df["PromptTokenCount"] / self.trace_df["CompletionTokenCount"]
        )
        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def generate_requests(self) -> List[Request]:
        requests = []

        for _, row in self.trace_df.iterrows():
            request = Request(
                arrived_at=row["Time"],
                prompt_len=row["PromptTokenCount"],
                gen_len=row["CompletionTokenCount"],
            )
            requests.append(request)

        return requests
