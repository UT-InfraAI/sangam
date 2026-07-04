"""Memory-efficient CDF tracking using DDSketch.

Adapted from sarathi-serve (sarathi/metrics/cdf_sketch.py).
Plotting uses seaborn instead of plotly; wandb removed.
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddsketch.ddsketch import DDSketch

logger = logging.getLogger(__name__)


class CDFSketch:
    def __init__(
        self,
        metric_name: str,
        relative_accuracy: float = 0.001,
        num_quantiles_in_df: int = 101,
    ) -> None:
        self.sketch = DDSketch(relative_accuracy=relative_accuracy)
        self.metric_name = metric_name
        self._last_data = 0
        self._num_quantiles_in_df = num_quantiles_in_df

    @property
    def mean(self) -> float:
        return self.sketch.avg

    @property
    def median(self) -> float:
        return self.sketch.get_quantile_value(0.5)

    @property
    def sum(self) -> float:
        return self.sketch.sum

    def __len__(self):
        return int(self.sketch.count)

    def merge(self, other: "CDFSketch") -> None:
        assert self.metric_name == other.metric_name
        self.sketch.merge(other.sketch)

    def put(self, data: float) -> None:
        self._last_data = data
        self.sketch.add(data)

    def put_pair(self, data_x: float, data_y: float) -> None:
        self._last_data = data_y
        self.sketch.add(data_y)

    def put_delta(self, delta: float) -> None:
        data = self._last_data + delta
        self.put(data)

    def print_distribution_stats(self, plot_name: str) -> None:
        if self.sketch._count == 0:
            return

        logger.debug(
            f"{plot_name}: {self.metric_name} stats:"
            f" min: {self.sketch._min},"
            f" max: {self.sketch._max},"
            f" mean: {self.sketch.avg},"
            f" median: {self.sketch.get_quantile_value(0.5)},"
            f" p95: {self.sketch.get_quantile_value(0.95)},"
            f" p99: {self.sketch.get_quantile_value(0.99)},"
            f" count: {self.sketch._count}"
        )

    def to_df(self) -> pd.DataFrame:
        quantiles = np.linspace(0, 1, self._num_quantiles_in_df)
        quantile_values = [self.sketch.get_quantile_value(q) for q in quantiles]
        return pd.DataFrame({"cdf": quantiles, self.metric_name: quantile_values})

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)

    def plot_cdf(
        self, path: str, plot_name: str, x_axis_label: str | None = None
    ) -> None:
        if self.sketch._count == 0:
            return

        if x_axis_label is None:
            x_axis_label = self.metric_name

        df = self.to_df()
        self.print_distribution_stats(plot_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(df[self.metric_name], df["cdf"], where="post")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("CDF")
        ax.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(f"{path}/{plot_name}.png", dpi=150)
        plt.close(fig)

        self._save_df(df, path, plot_name)
