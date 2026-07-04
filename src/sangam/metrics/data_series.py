"""Two-dimensional (x, y) data series for metrics collection.

Adapted from sarathi-serve (sarathi/metrics/data_series.py).
Plotting uses seaborn instead of plotly; wandb removed.
"""

import logging
from collections import defaultdict, deque

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_MAX_CDF_PLOT_POINTS = 1001
_MAX_TIME_SERIES_POINTS = 1001


def downsample_rows(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Thin a dataframe to at most ``max_points`` evenly-spaced rows, preserving
    the first and last row. Assumes df is already in plot order."""
    if len(df) <= max_points:
        return df
    row_indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[row_indices].reset_index(drop=True)


class DataSeries:
    def __init__(self, x_name: str, y_name: str) -> None:
        self.data_series = deque()
        self.x_name = x_name
        self.y_name = y_name
        self._last_data_y = 0

    def consolidate(self):
        res = defaultdict(list)
        for x, y in self.data_series:
            res[x].append(y)
        self.data_series = [(x, sum(y) / len(y)) for x, y in res.items()]
        self.data_series = sorted(self.data_series, key=lambda x: x[0])
        self._last_data_y = self.data_series[-1][1] if len(self.data_series) else 0

    def merge(self, other: "DataSeries"):
        if len(other) == 0:
            return
        assert self.x_name == other.x_name
        assert self.y_name == other.y_name
        self.data_series.extend(other.data_series)
        self.data_series = sorted(self.data_series, key=lambda x: x[0])
        self._last_data_y = self.data_series[-1][1]

    def elementwise_merge(self, other: "DataSeries"):
        if len(other) == 0:
            return
        assert self.x_name == other.x_name
        assert self.y_name == other.y_name
        self.data_series.extend(other.data_series)
        res = defaultdict(list)
        for x, y in self.data_series:
            res[x].append(y)
        self.data_series = [(x, sum(y) / len(y)) for x, y in res.items()]
        self.data_series = sorted(self.data_series, key=lambda x: x[0])
        self._last_data_y = self.data_series[-1][1]

    @property
    def min_x(self):
        if len(self.data_series) == 0:
            return 0
        return self.data_series[0][0]

    @property
    def max_x(self):
        if len(self.data_series) == 0:
            return 0
        return self.data_series[-1][0]

    def __len__(self):
        return len(self.data_series)

    @property
    def sum(self) -> float:
        return sum(data_y for _, data_y in self.data_series)

    @property
    def metric_name(self) -> str:
        return self.y_name

    def put(self, data_x: float, data_y: float) -> None:
        self._last_data_y = data_y
        self.data_series.append((data_x, data_y))

    def put_pair(self, data_x: float, data_y: float) -> None:
        self.put(data_x, data_y)

    def _peek_y(self):
        return self._last_data_y

    def to_df(self):
        return pd.DataFrame(self.data_series, columns=[self.x_name, self.y_name])

    def put_delta(self, data_x: float, data_y_delta: float) -> None:
        last_data_y = self._peek_y()
        data_y = last_data_y + data_y_delta
        self.put(data_x, data_y)

    def print_distribution_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str | None = None
    ) -> None:
        if len(self.data_series) == 0:
            return
        if y_name is None:
            y_name = self.y_name
        logger.debug(
            f"{plot_name}: {y_name} stats:"
            f" min: {df[y_name].min()},"
            f" max: {df[y_name].max()},"
            f" mean: {df[y_name].mean()},"
            f" median: {df[y_name].median()},"
            f" p95: {df[y_name].quantile(0.95)},"
            f" p99: {df[y_name].quantile(0.99)}"
        )

    def print_series_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str | None = None
    ) -> None:
        if len(self.data_series) == 0:
            return
        if y_name is None:
            y_name = self.y_name
        logger.debug(
            f"{plot_name}: {y_name} stats:"
            f" min: {df[y_name].min()},"
            f" max: {df[y_name].max()},"
            f" mean: {df[y_name].mean()}"
        )

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)

    def save_df(self, path: str, plot_name: str) -> None:
        df = self.to_df()
        self._save_df(df, path, plot_name)

    def _cdf_plot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=[self.y_name], kind="mergesort").reset_index(drop=True)
        df = downsample_rows(df, _MAX_CDF_PLOT_POINTS)

        if len(df) == 1:
            df["cdf"] = 1.0
        else:
            df["cdf"] = np.linspace(0, 1, len(df))
        return df[[self.y_name, "cdf"]]

    def plot_step(
        self,
        path: str,
        plot_name: str,
        y_axis_label: str | None = None,
        start_time: float = 0,
        y_cumsum: bool = True,
    ) -> None:
        if len(self.data_series) == 0:
            return
        if y_axis_label is None:
            y_axis_label = self.y_name

        df = self.to_df()
        df[self.x_name] -= start_time
        if y_cumsum:
            df[self.y_name] = df[self.y_name].cumsum()

        df = downsample_rows(df, _MAX_TIME_SERIES_POINTS)

        self.print_series_stats(df, plot_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=df, x=self.x_name, y=self.y_name, marker="o", markersize=2, ax=ax
        )
        ax.set_ylabel(y_axis_label)
        ax.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(f"{path}/{plot_name}.png", dpi=150)
        plt.close(fig)

        self._save_df(df, path, plot_name)

    def plot_cdf(
        self, path: str, plot_name: str, y_axis_label: str | None = None
    ) -> None:
        if len(self.data_series) == 0:
            return
        if y_axis_label is None:
            y_axis_label = self.y_name

        df = self.to_df()
        self.print_distribution_stats(df, plot_name)
        df = self._cdf_plot_df(df)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(df[self.y_name], df["cdf"], where="post")
        ax.set_xlabel(y_axis_label)
        ax.set_ylabel("CDF")
        ax.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(f"{path}/{plot_name}.png", dpi=150)
        plt.close(fig)

        self._save_df(df, path, plot_name)

    def plot_histogram(self, path: str, plot_name: str) -> None:
        if len(self.data_series) == 0:
            return

        df = self.to_df()
        self.print_distribution_stats(df, plot_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=self.y_name, bins=25, ax=ax)
        ax.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(f"{path}/{plot_name}.png", dpi=150)
        plt.close(fig)
