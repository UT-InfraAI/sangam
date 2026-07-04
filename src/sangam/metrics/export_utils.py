"""Shared export and plotting helpers for metrics modules."""

import os
from functools import reduce
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sangam.metrics.cdf_sketch import CDFSketch
from sangam.metrics.data_series import DataSeries


def save_merged_dataseries_csv(
    dataseries_list: list[DataSeries],
    key_to_join: str,
    base_path: str,
    file_name: str,
    float_format: str,
    extra_columns_df: pd.DataFrame | None = None,
) -> None:
    os.makedirs(base_path, exist_ok=True)
    dataseries_dfs = [ds.to_df() for ds in dataseries_list if len(ds) > 0]
    if not dataseries_dfs:
        return
    merged_df = reduce(
        lambda left, right: left.merge(right, on=key_to_join, how="outer"),
        dataseries_dfs,
    )
    if extra_columns_df is not None and not extra_columns_df.empty:
        merged_df = merged_df.merge(extra_columns_df, on=key_to_join, how="left")
    merged_df.to_csv(
        f"{base_path}/{file_name}.csv",
        index=False,
        float_format=float_format,
    )


def plot_worker_metric_cdf(
    output_dir: str,
    metric_name: str,
    metric_column: str,
    x_label: str,
    source_map: Dict[str, Dict[object, CDFSketch]],
    metric_enum: object,
) -> None:
    worker_dfs: list[pd.DataFrame] = []
    for worker_id in sorted(source_map.keys()):
        sketch = source_map[worker_id][metric_enum]
        if len(sketch) == 0:
            continue
        df = sketch.to_df()
        df["worker_id"] = worker_id
        worker_dfs.append(df)

    if not worker_dfs:
        return

    metric_df = pd.concat(worker_dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=metric_df,
        x=metric_column,
        y="cdf",
        hue="worker_id",
        ax=ax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_title(metric_name)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{metric_name}.png", dpi=150)
    plt.close(fig)
    metric_df.to_csv(f"{output_dir}/{metric_name}.csv", index=False)


def worker_plot_name(metric_name: str) -> str:
    return metric_name if metric_name.startswith("worker_") else f"worker_{metric_name}"
