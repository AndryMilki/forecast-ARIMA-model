"""Preprocessing helpers for time-series analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def descriptive_stats(values: pd.Series) -> pd.Series:
    """Return compact descriptive statistics for a series."""
    return pd.Series(
        {
            "mean": values.mean(),
            "min": values.min(),
            "max": values.max(),
            "std": values.std(ddof=1),
            "missing": values.isna().sum(),
        }
    )


def first_difference(series: pd.Series, order: int = 1) -> pd.Series:
    """Return an order-d differenced series."""
    if order < 1:
        raise ValueError("order must be at least 1.")
    return series.diff(order)


def train_test_split_series(
    series: pd.Series, train_ratio: float
) -> tuple[pd.Series, pd.Series]:
    """Split a series chronologically into train and test samples."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    clean = series.dropna()
    split_at = int(np.floor(len(clean) * train_ratio))
    if split_at == 0 or split_at == len(clean):
        raise ValueError("train_ratio leaves an empty train or test split.")

    return clean.iloc[:split_at], clean.iloc[split_at:]


def clean_indpro_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Convert a one-column malformed INDPRO CSV export into date/value columns."""
    raw = pd.read_csv(input_path, header=None)

    if raw.shape[1] == 1:
        split = raw.iloc[:, 0].astype(str).str.split(",", expand=True)
    else:
        split = raw.iloc[:, :2].copy()

    split.columns = ["observation_date", "INDPRO"]
    if split.iloc[0, 0] == "observation_date":
        split = split.iloc[1:]

    split["observation_date"] = pd.to_datetime(split["observation_date"], errors="coerce")
    split["INDPRO"] = pd.to_numeric(split["INDPRO"], errors="coerce")
    split = split.dropna(subset=["observation_date", "INDPRO"])
    split = split.sort_values("observation_date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, index=False)
    return split
