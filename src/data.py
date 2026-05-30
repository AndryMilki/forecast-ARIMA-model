"""Data loading and artifact persistence utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_series(path: Path | str, date_col: str, value_col: str) -> pd.Series:
    """Load, validate, sort, and frequency-align the input time series."""
    df = pd.read_csv(path)
    missing_columns = {date_col, value_col}.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    series = df[value_col].astype(float)
    freq = pd.infer_freq(series.index) or "MS"
    return series.asfreq(freq)


def save_table(table: pd.DataFrame | pd.Series, path: Path) -> None:
    """Persist a dataframe or series as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(table, pd.Series):
        table.to_frame("value").to_csv(path)
    else:
        table.to_csv(path, index=False)
