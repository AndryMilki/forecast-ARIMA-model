"""Forecast baselines and accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, test: pd.Series) -> pd.Series:
    """Forecast every test point with the final training observation."""
    if train.dropna().empty:
        raise ValueError("train must contain at least one non-missing value.")
    return pd.Series(float(train.dropna().iloc[-1]), index=test.index, name="forecast")


def moving_average_forecast(train: pd.Series, test: pd.Series, window: int = 12) -> pd.Series:
    """Forecast every test point with the trailing train moving average."""
    if window < 1:
        raise ValueError("window must be at least 1.")
    clean_train = train.dropna()
    if clean_train.empty:
        raise ValueError("train must contain at least one non-missing value.")
    forecast_value = float(clean_train.tail(window).mean())
    return pd.Series(forecast_value, index=test.index, name="forecast")


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute common forecast accuracy metrics."""
    aligned = pd.concat([y_true, y_pred], axis=1, keys=["actual", "predicted"]).dropna()
    errors = aligned["actual"] - aligned["predicted"]

    percentage_errors = 100.0 * errors / aligned["actual"].replace(0, np.nan)
    naive = aligned["actual"].shift(1)
    naive_aligned = pd.concat(
        [aligned["actual"], aligned["predicted"], naive],
        axis=1,
        keys=["actual", "predicted", "naive"],
    ).dropna()

    if len(naive_aligned) >= 2:
        numerator = np.sqrt(
            np.mean((naive_aligned["actual"] - naive_aligned["predicted"]) ** 2)
        )
        denominator = np.sqrt(np.mean((naive_aligned["actual"] - naive_aligned["naive"]) ** 2))
        theil_u2 = float(numerator / denominator) if denominator > 0 else np.nan
    else:
        theil_u2 = np.nan

    return {
        "ME": float(errors.mean()),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(np.abs(errors))),
        "MPE_percent": float(np.nanmean(percentage_errors)),
        "MAPE_percent": float(np.nanmean(np.abs(percentage_errors))),
        "Theil_U2": theil_u2,
    }
