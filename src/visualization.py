"""Plotting utilities for time-series analysis artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError as exc:  # pragma: no cover - exercised only without deps
    _MISSING_VISUALIZATION_DEPENDENCY = exc
    plt = None
    stats = None
    plot_acf = None
    plot_pacf = None
else:
    _MISSING_VISUALIZATION_DEPENDENCY = None


def require_visualization_dependencies() -> None:
    """Raise a clear setup error when visualization packages are missing."""
    if _MISSING_VISUALIZATION_DEPENDENCY is not None:
        raise RuntimeError(
            "Missing visualization dependencies. Install them with "
            "`pip install -r requirements.txt` before running the pipeline."
        ) from _MISSING_VISUALIZATION_DEPENDENCY


def save_line_plot(series: pd.Series, title: str, path: Path, ylabel: str = "Value") -> None:
    """Save a single time-series line plot."""
    require_visualization_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(series.index, series.values, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_acf_pacf_plots(series: pd.Series, output_dir: Path, lags: int = 24) -> None:
    """Save ACF and PACF plots for a stationary series."""
    require_visualization_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = series.dropna()
    safe_lags = min(lags, max(1, (len(clean) // 2) - 1))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    plot_acf(clean, lags=safe_lags, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "acf_first_difference.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    plot_pacf(clean, lags=safe_lags, method="ywm", ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "pacf_first_difference.png", dpi=160)
    plt.close(fig)


def save_residual_plots(residuals: pd.Series, model_name: str, output_dir: Path) -> None:
    """Save residual time, ACF, and Q-Q diagnostics for a fitted model."""
    require_visualization_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = slugify(model_name)
    clean = residuals.dropna()
    safe_lags = min(24, max(1, (len(clean) // 2) - 1))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(clean.index, clean.values, linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(f"Residuals: {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}_residuals.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    plot_acf(clean, lags=safe_lags, ax=ax)
    ax.set_title(f"Residual ACF: {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}_residual_acf.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(6.2, 6.2))
    stats.probplot(clean, dist="norm", plot=plt)
    plt.title(f"Q-Q plot: {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}_qq.png", dpi=160)
    plt.close(fig)


def save_test_forecast_plot(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    confidence_interval: pd.DataFrame | None,
    model_name: str,
    output_dir: Path,
) -> None:
    """Save train/test forecast plot with optional 95 percent intervals."""
    require_visualization_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(train.index, train.values, label="Train", linewidth=1.5)
    ax.plot(test.index, test.values, label="Test", linewidth=1.5)
    ax.plot(forecast.index, forecast.values, label="Forecast", linewidth=1.8)
    if confidence_interval is not None:
        ax.fill_between(
            confidence_interval.index,
            confidence_interval.iloc[:, 0],
            confidence_interval.iloc[:, 1],
            alpha=0.2,
            label="95% interval",
        )
    ax.set_title(f"Test forecast: {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("INDPRO")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(model_name)}_test_forecast.png", dpi=160)
    plt.close(fig)


def save_future_forecast_plot(
    series: pd.Series,
    forecast: pd.Series,
    confidence_interval: pd.DataFrame,
    model_name: str,
    path: Path,
) -> None:
    """Save future forecast plot with 95 percent intervals."""
    require_visualization_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(series.index, series.values, label="Observed", linewidth=1.5)
    ax.plot(forecast.index, forecast.values, label="Forecast", linewidth=1.8)
    ax.fill_between(
        confidence_interval.index,
        confidence_interval.iloc[:, 0],
        confidence_interval.iloc[:, 1],
        alpha=0.2,
        label="95% interval",
    )
    ax.set_title(f"Future forecast: {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("INDPRO")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def slugify(value: str) -> str:
    """Create a stable filename fragment from a model label."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")
