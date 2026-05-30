"""ARIMA analysis pipeline for the FRED Industrial Production Index.

The script loads a monthly time series, checks stationarity after first
differencing, selects candidate ARIMA models with information criteria, evaluates
them on a chronological test split, and saves forecast artifacts for review.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
except ImportError as exc:  # pragma: no cover - exercised only without deps
    _MISSING_ANALYSIS_DEPENDENCY = exc
    plt = None
    stats = None
    plot_acf = None
    plot_pacf = None
    acorr_ljungbox = None
    SARIMAX = None
    adfuller = None
else:
    _MISSING_ANALYSIS_DEPENDENCY = None


@dataclass(frozen=True)
class AnalysisConfig:
    data_path: Path = Path("INDPRO.csv")
    date_col: str = "observation_date"
    value_col: str = "INDPRO"
    train_ratio: float = 0.80
    max_p: int = 4
    max_q: int = 4
    differencing_order: int = 1
    forecast_steps: int = 10
    top_k: int = 3
    output_dir: Path = Path("reports")


def require_analysis_dependencies() -> None:
    """Raise a clear setup error when optional analysis packages are missing."""
    if _MISSING_ANALYSIS_DEPENDENCY is not None:
        raise RuntimeError(
            "Missing analysis dependencies. Install them with "
            "`pip install -r requirements.txt` before running the pipeline."
        ) from _MISSING_ANALYSIS_DEPENDENCY


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


def adf_test(values: pd.Series, maxlag: int | None = None) -> dict[str, Any]:
    """Run the Augmented Dickey-Fuller test on non-missing observations."""
    require_analysis_dependencies()
    result = adfuller(values.dropna(), maxlag=maxlag, autolag="AIC")
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "n_observations": int(result[3]),
        "critical_values": result[4],
    }


def adf_result_to_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Flatten ADF output into a single-row dataframe for export."""
    row = {
        "adf_statistic": result["adf_statistic"],
        "p_value": result["p_value"],
        "used_lag": result["used_lag"],
        "n_observations": result["n_observations"],
    }
    for key, value in result["critical_values"].items():
        row[f"critical_value_{key}"] = value
    return pd.DataFrame([row])


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


def ic_grid_search(diff_train: pd.Series, max_p: int, max_q: int) -> pd.DataFrame:
    """Estimate ARMA(p, q) candidates on a differenced series."""
    require_analysis_dependencies()
    y = diff_train.dropna()
    rows: list[dict[str, float | int]] = []

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = SARIMAX(
                    y,
                    order=(p, 0, q),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
            except Exception:
                continue

            rows.append(
                {
                    "p": p,
                    "q": q,
                    "AIC": float(result.aic),
                    "BIC": float(result.bic),
                    "HQIC": float(result.hqic),
                }
            )

    if not rows:
        raise RuntimeError(
            "No ARMA candidates were successfully estimated. "
            "Try reducing max_p/max_q or checking data quality."
        )

    return pd.DataFrame(rows).sort_values(["BIC", "AIC"]).reset_index(drop=True)


def select_candidate_orders(
    ic_table: pd.DataFrame, top_k: int
) -> list[tuple[str, int, int]]:
    """Select unique model orders from the leading information criteria."""
    selected: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(label: str, row: pd.Series) -> None:
        order = (int(row["p"]), int(row["q"]))
        if order not in seen:
            selected.append((label, order[0], order[1]))
            seen.add(order)

    for criterion in ["BIC", "AIC", "HQIC"]:
        add(f"best_{criterion.lower()}", ic_table.sort_values(criterion).iloc[0])

    for rank, (_, row) in enumerate(ic_table.sort_values("BIC").head(top_k).iterrows(), 1):
        add(f"bic_rank_{rank}", row)

    return selected


def fit_arima(train: pd.Series, order: tuple[int, int, int]):
    """Fit an ARIMA model through statsmodels' SARIMAX implementation."""
    require_analysis_dependencies()
    model = SARIMAX(
        train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_on_test(fit_result, test: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Produce forecasts aligned to the test set index."""
    forecast = fit_result.get_forecast(steps=len(test))
    mean_forecast = forecast.predicted_mean
    confidence_interval = forecast.conf_int(alpha=0.05)
    mean_forecast.index = test.index
    confidence_interval.index = test.index
    return mean_forecast, confidence_interval


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute common forecast accuracy metrics."""
    aligned = pd.concat([y_true, y_pred], axis=1, keys=["actual", "predicted"]).dropna()
    errors = aligned["actual"] - aligned["predicted"]

    percentage_errors = 100.0 * errors / aligned["actual"].replace(0, np.nan)
    naive_forecast = aligned["actual"].shift(1)
    naive_aligned = pd.concat(
        [aligned["actual"], aligned["predicted"], naive_forecast],
        axis=1,
        keys=["actual", "predicted", "naive"],
    ).dropna()

    if len(naive_aligned) >= 2:
        numerator = np.sqrt(
            np.mean((naive_aligned["actual"] - naive_aligned["predicted"]) ** 2)
        )
        denominator = np.sqrt(
            np.mean((naive_aligned["actual"] - naive_aligned["naive"]) ** 2)
        )
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


def residual_diagnostics(fit_result) -> dict[str, float]:
    """Check residual autocorrelation and normality."""
    require_analysis_dependencies()
    residuals = pd.Series(fit_result.resid).dropna()

    lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    jarque_bera = stats.jarque_bera(residuals)

    return {
        "LjungBox_pvalue_lag10": float(lb["lb_pvalue"].iloc[0]),
        "JB_pvalue": float(jarque_bera.pvalue),
        "Skew": float(stats.skew(residuals)),
        "Kurtosis": float(stats.kurtosis(residuals, fisher=False)),
        "ResidualMean": float(residuals.mean()),
        "ResidualStd": float(residuals.std(ddof=1)),
    }


def save_table(table: pd.DataFrame | pd.Series, path: Path) -> None:
    """Persist a dataframe or series as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(table, pd.Series):
        table.to_frame("value").to_csv(path)
    else:
        table.to_csv(path, index=False)


def save_line_plot(series: pd.Series, title: str, path: Path, ylabel: str = "Value") -> None:
    """Save a single time-series line plot."""
    require_analysis_dependencies()
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
    require_analysis_dependencies()
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
    require_analysis_dependencies()
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
    confidence_interval: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Save train/test forecast plot with 95 percent intervals."""
    require_analysis_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(train.index, train.values, label="Train", linewidth=1.5)
    ax.plot(test.index, test.values, label="Test", linewidth=1.5)
    ax.plot(forecast.index, forecast.values, label="Forecast", linewidth=1.8)
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
    require_analysis_dependencies()
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


def model_name(p: int, d: int, q: int, label: str | None = None) -> str:
    """Format an ARIMA order for readable output."""
    base = f"ARIMA({p},{d},{q})"
    return f"{base} [{label}]" if label else base


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Run the end-to-end analysis and save all portfolio artifacts."""
    require_analysis_dependencies()
    tables_dir = config.output_dir / "tables"
    figures_dir = config.output_dir / "figures"

    series = load_series(config.data_path, config.date_col, config.value_col)
    diff = series.diff(config.differencing_order)
    train, test = train_test_split_series(series, config.train_ratio)
    diff_train = train.diff(config.differencing_order).dropna()

    diff_stats = descriptive_stats(diff)
    adf_result = adf_test(diff)
    ic_table = ic_grid_search(diff_train, config.max_p, config.max_q)
    candidates = select_candidate_orders(ic_table, config.top_k)

    save_table(diff_stats, tables_dir / "descriptive_stats_first_difference.csv")
    save_table(adf_result_to_frame(adf_result), tables_dir / "adf_test_first_difference.csv")
    save_table(ic_table, tables_dir / "ic_grid_search.csv")

    save_line_plot(series, "Industrial Production Index (INDPRO)", figures_dir / "series.png")
    save_line_plot(
        diff,
        "First difference of Industrial Production Index",
        figures_dir / "first_difference.png",
        ylabel="Difference",
    )
    save_acf_pacf_plots(diff, figures_dir)

    comparison_rows: list[dict[str, Any]] = []
    forecast_frames: list[pd.DataFrame] = []

    for label, p, q in candidates:
        order = (p, config.differencing_order, q)
        name = model_name(p, config.differencing_order, q, label)
        fit_result = fit_arima(train, order)
        forecast, confidence_interval = forecast_on_test(fit_result, test)
        accuracy = metrics(test, forecast)
        diagnostics = residual_diagnostics(fit_result)

        comparison_rows.append(
            {
                "Model": name,
                **accuracy,
                "AIC": float(fit_result.aic),
                "BIC": float(fit_result.bic),
                "HQIC": float(fit_result.hqic),
                **diagnostics,
            }
        )

        forecast_frames.append(
            pd.DataFrame(
                {
                    "date": forecast.index,
                    "model": name,
                    "actual": test.values,
                    "forecast": forecast.values,
                    "lower_95": confidence_interval.iloc[:, 0].values,
                    "upper_95": confidence_interval.iloc[:, 1].values,
                }
            )
        )

        residuals = pd.Series(fit_result.resid).dropna()
        save_residual_plots(residuals, name, figures_dir)
        save_test_forecast_plot(train, test, forecast, confidence_interval, name, figures_dir)

    comparison = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)
    save_table(comparison, tables_dir / "model_comparison.csv")
    save_table(pd.concat(forecast_frames, ignore_index=True), tables_dir / "test_forecasts.csv")

    best = comparison.iloc[0]
    match = re.search(r"ARIMA\((\d+),(\d+),(\d+)\)", best["Model"])
    if match is None:
        raise RuntimeError(f"Could not parse best model order from: {best['Model']}")

    best_order = tuple(map(int, match.groups()))
    best_full_fit = fit_arima(series.dropna(), best_order)
    future = best_full_fit.get_forecast(steps=config.forecast_steps)
    future_mean = future.predicted_mean
    future_ci = future.conf_int(alpha=0.05)

    future_forecast = pd.DataFrame(
        {
            "date": future_mean.index,
            "forecast": future_mean.values,
            "lower_95": future_ci.iloc[:, 0].values,
            "upper_95": future_ci.iloc[:, 1].values,
        }
    )
    save_table(future_forecast, tables_dir / "future_forecast.csv")
    save_future_forecast_plot(
        series,
        future_mean,
        future_ci,
        best["Model"],
        figures_dir / "future_forecast.png",
    )

    return {
        "series": series,
        "train": train,
        "test": test,
        "diff_stats": diff_stats,
        "adf_result": adf_result,
        "ic_table": ic_table,
        "model_comparison": comparison,
        "future_forecast": future_forecast,
        "best_model": best["Model"],
        "output_dir": config.output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run ARIMA model selection and forecasting for INDPRO."
    )
    parser.add_argument("--data", type=Path, default=Path("INDPRO.csv"), help="Input CSV path.")
    parser.add_argument("--date-col", default="observation_date", help="Date column name.")
    parser.add_argument("--value-col", default="INDPRO", help="Value column name.")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Chronological train share.")
    parser.add_argument("--max-p", type=int, default=4, help="Maximum AR order for grid search.")
    parser.add_argument("--max-q", type=int, default=4, help="Maximum MA order for grid search.")
    parser.add_argument("--forecast-steps", type=int, default=10, help="Future periods to forecast.")
    parser.add_argument("--top-k", type=int, default=3, help="Extra BIC-ranked candidates.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Artifact directory.")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Map CLI arguments to an immutable config object."""
    return AnalysisConfig(
        data_path=args.data,
        date_col=args.date_col,
        value_col=args.value_col,
        train_ratio=args.train_ratio,
        max_p=args.max_p,
        max_q=args.max_q,
        forecast_steps=args.forecast_steps,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    config = config_from_args(parser.parse_args())
    try:
        artifacts = run_analysis(config)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    comparison = artifacts["model_comparison"]
    adf_result = artifacts["adf_result"]
    series = artifacts["series"]
    train = artifacts["train"]
    test = artifacts["test"]

    print("ARIMA analysis complete")
    print(f"Dataset: {config.value_col}, {series.index.min().date()} to {series.index.max().date()}")
    print(f"Train observations: {len(train)} | Test observations: {len(test)}")
    print(
        "ADF on first difference: "
        f"statistic={adf_result['adf_statistic']:.4f}, p={adf_result['p_value']:.4g}"
    )
    print(
        "Best model by test RMSE: "
        f"{artifacts['best_model']} (RMSE={comparison.iloc[0]['RMSE']:.4f})"
    )
    print(f"Artifacts saved to: {artifacts['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
