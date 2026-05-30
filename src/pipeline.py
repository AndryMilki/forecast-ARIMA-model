"""End-to-end forecasting pipeline orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_series, save_table
from src.diagnostics import adf_result_to_frame, adf_test, residual_diagnostics
from src.evaluation import metrics, moving_average_forecast, naive_forecast
from src.modeling import (
    fit_arima,
    forecast_on_test,
    ic_grid_search,
    model_name,
    select_candidate_orders,
)
from src.preprocessing import descriptive_stats, first_difference, train_test_split_series
from src.visualization import (
    save_acf_pacf_plots,
    save_future_forecast_plot,
    save_line_plot,
    save_residual_plots,
    save_test_forecast_plot,
)


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
    moving_average_window: int = 12
    output_dir: Path = Path("reports")


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Run the end-to-end analysis and save all portfolio artifacts."""
    tables_dir = config.output_dir / "tables"
    figures_dir = config.output_dir / "figures"

    series = load_series(config.data_path, config.date_col, config.value_col)
    diff = first_difference(series, config.differencing_order)
    train, test = train_test_split_series(series, config.train_ratio)
    diff_train = first_difference(train, config.differencing_order).dropna()

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

    baseline_forecasts = [
        ("Naive forecast", naive_forecast(train, test)),
        (
            f"Moving average ({config.moving_average_window}) forecast",
            moving_average_forecast(train, test, config.moving_average_window),
        ),
    ]
    for name, forecast in baseline_forecasts:
        accuracy = metrics(test, forecast)
        comparison_rows.append(
            {
                "Model": name,
                "ModelType": "Baseline",
                **accuracy,
                "AIC": np.nan,
                "BIC": np.nan,
                "HQIC": np.nan,
                "LjungBox_pvalue_lag10": np.nan,
                "JB_pvalue": np.nan,
                "Skew": np.nan,
                "Kurtosis": np.nan,
                "ResidualMean": np.nan,
                "ResidualStd": np.nan,
            }
        )
        forecast_frames.append(_forecast_frame(name, test, forecast))
        save_test_forecast_plot(train, test, forecast, None, name, figures_dir)

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
                "ModelType": "ARIMA",
                **accuracy,
                "AIC": float(fit_result.aic),
                "BIC": float(fit_result.bic),
                "HQIC": float(fit_result.hqic),
                **diagnostics,
            }
        )

        forecast_frames.append(_forecast_frame(name, test, forecast, confidence_interval))

        residuals = pd.Series(fit_result.resid).dropna()
        save_residual_plots(residuals, name, figures_dir)
        save_test_forecast_plot(train, test, forecast, confidence_interval, name, figures_dir)

    comparison = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)
    baseline_comparison = comparison.loc[
        comparison["ModelType"].isin(["Baseline", "ARIMA"]),
        ["Model", "ModelType", "RMSE", "MAE", "MAPE_percent", "Theil_U2"],
    ].copy()
    save_table(comparison, tables_dir / "model_comparison.csv")
    save_table(baseline_comparison, tables_dir / "baseline_comparison.csv")
    save_table(pd.concat(forecast_frames, ignore_index=True), tables_dir / "test_forecasts.csv")

    arima_comparison = comparison[comparison["ModelType"] == "ARIMA"].reset_index(drop=True)
    best_arima = arima_comparison.iloc[0]
    best_order = _parse_arima_order(best_arima["Model"])
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
        best_arima["Model"],
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
        "baseline_comparison": baseline_comparison,
        "future_forecast": future_forecast,
        "best_model": comparison.iloc[0]["Model"],
        "best_arima_model": best_arima["Model"],
        "best_arima_rmse": float(best_arima["RMSE"]),
        "output_dir": config.output_dir,
    }


def _forecast_frame(
    model: str,
    test: pd.Series,
    forecast: pd.Series,
    confidence_interval: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a normalized forecast table for export."""
    frame = pd.DataFrame(
        {
            "date": forecast.index,
            "model": model,
            "actual": test.values,
            "forecast": forecast.values,
        }
    )
    if confidence_interval is None:
        frame["lower_95"] = np.nan
        frame["upper_95"] = np.nan
    else:
        frame["lower_95"] = confidence_interval.iloc[:, 0].values
        frame["upper_95"] = confidence_interval.iloc[:, 1].values
    return frame


def _parse_arima_order(model: str) -> tuple[int, int, int]:
    """Parse an ARIMA(p,d,q) order from a model label."""
    match = re.search(r"ARIMA\((\d+),(\d+),(\d+)\)", model)
    if match is None:
        raise RuntimeError(f"Could not parse ARIMA order from: {model}")
    return tuple(map(int, match.groups()))
