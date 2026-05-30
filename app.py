"""Command-line entry point for the INDPRO ARIMA analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import AnalysisConfig, run_analysis


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run baseline and ARIMA forecasting analysis for INDPRO."
    )
    parser.add_argument("--data", type=Path, default=Path("INDPRO.csv"), help="Input CSV path.")
    parser.add_argument("--date-col", default="observation_date", help="Date column name.")
    parser.add_argument("--value-col", default="INDPRO", help="Value column name.")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Chronological train share.")
    parser.add_argument("--max-p", type=int, default=4, help="Maximum AR order for grid search.")
    parser.add_argument("--max-q", type=int, default=4, help="Maximum MA order for grid search.")
    parser.add_argument("--forecast-steps", type=int, default=10, help="Future periods to forecast.")
    parser.add_argument("--top-k", type=int, default=3, help="Extra BIC-ranked ARIMA candidates.")
    parser.add_argument(
        "--ma-window",
        type=int,
        default=12,
        help="Window size for the moving-average baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Artifact directory.")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Map CLI arguments to an immutable pipeline config."""
    return AnalysisConfig(
        data_path=args.data,
        date_col=args.date_col,
        value_col=args.value_col,
        train_ratio=args.train_ratio,
        max_p=args.max_p,
        max_q=args.max_q,
        forecast_steps=args.forecast_steps,
        top_k=args.top_k,
        moving_average_window=args.ma_window,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the CLI."""
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

    print("Forecasting analysis complete")
    print(f"Dataset: {config.value_col}, {series.index.min().date()} to {series.index.max().date()}")
    print(f"Train observations: {len(train)} | Test observations: {len(test)}")
    print(
        "ADF on first difference: "
        f"statistic={adf_result['adf_statistic']:.4f}, p={adf_result['p_value']:.4g}"
    )
    print(
        "Best overall model by test RMSE: "
        f"{comparison.iloc[0]['Model']} (RMSE={comparison.iloc[0]['RMSE']:.4f})"
    )
    print(
        "Best ARIMA model by test RMSE: "
        f"{artifacts['best_arima_model']} "
        f"(RMSE={artifacts['best_arima_rmse']:.4f})"
    )
    print(f"Artifacts saved to: {artifacts['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
