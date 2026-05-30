"""Normalize malformed INDPRO CSV exports.

This utility is kept for reproducibility in case the source file is downloaded
as a single comma-separated text column. The main analysis uses INDPRO.csv
directly and does not require this step when the CSV already has two columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean_indpro_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Convert a one-column malformed CSV into date/value columns."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean a malformed INDPRO CSV export.")
    parser.add_argument("--input", type=Path, default=Path("INDPRO.csv"), help="Source CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("INDPRO_clean.csv"),
        help="Cleaned CSV output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cleaned = clean_indpro_csv(args.input, args.output)
    print(f"Saved {len(cleaned)} rows to {args.output}")


if __name__ == "__main__":
    main()
