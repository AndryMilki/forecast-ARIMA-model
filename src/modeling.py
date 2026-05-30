"""ARIMA model selection and forecasting utilities."""

from __future__ import annotations

import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:  # pragma: no cover - exercised only without deps
    _MISSING_MODELING_DEPENDENCY = exc
    SARIMAX = None
else:
    _MISSING_MODELING_DEPENDENCY = None


def require_modeling_dependencies() -> None:
    """Raise a clear setup error when modeling packages are missing."""
    if _MISSING_MODELING_DEPENDENCY is not None:
        raise RuntimeError(
            "Missing modeling dependencies. Install them with "
            "`pip install -r requirements.txt` before running the pipeline."
        ) from _MISSING_MODELING_DEPENDENCY


def ic_grid_search(diff_train: pd.Series, max_p: int, max_q: int) -> pd.DataFrame:
    """Estimate ARMA(p, q) candidates on a differenced series."""
    require_modeling_dependencies()
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
    require_modeling_dependencies()
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


def model_name(p: int, d: int, q: int, label: str | None = None) -> str:
    """Format an ARIMA order for readable output."""
    base = f"ARIMA({p},{d},{q})"
    return f"{base} [{label}]" if label else base
