"""Stationarity and residual diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
except ImportError as exc:  # pragma: no cover - exercised only without deps
    _MISSING_DIAGNOSTICS_DEPENDENCY = exc
    stats = None
    acorr_ljungbox = None
    adfuller = None
else:
    _MISSING_DIAGNOSTICS_DEPENDENCY = None


def require_diagnostics_dependencies() -> None:
    """Raise a clear setup error when diagnostics packages are missing."""
    if _MISSING_DIAGNOSTICS_DEPENDENCY is not None:
        raise RuntimeError(
            "Missing diagnostics dependencies. Install them with "
            "`pip install -r requirements.txt` before running the pipeline."
        ) from _MISSING_DIAGNOSTICS_DEPENDENCY


def adf_test(values: pd.Series, maxlag: int | None = None) -> dict[str, Any]:
    """Run the Augmented Dickey-Fuller test on non-missing observations."""
    require_diagnostics_dependencies()
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


def residual_diagnostics(fit_result) -> dict[str, float]:
    """Check residual autocorrelation and normality."""
    require_diagnostics_dependencies()
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
