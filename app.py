import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import traceback

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "INDPRO.csv"  
DATE_COL  = "observation_date"                
VALUE_COL = "INDPRO"                

TRAIN_RATIO = 0.80
MAX_P = 4
MAX_Q = 4
D = 1  
FORECAST_STEPS_FUTURE = 10 

# ----------------------------
# HELPERS
# ----------------------------
def load_series(path: str, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    s = df[value_col].astype(float)
    s = s.asfreq(pd.infer_freq(s.index))  # попробуем выставить частоту
    return s

def descriptive_stats(x: pd.Series) -> pd.Series:
    return pd.Series({
        "Mean": x.mean(),
        "Min": x.min(),
        "Max": x.max(),
        "Std": x.std(ddof=1),
        "Missing": x.isna().sum()
    })

def adf_test(x: pd.Series, maxlag: Optional[int] = None) -> Dict[str, object]:
    x_clean = x.dropna()
    res = adfuller(x_clean, maxlag=maxlag, autolag="AIC")
    out = {
        "ADF Statistic": res[0],
        "p-value": res[1],
        "usedlag": res[2],
        "nobs": res[3],
        "critical_values": res[4]
    }
    return out

def train_test_split_series(s: pd.Series, train_ratio: float) -> Tuple[pd.Series, pd.Series]:
    n = len(s.dropna())
    split = int(np.floor(n * train_ratio))
    s_clean = s.dropna()
    train = s_clean.iloc[:split]
    test  = s_clean.iloc[split:]
    return train, test

def fit_sarimax(train: pd.Series, order: Tuple[int,int,int]) -> SARIMAX:
    model = SARIMAX(
        train,
        order=order,
        trend="c",                 # как в реферате: константа, если среднее разности != 0
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def forecast_on_test(fit_res, train: pd.Series, test: pd.Series):
    # динамический прогноз на длину test
    fc = fit_res.get_forecast(steps=len(test))
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    mean_fc.index = test.index
    ci.index = test.index
    return mean_fc, ci

def metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    e = (y_true - y_pred).dropna()
    y_true = y_true.loc[e.index]
    y_pred = y_pred.loc[e.index]

    me = float(np.mean(e))
    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(np.abs(e)))

    # percentage errors (осторожно с нулями)
    eps = 1e-12
    pe = 100.0 * e / (y_true.replace(0, np.nan) + eps)
    mpe = float(np.nanmean(pe))
    mape = float(np.nanmean(np.abs(pe)))

    # Theil's U2 (сравнение с наивным прогнозом y_{t-1})
    naive = y_true.shift(1).dropna()
    common_idx = naive.index.intersection(y_true.index).intersection(y_pred.index)
    if len(common_idx) >= 2:
        num = np.sqrt(np.mean((y_true.loc[common_idx] - y_pred.loc[common_idx])**2))
        den = np.sqrt(np.mean((y_true.loc[common_idx] - naive.loc[common_idx])**2))
        u2 = float(num / den) if den > 0 else np.nan
    else:
        u2 = np.nan

    return {
        "ME": me,
        "RMSE": rmse,
        "MAE": mae,
        "MPE(%)": mpe,
        "MAPE(%)": mape,
        "Theil_U2": u2
    }

def residual_diagnostics(fit_res) -> Dict[str, object]:
    resid = pd.Series(fit_res.resid).dropna()

    # Ljung-Box на нескольких лагах
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])

    # Jarque-Bera
    jb_res = stats.jarque_bera(resid)
    # SciPy's jarque_bera may return (stat, pval) or (stat, pval, skew, kurt)
    if len(jb_res) == 4:
        jb_stat, jb_p, skew, kurt = jb_res
    else:
        jb_stat, jb_p = jb_res
        skew = float(stats.skew(resid))
        # Use non-excess kurtosis to match many implementations
        kurt = float(stats.kurtosis(resid, fisher=False))

    return {
        "LjungBox_pvalue(lag10)": lb_p,
        "JB_pvalue": float(jb_p),
        "Skew": float(skew),
        "Kurtosis": float(kurt),
        "ResidualMean": float(resid.mean()),
        "ResidualStd": float(resid.std(ddof=1))
    }

def plot_series(s: pd.Series, title: str):
    plt.figure()
    plt.plot(s.index, s.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

def plot_residuals(resid: pd.Series, title: str):
    plt.figure()
    plt.plot(resid.index, resid.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()

def plot_qq(resid: pd.Series, title: str):
    plt.figure()
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------------------------
# MODEL SEARCH (Information Criteria)
# ----------------------------
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def ic_grid_search(diff_train: pd.Series, max_p: int, max_q: int) -> pd.DataFrame:
    """
    Grid search for ARMA(p,q) on already differenced series diff_train.
    Since series is already differenced, we set d=0 in the model.
    """
    y = diff_train.dropna()

    rows = []
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
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)

                rows.append({
                    "p": p,
                    "q": q,
                    "AIC": float(res.aic),
                    "BIC": float(res.bic),
                    "HQIC": float(res.hqic)
                })
            except Exception as e:
                continue

    if not rows:
        raise RuntimeError("ic_grid_search: no models were successfully estimated. "
                           "Try reducing max_p/max_q or check data quality.")

    out = pd.DataFrame(rows).sort_values(["BIC", "AIC"]).reset_index(drop=True)
    return out
# ----------------------------
# MAIN WORKFLOW
# ----------------------------
# 1) Load data
s = load_series(DATA_PATH, DATE_COL, VALUE_COL)

# 2) Preliminary analysis
plot_series(s, "Original time series")

# 3) Transformation: first difference
diff = s.diff(1)
plot_series(diff, "First difference Δx_t")

print("Descriptive stats of Δx_t:")
print(descriptive_stats(diff), "\n")

adf_res = adf_test(diff)
print("ADF test on Δx_t:")
print(adf_res, "\n")

# 4) ACF / PACF on stationary series (diff)
x = diff.dropna()
plt.figure()
plot_acf(x, lags=24)
plt.tight_layout()
plt.show()

plt.figure()
plot_pacf(x, lags=24, method="ywm")
plt.tight_layout()
plt.show()

# 5) Train/Test split on original series (как в реферате: split по времени)
train, test = train_test_split_series(s, TRAIN_RATIO)
print(f"Train: {train.index.min()} -> {train.index.max()}  (n={len(train)})")
print(f"Test : {test.index.min()} -> {test.index.max()}  (n={len(test)})\n")

# Для критериев будем искать ARMA(p,q) на Δtrain
diff_train = train.diff(1).dropna()

ic_table = ic_grid_search(diff_train, MAX_P, MAX_Q)
print("Top models by BIC/AIC (on Δtrain as ARMA(p,q)):")
print(ic_table.head(10), "\n")

best_by_bic = ic_table.iloc[0][["p","q"]].astype(int).tolist()
best_by_aic = ic_table.sort_values("AIC").iloc[0][["p","q"]].astype(int).tolist()
best_by_hq  = ic_table.sort_values("HQIC").iloc[0][["p","q"]].astype(int).tolist()

candidates = []
candidates.append(("Best_BIC", tuple(best_by_bic)))
candidates.append(("Best_AIC", tuple(best_by_aic)))
candidates.append(("Best_HQIC", tuple(best_by_hq)))

# Уберём дубликаты
seen = set()
TOP_K = 3

top_models = ic_table.sort_values("BIC").head(TOP_K)

uniq_candidates = []
for i, row in top_models.iterrows():
    uniq_candidates.append((f"Candidate_{i+1}", int(row["p"]), int(row["q"])))


print("Candidate models (Δx_t ~ ARMA(p,q) => x_t ~ ARIMA(p,1,q)):")
print(uniq_candidates, "\n")

# 6) Fit candidate ARIMA on TRAIN and evaluate on TEST
results = []
for label, p, q in uniq_candidates:
    order = (p, D, q)
    try:
        fit_res = fit_sarimax(train, order=order)
        pred_mean, pred_ci = forecast_on_test(fit_res, train, test)
        m = metrics(test, pred_mean)

        diag = residual_diagnostics(fit_res)

        results.append({
            "Model": f"ARIMA({p},{D},{q}) [{label}]",
            **m,
            "AIC": fit_res.aic,
            "BIC": fit_res.bic,
            "HQIC": fit_res.hqic,
            "LB_p(lag10)": diag["LjungBox_pvalue(lag10)"],
            "JB_p": diag["JB_pvalue"]
        })

        # plots like in report
        resid = pd.Series(fit_res.resid).dropna()
        plot_residuals(resid, f"Residuals vs time: ARIMA({p},{D},{q})")
        plt.figure(); plot_acf(resid, lags=24); plt.tight_layout(); plt.show()
        plot_qq(resid, f"QQ plot: ARIMA({p},{D},{q})")

        plt.figure()
        plt.plot(train.index, train.values, label="Train")
        plt.plot(test.index, test.values, label="Test")
        plt.plot(pred_mean.index, pred_mean.values, label="Forecast")
        plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], alpha=0.2)
        plt.title(f"Forecast on test: ARIMA({p},{D},{q})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Failed model ARIMA({p},{D},{q}): {e}")
        traceback.print_exc()

if not results:
    print("No candidate models were successfully estimated. Exiting.")
    sys.exit(1)

res_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("Model comparison (sorted by RMSE):")
print(res_df, "\n")

# 7) Pick best model and refit on full sample; forecast future
best_model_str = res_df.iloc[0]["Model"]
print("Best model:", best_model_str)

# Parse p,d,q from string
import re
m = re.search(r"ARIMA\((\d+),(\d+),(\d+)\)", best_model_str)
p_best, d_best, q_best = map(int, m.groups())

best_fit_full = fit_sarimax(s.dropna(), order=(p_best, d_best, q_best))
future_fc = best_fit_full.get_forecast(steps=FORECAST_STEPS_FUTURE)
future_mean = future_fc.predicted_mean
future_ci = future_fc.conf_int(alpha=0.05)

print("Future forecast:")
print(pd.DataFrame({
    "forecast": future_mean,
    "lower_95": future_ci.iloc[:, 0],
    "upper_95": future_ci.iloc[:, 1]
}), "\n")

plt.figure()
plt.plot(s.index, s.values, label="Observed")
plt.plot(future_mean.index, future_mean.values, label="Forecast")
plt.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1], alpha=0.2)
plt.title(f"Future forecast ({FORECAST_STEPS_FUTURE} steps): ARIMA({p_best},{d_best},{q_best})")
plt.legend()
plt.tight_layout()
plt.show()
