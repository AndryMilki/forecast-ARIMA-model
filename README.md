# Industrial Production Forecasting with ARIMA

This project analyzes the monthly U.S. Industrial Production Index (`INDPRO`) and compares simple forecasting baselines against ARIMA/SARIMAX candidates. It started as coursework and is now packaged as a portfolio-ready Python project with modular source code, generated artifacts, tests, CI, and an explanatory notebook.

## What the Project Shows

- Time-series loading, cleaning, and chronological train/test splitting.
- First-order differencing and Augmented Dickey-Fuller stationarity testing.
- ACF/PACF diagnostics for model identification.
- Baseline evaluation with naive and moving-average forecasts.
- ARMA grid search on the differenced series using AIC, BIC, and HQIC.
- ARIMA/SARIMAX candidate evaluation with RMSE, MAE, MAPE, Theil's U2, and residual diagnostics.
- Future forecasting with 95 percent confidence intervals.

## Dataset

The included `INDPRO.csv` file contains monthly FRED Industrial Production Index observations from January 1985 through November 2025.

Columns:

- `observation_date`: monthly timestamp
- `INDPRO`: industrial production index value

## Project Structure

```text
.
|-- .github/workflows/ci.yml
|-- app.py
|-- INDPRO.csv
|-- notebooks/
|   `-- 01_arima_indpro_analysis.ipynb
|-- reports/
|   |-- figures/
|   `-- tables/
|-- requirements.txt
|-- src/
|   |-- data.py
|   |-- preprocessing.py
|   |-- diagnostics.py
|   |-- modeling.py
|   |-- evaluation.py
|   |-- visualization.py
|   `-- pipeline.py
`-- tests/
    `-- test_app.py
```

`app.py` is only the CLI entry point. The reusable analysis logic lives in `src/`.

Generated artifacts are written to `reports/`:

- `reports/tables/baseline_comparison.csv`
- `reports/tables/ic_grid_search.csv`
- `reports/tables/model_comparison.csv`
- `reports/tables/future_forecast.csv`
- `reports/figures/*.png`

## Results Snapshot

The current dataset run produced the following headline results:

- Date range: January 1985 to November 2025
- Train/test split: 392 train observations and 99 test observations
- ADF test on first difference: statistic `-5.8802`, p-value `3.092e-07`
- Best overall test RMSE: `Naive forecast`
- Best ARIMA/SARIMAX test RMSE: `ARIMA(4,1,0) [best_bic]`
- Best ARIMA Ljung-Box p-value at lag 10: `0.5768`

Baseline comparison on the current multi-step test horizon:

| Model | Type | RMSE | MAE | MAPE (%) | Theil U2 |
|---|---:|---:|---:|---:|---:|
| Naive forecast | Baseline | 3.0293 | 2.0679 | 2.1084 | 1.8178 |
| Moving average (12) forecast | Baseline | 3.1511 | 2.3312 | 2.3636 | 1.8906 |
| ARIMA(4,1,0) [best_bic] | ARIMA/SARIMAX | 5.7159 | 4.9294 | 4.9795 | 3.4299 |
| ARIMA(1,1,3) [bic_rank_2] | ARIMA/SARIMAX | 5.8238 | 5.0201 | 5.0708 | 3.4946 |
| ARIMA(1,1,2) [bic_rank_3] | ARIMA/SARIMAX | 5.8868 | 5.0600 | 5.1119 | 3.5325 |

On this split, ARIMA does not beat the simple baselines by RMSE. That is an important modeling result: over a long multi-step horizon with crisis-period shocks, a univariate ARIMA extrapolation can be less competitive than a strong persistence baseline. The ARIMA residual autocorrelation check is still acceptable for the best ARIMA candidate, while the Jarque-Bera p-value is close to zero because large macroeconomic shocks create non-Gaussian residual tails.

![Industrial Production Index](reports/figures/series.png)

![Future forecast](reports/figures/future_forecast.png)

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --data INDPRO.csv --output-dir reports
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --data INDPRO.csv --output-dir reports
```

## Example Output

After running the pipeline, the console prints:

```text
Forecasting analysis complete
Dataset: INDPRO, 1985-01-01 to 2025-11-01
Train observations: 392 | Test observations: 99
ADF on first difference: statistic=-5.8802, p=3.092e-07
Best overall model by test RMSE: Naive forecast (RMSE=3.0293)
Best ARIMA model by test RMSE: ARIMA(4,1,0) [best_bic] (RMSE=5.7159)
Artifacts saved to: <project>\reports
```

The future forecast is produced by the best ARIMA/SARIMAX candidate and refit on the full sample.

## Notebook

The notebook [notebooks/01_arima_indpro_analysis.ipynb](notebooks/01_arima_indpro_analysis.ipynb) walks through:

1. Dataset overview
2. Time series plot
3. Stationarity
4. ACF/PACF
5. Model selection
6. Evaluation
7. Residual diagnostics
8. Forecast interpretation

## Testing and CI

Run tests locally:

```powershell
python -m unittest discover -s tests
```

GitHub Actions runs the same test command on every push and pull request through `.github/workflows/ci.yml`.

The tests cover loading, chronological splitting, descriptive statistics, forecast metrics, and baseline forecasts.

## Notes

The accompanying coursework document was used as the methodological reference for this repository: stationarity analysis, ARIMA identification, residual diagnostics, and forecast evaluation. The code is the source of truth for the current `INDPRO` dataset and regenerates the final results from scratch.
