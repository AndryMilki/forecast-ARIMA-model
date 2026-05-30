from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import load_series
from src.evaluation import metrics, moving_average_forecast, naive_forecast
from src.preprocessing import descriptive_stats, train_test_split_series


class AnalysisHelpersTest(unittest.TestCase):
    def test_load_series_sorts_and_aligns_monthly_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            pd.DataFrame(
                {
                    "observation_date": ["2020-03-01", "2020-01-01", "2020-02-01"],
                    "INDPRO": ["103.0", "101.0", "102.0"],
                }
            ).to_csv(path, index=False)

            series = load_series(path, "observation_date", "INDPRO")

        self.assertEqual(list(series.index.strftime("%Y-%m-%d")), [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
        ])
        self.assertEqual(series.index.freqstr, "MS")
        self.assertEqual(series.tolist(), [101.0, 102.0, 103.0])

    def test_train_test_split_is_chronological(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="MS")
        series = pd.Series(range(10), index=index)

        train, test = train_test_split_series(series, 0.8)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertEqual(train.index.max(), pd.Timestamp("2020-08-01"))
        self.assertEqual(test.index.min(), pd.Timestamp("2020-09-01"))

    def test_descriptive_stats_reports_missing_values(self) -> None:
        values = pd.Series([1.0, 2.0, np.nan, 4.0])

        stats = descriptive_stats(values)

        self.assertEqual(stats["missing"], 1)
        self.assertAlmostEqual(stats["mean"], 7.0 / 3.0)

    def test_metrics_returns_expected_error_values(self) -> None:
        index = pd.date_range("2020-01-01", periods=4, freq="MS")
        actual = pd.Series([10.0, 12.0, 14.0, 16.0], index=index)
        predicted = pd.Series([11.0, 11.0, 13.0, 15.0], index=index)

        result = metrics(actual, predicted)

        self.assertAlmostEqual(result["ME"], 0.5)
        self.assertAlmostEqual(result["RMSE"], 1.0)
        self.assertAlmostEqual(result["MAE"], 1.0)
        self.assertTrue(np.isfinite(result["Theil_U2"]))

    def test_naive_forecast_repeats_last_train_value(self) -> None:
        train = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
        test = pd.Series([4.0, 5.0], index=pd.date_range("2020-04-01", periods=2, freq="MS"))

        forecast = naive_forecast(train, test)

        self.assertEqual(forecast.tolist(), [3.0, 3.0])
        self.assertTrue(forecast.index.equals(test.index))

    def test_moving_average_forecast_uses_trailing_train_window(self) -> None:
        train = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
        test = pd.Series([5.0, 6.0], index=pd.date_range("2020-05-01", periods=2, freq="MS"))

        forecast = moving_average_forecast(train, test, window=2)

        self.assertEqual(forecast.tolist(), [3.5, 3.5])
        self.assertTrue(forecast.index.equals(test.index))


if __name__ == "__main__":
    unittest.main()
