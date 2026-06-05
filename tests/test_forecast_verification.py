"""Tests for forecast verification — save_forecast_json + ForecastVerifier."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from currency_predictor.verification.verifier import ForecastVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_forecast_json(tmp_path: Path, symbol: str = "USDTWD=X", n_dates: int = 5) -> Path:
    """Create a sample forecast JSON file."""
    base_date = datetime(2026, 4, 10)
    dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, n_dates + 1)]

    forecast = {
        "symbol": symbol,
        "forecast_generated_at": base_date.isoformat(timespec="seconds"),
        "prediction_horizon": n_dates,
        "models": {
            "naive": {
                "last_known_date": base_date.strftime("%Y-%m-%d"),
                "last_known_value": 32.0,
                "prediction_dates": dates,
                "predictions": [32.0] * n_dates,
            },
            "patchtst_sklearn": {
                "last_known_date": base_date.strftime("%Y-%m-%d"),
                "last_known_value": 32.0,
                "prediction_dates": dates,
                "predictions": [32.0 + 0.1 * i for i in range(1, n_dates + 1)],
            },
        },
    }

    fpath = tmp_path / "forecast_USDTWD.json"
    with open(fpath, "w") as f:
        json.dump(forecast, f)
    return fpath


def _make_actual_data(dates: list[str], close_values: list[float]) -> pd.DataFrame:
    """Create a mock actual data DataFrame like yfinance returns."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame(
        {
            "Open": close_values,
            "High": [v + 0.5 for v in close_values],
            "Low": [v - 0.5 for v in close_values],
            "Close": close_values,
            "Volume": [10000] * len(close_values),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests: ForecastVerifier core logic
# ---------------------------------------------------------------------------


class TestForecastVerifierMetrics:
    """Test metric calculations in ForecastVerifier."""

    def test_verify_perfect_prediction(self, tmp_path):
        """Perfect predictions should have zero error metrics."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        # Set both models to predict exactly the actual values
        actual_values = [32.05, 31.95, 32.10]
        dates = forecast["models"]["naive"]["prediction_dates"]
        for model in forecast["models"].values():
            model["predictions"] = actual_values

        with open(fpath, "w") as f:
            json.dump(forecast, f)

        actual_df = _make_actual_data(dates, actual_values)

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        assert report["success"]
        for model_result in report["symbols"]["USDTWD=X"]["models"].values():
            m = model_result["metrics"]
            assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
            assert m["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_verify_with_known_errors(self, tmp_path):
        """Check RMSE/MAE with known error values."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        # Naive predicts [32, 32, 32], actual is [32.1, 31.9, 32.0]
        forecast["models"]["naive"]["predictions"] = [32.0, 32.0, 32.0]
        actual_values = [32.1, 31.9, 32.0]

        with open(fpath, "w") as f:
            json.dump(forecast, f)

        actual_df = _make_actual_data(dates, actual_values)

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        naive = report["symbols"]["USDTWD=X"]["models"]["naive"]
        m = naive["metrics"]
        errors = np.array([-0.1, 0.1, 0.0])
        expected_rmse = float(np.sqrt(np.mean(errors**2)))
        expected_mae = float(np.mean(np.abs(errors)))
        assert m["rmse"] == pytest.approx(expected_rmse, abs=1e-6)
        assert m["mae"] == pytest.approx(expected_mae, abs=1e-6)

    def test_verify_n_matched(self, tmp_path):
        """n_matched should equal number of dates with actual data."""
        fpath = _make_forecast_json(tmp_path, n_dates=5)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        # Only provide actual data for first 3 dates
        actual_df = _make_actual_data(dates[:3], [32.0, 32.1, 31.9])

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        naive = report["symbols"]["USDTWD=X"]["models"]["naive"]
        assert naive["n_matched"] == 3

    def test_verify_mape_calculation(self, tmp_path):
        """MAPE should be calculated correctly."""
        fpath = _make_forecast_json(tmp_path, n_dates=2)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        forecast["models"]["naive"]["predictions"] = [32.0, 32.0]
        actual_values = [32.0, 31.0]  # 0% and ~3.23% error

        with open(fpath, "w") as f:
            json.dump(forecast, f)

        actual_df = _make_actual_data(dates, actual_values)

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        naive = report["symbols"]["USDTWD=X"]["models"]["naive"]
        # errors: [0.0, 1.0], actuals: [32.0, 31.0]
        # MAPE = mean(|0/32|, |1/31|) * 100 = mean(0, 3.226) * 100
        expected_mape = (0.0 + abs(1.0 / 31.0)) / 2 * 100
        assert naive["metrics"]["mape"] == pytest.approx(expected_mape, abs=0.01)


class TestForecastVerifierCoverage:
    """Test date matching and coverage."""

    def test_partial_coverage(self, tmp_path):
        """Only some dates have actual data — should still compute metrics."""
        fpath = _make_forecast_json(tmp_path, n_dates=5)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        actual_df = _make_actual_data(dates[:2], [32.0, 32.05])

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        sym = report["symbols"]["USDTWD=X"]
        assert sym["coverage"]["total"] == 5
        assert sym["coverage"]["matched"] == 2
        assert sym["coverage"]["missing"] == 3

    def test_no_actual_data_available(self, tmp_path):
        """No actual data available — should return error."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=None):
            report = verifier.verify(forecast_path=str(fpath))

        sym = report["symbols"]["USDTWD=X"]
        assert "error" in sym

    def test_nontrading_days_skipped(self, tmp_path):
        """Dates without actual data (weekends) are naturally skipped."""
        fpath = _make_forecast_json(tmp_path, n_dates=5)
        with open(fpath) as f:
            forecast = json.load(f)

        # Only provide data for 3 out of 5 dates (simulating weekends)
        dates = forecast["models"]["naive"]["prediction_dates"]
        trading_dates = [dates[0], dates[1], dates[3]]
        actual_df = _make_actual_data(trading_dates, [32.0, 31.95, 32.1])

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        naive = report["symbols"]["USDTWD=X"]["models"]["naive"]
        assert naive["n_matched"] == 3


class TestForecastVerifierDirectional:
    """Test directional accuracy computation."""

    def test_directional_accuracy_all_correct(self, tmp_path):
        """All directions predicted correctly → 100%."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        # last_known = 32.0, actuals go up then down then up
        actual_values = [32.2, 32.0, 32.3]
        # predictions also go up then down then up
        forecast["models"]["naive"]["predictions"] = [32.1, 31.9, 32.4]
        forecast["models"]["naive"]["last_known_value"] = 32.0

        with open(fpath, "w") as f:
            json.dump(forecast, f)

        actual_df = _make_actual_data(dates, actual_values)

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        naive = report["symbols"]["USDTWD=X"]["models"]["naive"]
        assert naive["metrics"]["directional_accuracy"] == pytest.approx(1.0)

    def test_directional_accuracy_all_wrong(self, tmp_path):
        """All directions predicted wrong → 0%."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        # last_known = 32.0, actuals go up then down then up
        actual_values = [32.2, 32.0, 32.3]
        # predictions go opposite: down then up then down
        forecast["models"]["naive"]["predictions"] = [31.9, 32.5, 31.8]
        forecast["models"]["naive"]["last_known_value"] = 32.0

        with open(fpath, "w") as f:
            json.dump(forecast, f)

        actual_df = _make_actual_data(dates, actual_values)

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        naive = report["symbols"]["USDTWD=X"]["models"]["naive"]
        assert naive["metrics"]["directional_accuracy"] == pytest.approx(0.0)


class TestForecastVerifierFileDiscovery:
    """Test finding forecast files by run_id/op_id."""

    def test_find_by_forecast_path(self, tmp_path):
        """Direct forecast_path should work."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        actual_df = _make_actual_data(dates, [32.0, 32.1, 31.9])

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        assert report["success"]
        assert "USDTWD=X" in report["symbols"]

    def test_find_by_run_id(self, tmp_path):
        """Should find forecast JSON via run_id."""
        # Create run directory structure
        run_dir = tmp_path / "runs" / "20260413_022039"
        op_dir = run_dir / "a2828854"
        op_dir.mkdir(parents=True)

        # Create manifest
        manifest = {
            "run_id": "20260413_022039",
            "operations": [
                {"op_id": "a2828854", "status": "completed"}
            ],
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create forecast JSON in op_dir
        fpath = _make_forecast_json(op_dir, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        actual_df = _make_actual_data(dates, [32.0, 32.1, 31.9])

        verifier = ForecastVerifier(base_dir=str(tmp_path))
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(run_id="20260413_022039")

        assert report["success"]
        assert "USDTWD=X" in report["symbols"]

    def test_find_latest_run(self, tmp_path):
        """Without run_id, should use latest run."""
        # Create two runs — should find the newer one
        for run_id in ["20260410_010000", "20260413_020000"]:
            run_dir = tmp_path / "runs" / run_id
            op_dir = run_dir / "op001"
            op_dir.mkdir(parents=True)
            manifest = {
                "run_id": run_id,
                "operations": [
                    {"op_id": "op001", "status": "completed"}
                ],
            }
            with open(run_dir / "manifest.json", "w") as f:
                json.dump(manifest, f)

        # Only create forecast in the latest run
        fpath = _make_forecast_json(
            tmp_path / "runs" / "20260413_020000" / "op001", n_dates=2
        )
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        actual_df = _make_actual_data(dates, [32.0, 32.1])

        verifier = ForecastVerifier(base_dir=str(tmp_path))
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify()

        assert report["success"]

    def test_no_forecast_files_returns_error(self, tmp_path):
        """No forecast files found should return error."""
        verifier = ForecastVerifier(base_dir=str(tmp_path))
        report = verifier.verify()
        assert not report["success"]
        assert "error" in report


class TestForecastVerifierReport:
    """Test report formatting."""

    def test_format_report_output(self, tmp_path):
        """format_report should produce readable text."""
        fpath = _make_forecast_json(tmp_path, n_dates=3)
        with open(fpath) as f:
            forecast = json.load(f)

        dates = forecast["models"]["naive"]["prediction_dates"]
        actual_df = _make_actual_data(dates, [32.0, 32.1, 31.9])

        verifier = ForecastVerifier()
        with patch.object(verifier.collector, "get_currency_data_range", return_value=actual_df):
            report = verifier.verify(forecast_path=str(fpath))

        text = verifier.format_report(report)
        assert "Forecast Verification Report" in text
        assert "USDTWD=X" in text
        assert "RMSE" in text
        assert "naive" in text

    def test_save_verification_report(self, tmp_path):
        """save_verification_report should write valid JSON."""
        report = {
            "success": True,
            "verification_date": "2026-04-15T10:00:00",
            "symbols": {},
        }

        out_path = tmp_path / "verification_report.json"
        verifier = ForecastVerifier()
        verifier.save_verification_report(report, out_path)

        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["success"] is True


# ---------------------------------------------------------------------------
# Tests: save_forecast_json (comparer)
# ---------------------------------------------------------------------------


class TestSaveForecastJson:
    """Test ModelComparer.save_forecast_json() output format."""

    def test_forecast_json_structure(self, tmp_path):
        """Forecast JSON should have correct top-level keys."""
        from currency_predictor.prediction.comparer import ModelComparer

        comparison_results = {
            "symbols_results": {
                "USDTWD=X": {
                    "models": {
                        "naive": {
                            "predictions": np.array([32.0, 32.0, 32.0]),
                            "prediction_dates": [
                                datetime(2026, 4, 11),
                                datetime(2026, 4, 12),
                                datetime(2026, 4, 13),
                            ],
                            "last_known_date": datetime(2026, 4, 10),
                            "last_known_value": 31.98,
                        },
                    },
                },
            },
        }

        comparer = ModelComparer(
            model_names=["naive"],
            config={"model_params": {}},
            output_dir=str(tmp_path),
        )
        comparer.save_forecast_json(comparison_results, tmp_path)

        json_path = tmp_path / "forecast_USDTWD.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert data["symbol"] == "USDTWD=X"
        assert "forecast_generated_at" in data
        assert data["prediction_horizon"] == 3
        assert "naive" in data["models"]
        naive = data["models"]["naive"]
        assert naive["last_known_date"] == "2026-04-10"
        assert naive["last_known_value"] == pytest.approx(31.98)
        assert len(naive["prediction_dates"]) == 3
        assert len(naive["predictions"]) == 3

    def test_forecast_json_skips_errored_models(self, tmp_path):
        """Models with errors should be skipped."""
        from currency_predictor.prediction.comparer import ModelComparer

        comparison_results = {
            "symbols_results": {
                "USDTWD=X": {
                    "models": {
                        "patchtst_sklearn": {
                            "error": "Training failed",
                        },
                        "naive": {
                            "predictions": np.array([32.0]),
                            "prediction_dates": [datetime(2026, 4, 11)],
                            "last_known_date": datetime(2026, 4, 10),
                            "last_known_value": 31.98,
                        },
                    },
                },
            },
        }

        comparer = ModelComparer(
            model_names=["naive", "sklearn"],
            config={"model_params": {}},
            output_dir=str(tmp_path),
        )
        comparer.save_forecast_json(comparison_results, tmp_path)

        json_path = tmp_path / "forecast_USDTWD.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "patchtst_sklearn" not in data["models"]
        assert "naive" in data["models"]


# ---------------------------------------------------------------------------
# Tests: get_currency_data_range
# ---------------------------------------------------------------------------


class TestGetCurrencyDataRange:
    """Test YahooFinanceCollector.get_currency_data_range() (mocked)."""

    def test_calls_yfinance_with_start_end(self):
        """Should call ticker.history with start and end params."""
        from currency_predictor.data.collectors import YahooFinanceCollector

        collector = YahooFinanceCollector()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [32.0],
                "High": [32.5],
                "Low": [31.5],
                "Close": [32.1],
                "Volume": [10000],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-04-11")]),
        )

        with patch("currency_predictor.data.collectors.yf.Ticker", return_value=mock_ticker):
            result = collector.get_currency_data_range(
                "USDTWD=X", "2026-04-11", "2026-04-15"
            )

        mock_ticker.history.assert_called_once_with(
            start="2026-04-11", end="2026-04-15", interval="1d"
        )
        assert result is not None
        assert len(result) == 1

    def test_returns_none_for_empty_data(self):
        """Should return None when no data available."""
        from currency_predictor.data.collectors import YahooFinanceCollector

        collector = YahooFinanceCollector()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("currency_predictor.data.collectors.yf.Ticker", return_value=mock_ticker):
            result = collector.get_currency_data_range(
                "INVALID=X", "2026-04-11", "2026-04-15"
            )

        assert result is None

    def test_accepts_datetime_objects(self):
        """Should accept datetime objects as start/end."""
        from currency_predictor.data.collectors import YahooFinanceCollector

        collector = YahooFinanceCollector()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [32.0],
                "High": [32.5],
                "Low": [31.5],
                "Close": [32.1],
                "Volume": [10000],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-04-11")]),
        )

        with patch("currency_predictor.data.collectors.yf.Ticker", return_value=mock_ticker):
            result = collector.get_currency_data_range(
                "USDTWD=X",
                datetime(2026, 4, 11),
                datetime(2026, 4, 15),
            )

        mock_ticker.history.assert_called_once_with(
            start="2026-04-11", end="2026-04-15", interval="1d"
        )
        assert result is not None
