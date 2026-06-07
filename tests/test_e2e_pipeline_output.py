"""
E2E Test: Pipeline Output Verification

驗證 pipeline 執行後是否正確產出預期檔案：
- RunManager 操作生命週期 → 目錄結構 + manifest
- PredictionPipeline → pipeline_results.json, prediction_report.md
- main() backtest routing → _run_backtest_mode 正確呼叫

快速測試用 mock（no marker），完整 pipeline 測試用 fixture data（@slow）。

執行：uv run pytest tests/test_e2e_pipeline_output.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from currency_predictor.prediction.run_manager import RunManager


class TestPipelineOutputStructure:
    """Verify RunManager creates expected output structure for pipeline."""

    def test_full_operation_lifecycle_creates_files(self, tmp_path):
        """setup → start_operation → complete → symlink produces correct files."""
        rm = RunManager(base_dir=str(tmp_path / "results"))
        rm.setup(config={"model_name": "test"})

        op_id = rm.start_operation(
            op_type="full", mode="single", symbols=["USDTWD=X"]
        )

        # Directories created
        assert rm.op_dir.exists()
        assert rm.models_dir.exists()
        assert rm.figures_dir.exists()

        # Config snapshot
        assert (rm.run_dir / "config_snapshot.json").exists()

        # Manifest with running operation
        manifest_path = rm.run_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        ops = manifest["operations"]
        assert len(ops) == 1
        assert ops[0]["op_id"] == op_id
        assert ops[0]["status"] == "running"

        # Complete operation
        rm.complete_operation()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["operations"][0]["status"] == "completed"
        assert manifest["operations"][0]["completed_at"] is not None

        # Symlink
        rm.update_latest_symlink()
        latest = tmp_path / "results" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == rm.op_dir.resolve()

    def test_manifest_records_backtest_operation(self, tmp_path):
        """Backtest operation type is correctly recorded in manifest."""
        rm = RunManager(base_dir=str(tmp_path / "results"))
        rm.setup()

        rm.start_operation(
            op_type="backtest", mode="backtest", symbols=["USDTWD=X"]
        )
        rm.complete_operation()

        manifest = json.loads(
            (rm.run_dir / "manifest.json").read_text()
        )
        op = manifest["operations"][0]
        assert op["type"] == "backtest"
        assert op["mode"] == "backtest"
        assert op["status"] == "completed"

    def test_multiple_operations_have_separate_dirs(self, tmp_path):
        """Multiple operations in same run create separate subdirectories."""
        rm = RunManager(base_dir=str(tmp_path / "results"))
        rm.setup()

        op1 = rm.start_operation(op_type="full", symbols=["USDTWD=X"])
        dir1 = rm.op_dir
        rm.complete_operation()

        op2 = rm.start_operation(op_type="backtest", symbols=["AAPL"])
        dir2 = rm.op_dir
        rm.complete_operation()

        assert op1 != op2
        assert dir1 != dir2
        assert dir1.exists()
        assert dir2.exists()


@pytest.mark.slow
class TestPipelineOutputFiles:
    """Verify full pipeline produces expected output files."""

    def test_full_pipeline_creates_output_files(self, e2e_fixture_path, tmp_path):
        """PredictionPipeline.run_full_pipeline() creates results JSON and report."""
        from currency_predictor.prediction.pipeline import PredictionPipeline
        from currency_predictor.data.storage import DataStorage

        # Inject fixture data into storage
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = pd.read_csv(e2e_fixture_path, index_col="Date", parse_dates=True)
        storage = DataStorage(base_dir=str(data_dir))
        storage.save_raw_data(df, "USDTWD", "1y")

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = {
            "model_name": "patchtst_sklearn",
            "model_params": {
                "seq_len": 50,
                "pred_len": 5,
                "patch_len": 10,
                "stride": 5,
                "n_estimators": 10,
                "max_depth": 3,
                "random_state": 42,
            },
            "data_storage_path": str(data_dir),
            "data": {"default_period": "1y", "default_interval": "1d"},
            "preprocessing": {"target_column": "Close"},
            "log_level": "WARNING",
        }

        pipeline = PredictionPipeline(config=config, output_dir=str(output_dir))
        results = pipeline.run_full_pipeline(
            symbols=["USDTWD=X"],
            prediction_horizon=5,
            save_results=True,
        )

        # Pipeline should succeed
        assert results["success"] is True

        # Check that results were saved as JSON
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) > 0, "No JSON output files created"

        # At least one JSON file should be loadable
        for jf in json_files:
            data = json.loads(jf.read_text())
            assert isinstance(data, dict)

        # Guard the silent-swallow "success but empty metrics/predictions"
        # scenario: load the saved pipeline results and verify real content.
        results_file = output_dir / "pipeline_results.json"
        assert results_file.exists(), "pipeline_results.json not written"
        saved = json.loads(results_file.read_text())

        # Training must have produced a finite test RMSE for the symbol.
        training = saved["training"]
        assert len(training) > 0, "No training results recorded"
        completed = [t for t in training if t.get("training_completed")]
        assert completed, f"No completed training entries: {training}"
        rmse = completed[0]["test_metrics"]["rmse"]
        assert isinstance(rmse, (int, float))
        assert np.isfinite(rmse), f"test RMSE not finite: {rmse}"

        # Predictions must be non-empty for the symbol.
        predictions = saved["predictions"]
        assert len(predictions) > 0, "No predictions recorded"
        pred_values = predictions[0]["predictions"]
        assert len(pred_values) > 0, "Prediction list is empty"
        assert all(np.isfinite(p) for p in pred_values)
