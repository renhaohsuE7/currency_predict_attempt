"""
測試 RunManager

測試每次 pipeline 執行的目錄結構管理功能，
包含 operation sub-directory、manifest、continue_latest
"""

import json
import os
import pytest
from pathlib import Path

from currency_predictor.prediction.run_manager import RunManager


class TestRunManager:
    """RunManager 基本功能測試"""

    def test_init_default(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path))
        assert rm.base_dir == tmp_path
        assert rm.runs_dir == tmp_path / "runs"
        assert rm.run_id  # 自動生成的 run_id 不為空

    def test_init_custom_run_id(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test_run_001")
        assert rm.run_id == "test_run_001"
        assert rm.run_dir == tmp_path / "runs" / "test_run_001"

    def test_setup_creates_run_dir(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="20260404_120000")
        run_dir = rm.setup()

        assert run_dir.exists()
        assert run_dir == rm.run_dir

    def test_setup_saves_config_snapshot(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="20260404_120000")
        config = {"model_name": "patchtst_sklearn", "symbols": ["USDTWD=X"]}
        rm.setup(config=config)

        snapshot_path = rm.run_dir / "config_snapshot.json"
        assert snapshot_path.exists()

        with open(snapshot_path, "r") as f:
            saved = json.load(f)
        assert saved == config

    def test_setup_without_config(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test_no_config")
        rm.setup()

        snapshot_path = rm.run_dir / "config_snapshot.json"
        assert not snapshot_path.exists()

    def test_setup_idempotent(self, tmp_path):
        """setup() 可重複呼叫（continue 模式用）"""
        rm = RunManager(base_dir=str(tmp_path), run_id="test_idempotent")
        rm.setup(config={"a": 1})
        rm.setup()  # 不應失敗
        assert rm.run_dir.exists()


class TestOperationLifecycle:
    """Operation sub-directory 生命週期測試"""

    def test_op_id_raises_without_start(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        with pytest.raises(RuntimeError, match="No operation started"):
            _ = rm.op_id

    def test_op_dir_raises_without_start(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        with pytest.raises(RuntimeError, match="No operation started"):
            _ = rm.op_dir

    def test_start_operation_returns_op_id(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        op_id = rm.start_operation(op_type="full", mode="compare")

        assert len(op_id) == 8
        assert op_id == rm.op_id

    def test_start_operation_creates_directories(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        rm.start_operation(op_type="full")

        assert rm.op_dir.exists()
        assert rm.models_dir.exists()
        assert rm.figures_dir.exists()
        assert rm.models_dir == rm.op_dir / "models"
        assert rm.figures_dir == rm.op_dir / "figures"

    def test_start_operation_writes_manifest(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        op_id = rm.start_operation(
            op_type="full",
            mode="compare",
            symbols=["USDTWD=X"],
            models=["patchtst_sklearn"],
        )

        manifest_path = rm.run_dir / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        assert manifest["run_id"] == "test"
        assert len(manifest["operations"]) == 1

        op = manifest["operations"][0]
        assert op["op_id"] == op_id
        assert op["type"] == "full"
        assert op["mode"] == "compare"
        assert op["symbols"] == ["USDTWD=X"]
        assert op["models"] == ["patchtst_sklearn"]
        assert op["status"] == "running"

    def test_complete_operation(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        rm.start_operation(op_type="full")
        rm.complete_operation()

        with open(rm.run_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        op = manifest["operations"][0]
        assert op["status"] == "completed"
        assert op["completed_at"] is not None

    def test_fail_operation(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()
        rm.start_operation(op_type="train_only")
        rm.fail_operation("模型訓練失敗")

        with open(rm.run_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        op = manifest["operations"][0]
        assert op["status"] == "failed"
        assert op["error"] == "模型訓練失敗"

    def test_multiple_operations_in_same_run(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()

        op1 = rm.start_operation(op_type="full")
        rm.complete_operation()

        op2 = rm.start_operation(op_type="train_only")
        rm.complete_operation()

        assert op1 != op2

        with open(rm.run_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        assert len(manifest["operations"]) == 2
        assert manifest["operations"][0]["op_id"] == op1
        assert manifest["operations"][1]["op_id"] == op2

    def test_model_source_op_id_recorded(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()

        rm.start_operation(op_type="predict_only", model_source_op_id="abc12345")

        with open(rm.run_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        assert manifest["operations"][0]["model_source_op_id"] == "abc12345"


class TestGetLatestModelPath:
    """模型路徑查找測試"""

    def _create_run_with_models(self, tmp_path):
        """建立一個有模型的 run"""
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()

        # Operation 1: full — 儲存 sklearn model
        op1 = rm.start_operation(op_type="full", symbols=["USDTWD=X"])
        model_file = rm.models_dir / "USDTWD_patchtst_sklearn.joblib"
        model_file.write_text("model_v1")
        rm.complete_operation()

        # Operation 2: train_only — 儲存新版 sklearn model
        op2 = rm.start_operation(op_type="train_only", symbols=["USDTWD=X"])
        model_file2 = rm.models_dir / "USDTWD_patchtst_sklearn.joblib"
        model_file2.write_text("model_v2")
        rm.complete_operation()

        return rm, op1, op2

    def test_find_latest_model(self, tmp_path):
        rm, op1, op2 = self._create_run_with_models(tmp_path)

        path = rm.get_latest_model_path("USDTWD", "patchtst_sklearn")
        assert path is not None
        assert op2 in str(path)  # 應該找到第二次操作的模型
        assert path.read_text() == "model_v2"

    def test_find_model_by_source_op_id(self, tmp_path):
        rm, op1, op2 = self._create_run_with_models(tmp_path)

        path = rm.get_latest_model_path("USDTWD", "patchtst_sklearn", source_op_id=op1)
        assert path is not None
        assert op1 in str(path)
        assert path.read_text() == "model_v1"

    def test_model_not_found(self, tmp_path):
        rm, op1, op2 = self._create_run_with_models(tmp_path)

        path = rm.get_latest_model_path("EURUSD", "patchtst_sklearn")
        assert path is None

    def test_model_not_found_wrong_op_id(self, tmp_path):
        rm, op1, op2 = self._create_run_with_models(tmp_path)

        path = rm.get_latest_model_path("USDTWD", "patchtst_sklearn", source_op_id="nonexist")
        assert path is None

    def test_skips_failed_operations(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="test")
        rm.setup()

        # Operation 1: completed
        op1 = rm.start_operation(op_type="full")
        (rm.models_dir / "USDTWD_patchtst_sklearn.joblib").write_text("good")
        rm.complete_operation()

        # Operation 2: failed
        op2 = rm.start_operation(op_type="train_only")
        (rm.models_dir / "USDTWD_patchtst_sklearn.joblib").write_text("bad")
        rm.fail_operation("crashed")

        path = rm.get_latest_model_path("USDTWD", "patchtst_sklearn")
        assert path is not None
        assert op1 in str(path)  # 跳過 failed，回到 op1


class TestRunManagerSymlink:
    """latest symlink 功能測試"""

    def test_symlink_without_operation(self, tmp_path):
        """沒有操作時 symlink 指向 run_dir"""
        rm = RunManager(base_dir=str(tmp_path), run_id="20260404_120000")
        rm.setup()
        rm.update_latest_symlink()

        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == rm.run_dir.resolve()

    def test_symlink_with_operation(self, tmp_path):
        """有操作時 symlink 指向 op_dir"""
        rm = RunManager(base_dir=str(tmp_path), run_id="20260404_120000")
        rm.setup()
        rm.start_operation(op_type="full")
        rm.update_latest_symlink()

        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == rm.op_dir.resolve()

    def test_symlink_replaces_old(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="run_001")
        rm.setup()
        rm.start_operation(op_type="full")
        rm.update_latest_symlink()

        rm2 = RunManager(base_dir=str(tmp_path), run_id="run_002")
        rm2.setup()
        rm2.start_operation(op_type="full")
        rm2.update_latest_symlink()

        latest = tmp_path / "latest"
        assert latest.resolve() == rm2.op_dir.resolve()

    def test_latest_symlink_is_relative(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="20260404_120000")
        rm.setup()
        rm.start_operation(op_type="full")
        rm.update_latest_symlink()

        latest = tmp_path / "latest"
        target = os.readlink(str(latest))
        assert not os.path.isabs(target), f"Symlink should be relative, got: {target}"


class TestRunManagerListRuns:
    """list_runs 功能測試"""

    def test_list_runs_empty(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path))
        assert rm.list_runs() == []

    def test_list_runs_with_runs(self, tmp_path):
        for rid in ["run_a", "run_b", "run_c"]:
            rm = RunManager(base_dir=str(tmp_path), run_id=rid)
            rm.setup()

        rm = RunManager(base_dir=str(tmp_path))
        runs = rm.list_runs()
        assert set(runs) == {"run_a", "run_b", "run_c"}

    def test_list_runs_excludes_symlinks(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="real_run")
        rm.setup()

        fake_link = rm.runs_dir / "fake_link"
        fake_link.symlink_to(rm.run_dir)

        runs = rm.list_runs()
        assert "real_run" in runs
        assert "fake_link" not in runs


class TestContinueLatest:
    """continue_latest 功能測試"""

    def test_continue_latest_returns_latest_run(self, tmp_path):
        for rid in ["20260401_100000", "20260402_100000", "20260403_100000"]:
            rm = RunManager(base_dir=str(tmp_path), run_id=rid)
            rm.setup()

        rm = RunManager.continue_latest(base_dir=str(tmp_path))
        assert rm.run_id == "20260403_100000"

    def test_continue_latest_loads_manifest(self, tmp_path):
        rm = RunManager(base_dir=str(tmp_path), run_id="20260404_120000")
        rm.setup()
        op_id = rm.start_operation(op_type="full", symbols=["USDTWD=X"])
        rm.complete_operation()

        rm2 = RunManager.continue_latest(base_dir=str(tmp_path))
        # 新操作應該 append 到既有 manifest
        rm2.start_operation(op_type="train_only")
        rm2.complete_operation()

        with open(rm2.run_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        assert len(manifest["operations"]) == 2
        assert manifest["operations"][0]["op_id"] == op_id

    def test_continue_latest_no_runs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunManager.continue_latest(base_dir=str(tmp_path))

    def test_continue_latest_empty_runs_dir_raises(self, tmp_path):
        (tmp_path / "runs").mkdir()
        with pytest.raises(FileNotFoundError):
            RunManager.continue_latest(base_dir=str(tmp_path))


class TestManifestPersistence:
    """Manifest 跨實例持久化測試"""

    def test_manifest_persists_across_instances(self, tmp_path):
        """不同 RunManager 實例共享同一 manifest"""
        rm1 = RunManager(base_dir=str(tmp_path), run_id="test")
        rm1.setup()
        op1 = rm1.start_operation(op_type="full")
        rm1.complete_operation()

        # 新實例讀取同一 run
        rm2 = RunManager(base_dir=str(tmp_path), run_id="test")
        op2 = rm2.start_operation(op_type="train_only")
        rm2.complete_operation()

        with open(rm2.run_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        assert len(manifest["operations"]) == 2
