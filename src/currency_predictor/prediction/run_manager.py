"""
執行管理器

管理每次 pipeline 執行的目錄結構與產出，
以 datetime 時間戳區分不同次的執行結果，
以 operation sub-directory 區分同一 run 內的不同操作。
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RunManager:
    """管理每次執行的目錄與產出

    每次執行（run）建立獨立的時間戳目錄，
    每次操作（operation）在 run 目錄內建立獨立的 op_id 子目錄：

        results/runs/{YYYYMMDD_HHMMSS}/
            ├── config_snapshot.json       # per-run
            ├── manifest.json              # 追蹤所有操作
            ├── {op_id}/                   # operation 子目錄
            │   ├── models/
            │   ├── figures/
            │   ├── pipeline_results.json
            │   └── prediction_report.md
            └── {op_id_2}/                 # 另一次操作
                └── models/

    results/latest symlink 指向最新操作目錄。
    """

    def __init__(self, base_dir: str = "results", run_id: Optional[str] = None):
        """
        Args:
            base_dir: 結果根目錄
            run_id: 可指定 run ID（主要用於測試），預設自動生成時間戳
        """
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.runs_dir / self.run_id
        self._current_op_id: Optional[str] = None
        self._manifest: Optional[Dict[str, Any]] = None

    def setup(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """建立 run 目錄結構並儲存 config 快照

        Args:
            config: 本次執行的配置字典（可選）

        Returns:
            run 目錄路徑
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            snapshot_path = self.run_dir / "config_snapshot.json"
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Config snapshot saved to: {snapshot_path}")

        logger.info(f"Run directory created: {self.run_dir}")
        return self.run_dir

    # ------------------------------------------------------------------
    # Operation lifecycle
    # ------------------------------------------------------------------

    @property
    def op_id(self) -> str:
        """當前操作 ID。需先呼叫 start_operation()。"""
        if self._current_op_id is None:
            raise RuntimeError("No operation started. Call start_operation() first.")
        return self._current_op_id

    @property
    def op_dir(self) -> Path:
        """當前操作的子目錄"""
        return self.run_dir / self.op_id

    @property
    def models_dir(self) -> Path:
        """模型儲存目錄（在操作子目錄內）"""
        return self.op_dir / "models"

    @property
    def figures_dir(self) -> Path:
        """圖表儲存目錄（在操作子目錄內）"""
        return self.op_dir / "figures"

    def start_operation(
        self,
        op_type: str,
        mode: str = "single",
        symbols: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        model_source_op_id: Optional[str] = None,
    ) -> str:
        """開始新操作，生成 op_id 並建立子目錄

        Args:
            op_type: 操作類型 ("full" / "train_only" / "predict_only")
            mode: 執行模式 ("single" / "compare")
            symbols: 操作的符號列表
            models: 操作的模型列表
            model_source_op_id: predict_only 時指定模型來源操作 ID

        Returns:
            op_id (8 字元 hex)
        """
        self._current_op_id = uuid.uuid4().hex[:8]

        # 建立操作子目錄
        self.op_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 寫入 manifest
        op_entry: Dict[str, Any] = {
            "op_id": self._current_op_id,
            "type": op_type,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "mode": mode,
            "symbols": symbols or [],
            "models": models or [],
            "status": "running",
        }
        if model_source_op_id:
            op_entry["model_source_op_id"] = model_source_op_id

        self._load_manifest()
        self._manifest["operations"].append(op_entry)
        self._save_manifest()

        logger.info(f"Operation started: {self._current_op_id} ({op_type})")
        return self._current_op_id

    def complete_operation(self) -> None:
        """標記當前操作為已完成"""
        self._load_manifest()
        for op in self._manifest["operations"]:
            if op["op_id"] == self.op_id:
                op["completed_at"] = datetime.now().isoformat()
                op["status"] = "completed"
                break
        self._save_manifest()
        logger.info(f"Operation completed: {self.op_id}")

    def fail_operation(self, error: str) -> None:
        """標記當前操作為失敗"""
        self._load_manifest()
        for op in self._manifest["operations"]:
            if op["op_id"] == self.op_id:
                op["completed_at"] = datetime.now().isoformat()
                op["status"] = "failed"
                op["error"] = error
                break
        self._save_manifest()
        logger.warning(f"Operation failed: {self.op_id} — {error}")

    # ------------------------------------------------------------------
    # Model lookup
    # ------------------------------------------------------------------

    def get_latest_model_path(
        self,
        symbol: str,
        model_name: str,
        source_op_id: Optional[str] = None,
    ) -> Optional[Path]:
        """查找指定 symbol + model 的最新模型檔案

        Args:
            symbol: 金融符號（clean 版，如 "USDTWD"）
            model_name: 模型名稱（如 "patchtst_sklearn"）
            source_op_id: 指定操作 ID（省略則自動找最新）

        Returns:
            模型檔案路徑，找不到時回傳 None
        """
        self._load_manifest()
        ops = self._manifest.get("operations", [])

        # 確定要搜尋的操作列表
        if source_op_id:
            candidates = [op for op in ops if op["op_id"] == source_op_id]
        else:
            # 反向掃描，找最新的已完成訓練操作
            candidates = [
                op for op in reversed(ops)
                if op["type"] in ("full", "train_only") and op["status"] == "completed"
            ]

        for op in candidates:
            op_models_dir = self.run_dir / op["op_id"] / "models"
            if not op_models_dir.exists():
                continue

            # 尋找匹配的模型檔案
            for f in op_models_dir.iterdir():
                if symbol in f.name and model_name in f.name:
                    return f

            # 也檢查子目錄（HuggingFace/Lightning 模型存為目錄）
            for d in op_models_dir.iterdir():
                if d.is_dir() and symbol in d.name and model_name in d.name:
                    return d

        return None

    # ------------------------------------------------------------------
    # Symlink & listing
    # ------------------------------------------------------------------

    def update_latest_symlink(self) -> None:
        """更新 results/latest symlink 指向最新操作目錄（有操作時）或 run 目錄"""
        if self._current_op_id:
            target = self.run_dir / self._current_op_id
        else:
            target = self.run_dir
        latest_link = self.base_dir / "latest"

        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()

            relative_target = os.path.relpath(target, self.base_dir)
            latest_link.symlink_to(relative_target)
            logger.info(f"Updated latest symlink -> {relative_target}")
        except OSError as e:
            logger.warning(f"Failed to create latest symlink: {e}")

    def list_runs(self) -> list:
        """列出所有歷史 run（按時間排序）

        Returns:
            run ID 列表（最新在最後）
        """
        if not self.runs_dir.exists():
            return []

        runs = [
            d.name for d in sorted(self.runs_dir.iterdir())
            if d.is_dir() and not d.is_symlink()
        ]
        return runs

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """從磁碟載入 manifest，不存在則初始化"""
        if self._manifest is not None:
            return

        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {
                "run_id": self.run_id,
                "created_at": datetime.now().isoformat(),
                "operations": [],
            }

    def _save_manifest(self) -> None:
        """將 manifest 寫入磁碟"""
        if self._manifest is None:
            return

        manifest_path = self.run_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Continue existing run
    # ------------------------------------------------------------------

    @classmethod
    def continue_latest(cls, base_dir: str = "results") -> "RunManager":
        """建立 RunManager 並重用最新的 run 目錄

        Args:
            base_dir: 結果根目錄

        Returns:
            指向最新 run 的 RunManager

        Raises:
            FileNotFoundError: 找不到任何既有 run
        """
        runs_dir = Path(base_dir) / "runs"
        if not runs_dir.exists():
            raise FileNotFoundError(f"No runs directory found at {runs_dir}")

        existing_runs = sorted(
            d.name for d in runs_dir.iterdir()
            if d.is_dir() and not d.is_symlink()
        )
        if not existing_runs:
            raise FileNotFoundError("No existing runs found")

        latest_run_id = existing_runs[-1]
        logger.info(f"Continuing latest run: {latest_run_id}")
        return cls(base_dir=base_dir, run_id=latest_run_id)
