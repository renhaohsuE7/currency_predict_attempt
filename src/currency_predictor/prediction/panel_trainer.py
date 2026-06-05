"""Multi-stock panel training.

Trains a single global PatchTST model on sequences pooled across many stocks,
then evaluates per symbol. Pooling multiplies the training-sample count, which
is the main lever for the Transformer-style models that need more data than a
single ticker's history provides.

Requires a return-space target (``model_training.target_transform="log_return"``)
so targets are comparable across symbols.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ..data_processor import DataProcessor
from ..models.factory import ModelFactory
from .predictor import CurrencyPredictor

logger = logging.getLogger(__name__)


class PanelTrainer:
    """Train one global model across a universe of symbols, evaluate per symbol."""

    def __init__(self, config: Dict[str, Any]):
        """Initialise the panel trainer from a config dict.

        Args:
            config: Full app config (model_name, model_params, capm,
                model_training, panel, ...).
        """
        self.config = config
        self.model_name: str = config.get("model_name", "patchtst_sklearn")
        self.model_params: Dict[str, Any] = config.get("model_params", {})
        self.capm_config: Dict[str, Any] = config.get("capm", {})
        training_cfg = config.get("model_training", {})
        self.target_transform: str = training_cfg.get("target_transform", "log_return")
        self.panel_cfg: Dict[str, Any] = config.get("panel", {})

        # CurrencyPredictor is reused purely for per-symbol data prep.
        self._prep = CurrencyPredictor(
            model_name=self.model_name,
            model_params=self.model_params,
            data_storage_path=config.get("data_storage_path", "data"),
            capm_config=self.capm_config,
            target_transform=self.target_transform,
        )

        # The global model trained on pooled sequences.
        self.model = ModelFactory.create_model(self.model_name, **self.model_params)

    def _window(self) -> Tuple[int, int]:
        """Return (seq_len, pred_len) from model params."""
        seq_len = self.model_params.get(
            "seq_len", self.model_params.get("context_length", 64)
        )
        pred_len = self.model_params.get(
            "pred_len", self.model_params.get("prediction_length", 15)
        )
        return int(seq_len), int(pred_len)

    def run(self) -> Dict[str, Any]:
        """Collect, pool-train a global model, and evaluate per symbol.

        Returns:
            Dict with keys: ``symbols``, ``feature_columns``, ``per_symbol``
            (symbol -> test metrics), ``aggregate`` (mean metrics), and
            ``n_train_symbols``.
        """
        if self.target_transform != "log_return":
            logger.warning(
                "Panel 訓練建議 target_transform='log_return'（跨股可比）；"
                f"目前為 '{self.target_transform}'。"
            )

        symbols: List[str] = self.panel_cfg.get("symbols", [])
        if not symbols:
            raise ValueError("panel.symbols 為空，無法進行 panel 訓練")

        period = self.config.get("model_training", {}).get("period", "10y")
        test_days = self.config.get("model_training", {}).get("test_days")
        min_history = self.panel_cfg.get("min_history_days", 500)
        explicit_cols = self.panel_cfg.get("feature_columns")

        # 事前驗證：在抓資料/訓練(可能數十分鐘)之前就擋下無法評估的設定
        seq_len, pred_len = self._window()
        if test_days is not None and test_days < seq_len + pred_len:
            raise ValueError(
                f"panel: test_days ({test_days}) < seq_len + pred_len "
                f"({seq_len}+{pred_len}={seq_len + pred_len})；測試集太短無法評估，"
                f"請將 model_training.test_days 設為 >= {seq_len + pred_len}。"
            )

        # 1) Collect + prepare each symbol (per-symbol split, return-space target)
        prepared: Dict[str, Tuple] = {}
        for sym in symbols:
            try:
                self._prep.collect_and_store_data([sym], period=period)
                X_tr, y_tr, X_te, y_te = self._prep.prepare_training_data(
                    sym, period, test_days=test_days
                )
            except Exception as e:  # noqa: BLE001 — skip unusable symbol, keep panel
                logger.warning(f"Panel 跳過 {sym}（資料準備失敗）: {e}")
                continue
            if len(X_tr) < min_history:
                logger.warning(
                    f"Panel 跳過 {sym}：訓練資料 {len(X_tr)} < min_history {min_history}"
                )
                continue
            prepared[sym] = (X_tr, y_tr, X_te, y_te)

        if not prepared:
            raise ValueError("Panel 訓練無任何可用股票（皆資料不足或失敗）")

        # 2) Align feature columns across all symbols (train frames drive schema)
        train_frames = [prepared[s][0] for s in prepared]
        _, common = DataProcessor.align_feature_columns(train_frames, explicit_cols)
        logger.info(
            f"Panel 共用特徵欄位 {len(common)} 個，universe {len(prepared)} 檔"
        )

        # 3) Build pooled datasets and fit the global model
        datasets = [(prepared[s][0][common], prepared[s][1]) for s in prepared]
        self.model.fit_panel(datasets)

        # 4) Evaluate per symbol with the global model (return space)
        per_symbol: Dict[str, Dict[str, float]] = {}
        for sym, (X_tr, y_tr, X_te, y_te) in prepared.items():
            X_te_aligned = X_te[common]
            try:
                if len(X_te_aligned) >= seq_len + pred_len:
                    res = self.model.evaluate_rolling(
                        X_te_aligned, y_te, seq_len, pred_len, y_train=y_tr
                    )
                    per_symbol[sym] = res["aggregate"]
                else:
                    per_symbol[sym] = self.model.evaluate_single_shot(
                        X_te_aligned, y_te, y_train=y_tr
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Panel 評估 {sym} 失敗: {e}")
                per_symbol[sym] = {}

        aggregate = self._aggregate(per_symbol)
        if not aggregate:
            logger.warning(
                "Panel 評估產出空 metrics —— 常見原因:test 視窗 < seq_len+pred_len "
                f"(seq_len={seq_len}, pred_len={pred_len});請將 model_training.test_days "
                f"設為 >= {seq_len + pred_len}。"
            )

        return {
            "symbols": list(prepared.keys()),
            "feature_columns": common,
            "per_symbol": per_symbol,
            "aggregate": aggregate,
            "n_train_symbols": len(prepared),
        }

    @staticmethod
    def _aggregate(per_symbol: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Mean of each metric across symbols (ignoring missing values)."""
        keys: set = set()
        for m in per_symbol.values():
            keys |= set(m.keys())
        agg: Dict[str, float] = {}
        for k in keys:
            vals = [
                m[k]
                for m in per_symbol.values()
                if isinstance(m.get(k), (int, float)) and np.isfinite(m[k])
            ]
            if vals:
                agg[k] = float(np.mean(vals))
        return agg
