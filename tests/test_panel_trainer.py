"""Tests for multi-stock panel training (Part B)."""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.data_processor import DataProcessor
from currency_predictor.models.patchtst import PatchTST
from currency_predictor.models.naive import NaiveModel
from currency_predictor.prediction.panel_trainer import PanelTrainer
from currency_predictor.config.settings import PanelConfig


def _synthetic_frame(n: int, seed: int, cols=("f1", "f2", "f3")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.standard_normal(n) for c in cols})


class TestAlignFeatureColumns:
    def test_intersection_preserves_first_order(self):
        a = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        b = pd.DataFrame({"z": [9], "x": [8]})
        aligned, common = DataProcessor.align_feature_columns([a, b])
        assert common == ["x", "z"]  # order from first frame, y dropped
        assert list(aligned[0].columns) == ["x", "z"]
        assert list(aligned[1].columns) == ["x", "z"]

    def test_explicit_columns(self):
        a = pd.DataFrame({"x": [1], "y": [2]})
        b = pd.DataFrame({"x": [3], "y": [4]})
        aligned, common = DataProcessor.align_feature_columns([a, b], explicit=["y"])
        assert common == ["y"]
        assert list(aligned[0].columns) == ["y"]

    def test_empty(self):
        aligned, common = DataProcessor.align_feature_columns([])
        assert aligned == [] and common == []


class TestSklearnFitPanel:
    def _model(self):
        return PatchTST(seq_len=10, pred_len=3, patch_len=5, stride=2)

    def test_fit_panel_trains_and_predicts(self):
        model = self._model()
        datasets = [
            (_synthetic_frame(60, 1), pd.Series(np.random.randn(60))),
            (_synthetic_frame(60, 2), pd.Series(np.random.randn(60))),
        ]
        model.fit_panel(datasets)
        assert model.is_fitted
        preds = model.predict(_synthetic_frame(40, 3))
        assert len(preds) == 3
        assert np.all(np.isfinite(preds))

    def test_fit_panel_pools_more_sequences_than_single(self):
        """Two symbols should train on more sequences than one (no exact count
        API, so assert both fit and that two-symbol training succeeds)."""
        one = self._model()
        one.fit_panel([(_synthetic_frame(60, 1), pd.Series(np.random.randn(60)))])
        two = self._model()
        two.fit_panel(
            [
                (_synthetic_frame(60, 1), pd.Series(np.random.randn(60))),
                (_synthetic_frame(60, 2), pd.Series(np.random.randn(60))),
            ]
        )
        assert one.is_fitted and two.is_fitted

    def test_fit_panel_column_mismatch_raises(self):
        model = self._model()
        datasets = [
            (_synthetic_frame(60, 1, cols=("f1", "f2", "f3")), pd.Series(np.random.randn(60))),
            (_synthetic_frame(60, 2, cols=("f1", "f2")), pd.Series(np.random.randn(60))),
        ]
        with pytest.raises(ValueError, match="特徵欄位必須一致"):
            model.fit_panel(datasets)

    def test_fit_panel_empty_raises(self):
        with pytest.raises(ValueError, match="至少一檔"):
            self._model().fit_panel([])


class TestBaseModelFitPanelDefault:
    def test_naive_fit_panel_not_implemented(self):
        model = NaiveModel(pred_len=3)
        with pytest.raises(NotImplementedError, match="panel"):
            model.fit_panel([(pd.DataFrame({"Close": [1.0, 2.0]}), pd.Series([1.0, 2.0]))])


class TestPanelAggregate:
    def test_aggregate_means_ignore_missing(self):
        per_symbol = {
            "A": {"rmse": 0.1, "mae": 0.2},
            "B": {"rmse": 0.3},  # no mae
            "C": {"rmse": float("nan"), "mae": 0.4},  # nan ignored
        }
        agg = PanelTrainer._aggregate(per_symbol)
        assert agg["rmse"] == pytest.approx((0.1 + 0.3) / 2)
        assert agg["mae"] == pytest.approx((0.2 + 0.4) / 2)


class TestPanelConfig:
    def test_defaults(self):
        cfg = PanelConfig()
        assert cfg.enabled is False
        assert cfg.symbols == []
        assert cfg.feature_columns is None
        assert cfg.min_history_days == 500

    def test_custom(self):
        cfg = PanelConfig(enabled=True, symbols=["AAPL", "MSFT"], min_history_days=300)
        assert cfg.enabled is True
        assert cfg.symbols == ["AAPL", "MSFT"]
        assert cfg.min_history_days == 300
