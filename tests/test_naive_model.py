"""Tests for NaiveModel."""

import numpy as np
import pandas as pd
import pytest

from currency_predictor.models.naive import NaiveModel
from currency_predictor.models.factory import ModelFactory


class TestNaiveModel:
    """Tests for the naive persistence baseline model."""

    def test_init(self):
        """Default pred_len is 24."""
        model = NaiveModel()
        assert model.pred_len == 24
        assert model.model_name == "NaiveModel"
        assert not model.is_fitted

    def test_init_custom_pred_len(self):
        model = NaiveModel(pred_len=7)
        assert model.pred_len == 7

    def test_fit_is_noop(self):
        """fit() should set is_fitted=True without doing real work."""
        model = NaiveModel()
        X = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 2.0, 3.0], name="Close")

        result = model.fit(X, y)
        assert result is model
        assert model.is_fitted
        assert model.training_history["epochs"] == 0

    def test_predict_repeats_last_close(self):
        """predict() should return pred_len copies of the last Close value."""
        model = NaiveModel(pred_len=5)
        X = pd.DataFrame({"Close": [10.0, 20.0, 30.0]})
        model.fit(X, pd.Series([10.0, 20.0, 30.0]))

        preds = model.predict(X)
        assert len(preds) == 5
        assert np.all(preds == 30.0)

    def test_predict_with_horizon(self):
        """horizon > 1 should override pred_len."""
        model = NaiveModel(pred_len=5)
        X = pd.DataFrame({"Close": [10.0, 20.0, 30.0]})
        model.fit(X, pd.Series([10.0, 20.0, 30.0]))

        preds = model.predict(X, horizon=3)
        assert len(preds) == 3
        assert np.all(preds == 30.0)

    def test_predict_with_extra_columns(self):
        """predict() should use Close even when other columns exist."""
        model = NaiveModel(pred_len=3)
        X = pd.DataFrame({
            "Open": [100.0, 200.0],
            "Close": [50.0, 75.0],
            "Volume": [1000, 2000],
        })
        model.fit(X, pd.Series([50.0, 75.0]))

        preds = model.predict(X)
        assert np.all(preds == 75.0)

    def test_predict_without_close_fallback(self):
        """predict() should fallback to first numeric column when no Close."""
        model = NaiveModel(pred_len=3)
        X = pd.DataFrame({"Price": [10.0, 20.0, 30.0]})
        model.fit(X, pd.Series([10.0, 20.0, 30.0]))

        preds = model.predict(X)
        assert np.all(preds == 30.0)

    def test_predict_raises_when_not_fitted(self):
        """predict() should raise when model is not fitted."""
        model = NaiveModel()
        X = pd.DataFrame({"Close": [1.0, 2.0]})

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_predict_with_uncertainty(self):
        """predict_with_uncertainty() should return zero uncertainty."""
        model = NaiveModel(pred_len=3)
        X = pd.DataFrame({"Close": [10.0, 20.0]})
        model.fit(X, pd.Series([10.0, 20.0]))

        result = model.predict_with_uncertainty(X)
        assert np.all(result["predictions"] == 20.0)
        assert np.all(result["std"] == 0.0)
        assert np.all(result["upper_bound"] == result["predictions"])
        assert np.all(result["lower_bound"] == result["predictions"])

    def test_save_load_noop(self):
        """save/load should succeed without doing real I/O."""
        model = NaiveModel()
        assert model.save_model("/tmp/naive_test") is True
        assert model.load_model("/tmp/naive_test") is True
        assert model.is_fitted  # load sets is_fitted

    def test_get_model_info(self):
        model = NaiveModel(pred_len=10)
        info = model.get_model_info()
        assert info["model_name"] == "NaiveModel"
        assert info["pred_len"] == 10
        assert "description" in info

    def test_registered_in_factory(self):
        """NaiveModel should be available via ModelFactory."""
        available = ModelFactory.get_available_models()
        assert "naive" in available
        assert available["naive"]["available"] is True
        assert available["naive"]["class"] is NaiveModel

    def test_create_via_factory(self):
        """ModelFactory.create_model('naive') should return NaiveModel instance."""
        model = ModelFactory.create_model("naive")
        assert isinstance(model, NaiveModel)
