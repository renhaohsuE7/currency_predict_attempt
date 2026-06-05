"""Tests for prediction.metrics (MASE and MDA)."""

import numpy as np
import pytest

from currency_predictor.prediction.metrics import mase, mda


# ---------------------------------------------------------------
# MASE
# ---------------------------------------------------------------

class TestMASE:
    """Tests for Mean Absolute Scaled Error."""

    def test_perfect_prediction(self):
        """MASE = 0 when predictions exactly match actuals."""
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0, 8.0])
        y_pred = np.array([6.0, 7.0, 8.0])
        assert mase(y_true, y_pred, y_train) == pytest.approx(0.0)

    def test_naive_equals_one(self):
        """MASE ~ 1.0 when model error equals naive error on training set."""
        # Training set with constant step of 1.0 → naive MAE = 1.0
        y_train = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        # Test set where model error = 1.0 per step (same as naive)
        y_true = np.array([15.0, 16.0, 17.0])
        y_pred = np.array([16.0, 17.0, 18.0])  # off by 1.0 each
        result = mase(y_true, y_pred, y_train)
        assert result == pytest.approx(1.0)

    def test_worse_than_naive(self):
        """MASE > 1.0 when model is worse than naive."""
        y_train = np.array([10.0, 11.0, 12.0, 13.0, 14.0])  # naive MAE = 1.0
        y_true = np.array([15.0, 16.0, 17.0])
        y_pred = np.array([20.0, 21.0, 22.0])  # off by 5.0 each → MASE = 5.0
        result = mase(y_true, y_pred, y_train)
        assert result == pytest.approx(5.0)

    def test_better_than_naive(self):
        """MASE < 1.0 when model is better than naive."""
        y_train = np.array([10.0, 12.0, 14.0, 16.0])  # naive MAE = 2.0
        y_true = np.array([18.0, 20.0])
        y_pred = np.array([18.5, 20.5])  # off by 0.5 → MASE = 0.5/2.0 = 0.25
        result = mase(y_true, y_pred, y_train)
        assert result == pytest.approx(0.25)

    def test_constant_train_returns_zero(self):
        """MASE = 0.0 when training set is constant (naive MAE = 0)."""
        y_train = np.array([5.0, 5.0, 5.0, 5.0])
        y_true = np.array([5.0, 6.0])
        y_pred = np.array([5.0, 5.5])
        assert mase(y_true, y_pred, y_train) == 0.0

    def test_single_train_point_returns_zero(self):
        """MASE = 0.0 when training set has only one point."""
        y_train = np.array([5.0])
        y_true = np.array([6.0])
        y_pred = np.array([7.0])
        assert mase(y_true, y_pred, y_train) == 0.0


# ---------------------------------------------------------------
# MDA
# ---------------------------------------------------------------

class TestMDA:
    """Tests for Mean Directional Accuracy."""

    def test_perfect_direction(self):
        """MDA = 1.0 when all directions match."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])  # all up
        y_pred = np.array([1.0, 1.5, 2.0, 2.5])  # all up
        assert mda(y_true, y_pred) == pytest.approx(1.0)

    def test_opposite_direction(self):
        """MDA = 0.0 when all directions are wrong."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])  # all up
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # all down
        assert mda(y_true, y_pred) == pytest.approx(0.0)

    def test_mixed_directions(self):
        """MDA reflects partial correctness."""
        y_true = np.array([1.0, 2.0, 1.0, 2.0])  # up, down, up
        y_pred = np.array([1.0, 2.0, 3.0, 2.0])  # up, up, down
        # Directions: true=[+, -, +], pred=[+, +, -] → 1 correct out of 3
        assert mda(y_true, y_pred) == pytest.approx(1.0 / 3.0)

    def test_empty_input(self):
        """MDA = 0.0 for empty arrays."""
        assert mda(np.array([]), np.array([])) == 0.0

    def test_single_value(self):
        """MDA = 0.0 for single-element arrays (no direction to compute)."""
        assert mda(np.array([1.0]), np.array([2.0])) == 0.0

    def test_two_values(self):
        """MDA with exactly 2 values → 1 direction to check."""
        assert mda(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(1.0)
        assert mda(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == pytest.approx(0.0)
