"""Tests for WalkForwardSplitter."""

import pytest

from currency_predictor.backtesting.splitter import WalkForwardSplitter, FoldSpec


class TestWalkForwardSplitterRolling:
    """Test rolling window strategy."""

    def test_basic_rolling_split(self):
        """Simple rolling split with known values."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=4, test_size=2, step_size=2
        )
        folds = splitter.split(10)

        assert len(folds) == 3
        assert folds[0] == FoldSpec(0, train_start=0, train_end=4, test_start=4, test_end=6)
        assert folds[1] == FoldSpec(1, train_start=2, train_end=6, test_start=6, test_end=8)
        assert folds[2] == FoldSpec(2, train_start=4, train_end=8, test_start=8, test_end=10)

    def test_rolling_train_size_constant(self):
        """Rolling strategy keeps train size constant."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=100, test_size=20, step_size=20
        )
        folds = splitter.split(300)
        for fold in folds:
            assert fold.train_size == 100

    def test_rolling_no_overlap_in_test(self):
        """Test windows should not overlap when step_size == test_size."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=50, test_size=10, step_size=10
        )
        folds = splitter.split(100)
        for i in range(len(folds) - 1):
            assert folds[i].test_end <= folds[i + 1].test_start

    def test_rolling_step_smaller_than_test(self):
        """When step < test, test windows overlap — that's allowed."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=4, test_size=3, step_size=1
        )
        folds = splitter.split(10)
        assert len(folds) >= 3

    def test_rolling_single_fold(self):
        """Exact minimum data for one fold."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=5, test_size=3, step_size=3
        )
        folds = splitter.split(8)
        assert len(folds) == 1
        assert folds[0].train_start == 0
        assert folds[0].train_end == 5
        assert folds[0].test_start == 5
        assert folds[0].test_end == 8


class TestWalkForwardSplitterExpanding:
    """Test expanding window strategy."""

    def test_basic_expanding_split(self):
        """Simple expanding split with known values."""
        splitter = WalkForwardSplitter(
            strategy="expanding", initial_train_size=4, test_size=2, step_size=2
        )
        folds = splitter.split(10)

        assert len(folds) == 3
        assert folds[0] == FoldSpec(0, train_start=0, train_end=4, test_start=4, test_end=6)
        assert folds[1] == FoldSpec(1, train_start=0, train_end=6, test_start=6, test_end=8)
        assert folds[2] == FoldSpec(2, train_start=0, train_end=8, test_start=8, test_end=10)

    def test_expanding_train_grows(self):
        """Expanding strategy grows the training window."""
        splitter = WalkForwardSplitter(
            strategy="expanding", initial_train_size=50, test_size=10, step_size=10
        )
        folds = splitter.split(200)
        for i in range(len(folds) - 1):
            assert folds[i + 1].train_size > folds[i].train_size

    def test_expanding_always_starts_at_zero(self):
        """Expanding strategy always starts training from index 0."""
        splitter = WalkForwardSplitter(
            strategy="expanding", initial_train_size=30, test_size=10, step_size=10
        )
        folds = splitter.split(100)
        for fold in folds:
            assert fold.train_start == 0


class TestWalkForwardSplitterValidation:
    """Test validation and edge cases."""

    def test_insufficient_data_raises(self):
        """Should raise ValueError when data is too short."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=100, test_size=30, step_size=30
        )
        with pytest.raises(ValueError, match="Need at least 130 samples"):
            splitter.split(50)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            WalkForwardSplitter(strategy="invalid")  # type: ignore[arg-type]

    def test_invalid_train_size(self):
        with pytest.raises(ValueError, match="initial_train_size"):
            WalkForwardSplitter(initial_train_size=0)

    def test_invalid_test_size(self):
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardSplitter(test_size=0)

    def test_invalid_step_size(self):
        with pytest.raises(ValueError, match="step_size"):
            WalkForwardSplitter(step_size=0)

    def test_no_data_leakage(self):
        """Train end must equal test start — no gap, no overlap."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=50, test_size=20, step_size=10
        )
        folds = splitter.split(200)
        for fold in folds:
            assert fold.train_end == fold.test_start

    def test_fold_spec_properties(self):
        """FoldSpec train_size and test_size properties."""
        fold = FoldSpec(fold_index=0, train_start=10, train_end=60, test_start=60, test_end=80)
        assert fold.train_size == 50
        assert fold.test_size == 20

    def test_min_samples_required(self):
        """min_samples_required() correctly calculates for given folds."""
        splitter = WalkForwardSplitter(
            strategy="rolling", initial_train_size=100, test_size=30, step_size=30
        )
        assert splitter.min_samples_required(1) == 130
        assert splitter.min_samples_required(4) == 220
