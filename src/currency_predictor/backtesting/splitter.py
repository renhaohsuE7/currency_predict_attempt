"""Walk-forward splitter for backtesting."""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class FoldSpec:
    """Specification for a single walk-forward fold."""

    fold_index: int
    train_start: int  # iloc index (inclusive)
    train_end: int  # iloc index (exclusive)
    test_start: int  # iloc index (inclusive)
    test_end: int  # iloc index (exclusive)

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class WalkForwardSplitter:
    """Generate train/test splits for walk-forward validation.

    Supports two strategies:
    - "rolling": fixed-size training window that slides forward
    - "expanding": training window grows with each fold

    Example (rolling, initial_train=4, test_size=2, step_size=2):
        Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Fold 0: train=[0:4], test=[4:6]
        Fold 1: train=[2:6], test=[6:8]
        Fold 2: train=[4:8], test=[8:10]

    Example (expanding, initial_train=4, test_size=2, step_size=2):
        Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Fold 0: train=[0:4], test=[4:6]
        Fold 1: train=[0:6], test=[6:8]
        Fold 2: train=[0:8], test=[8:10]
    """

    def __init__(
        self,
        strategy: Literal["rolling", "expanding"] = "rolling",
        initial_train_size: int = 252,
        test_size: int = 30,
        step_size: int = 30,
    ):
        if initial_train_size < 1:
            raise ValueError(f"initial_train_size must be >= 1, got {initial_train_size}")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if strategy not in ("rolling", "expanding"):
            raise ValueError(f"strategy must be 'rolling' or 'expanding', got '{strategy}'")

        self.strategy = strategy
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, n_samples: int) -> List[FoldSpec]:
        """Generate fold specifications for n_samples data points.

        Args:
            n_samples: Total number of data points.

        Returns:
            List of FoldSpec defining train/test index ranges.

        Raises:
            ValueError: If data is too short for even one fold.
        """
        min_required = self.initial_train_size + self.test_size
        if n_samples < min_required:
            raise ValueError(
                f"Need at least {min_required} samples "
                f"(initial_train={self.initial_train_size} + test={self.test_size}), "
                f"got {n_samples}"
            )

        folds: List[FoldSpec] = []
        fold_idx = 0
        test_start = self.initial_train_size

        while test_start + self.test_size <= n_samples:
            test_end = test_start + self.test_size

            if self.strategy == "rolling":
                train_start = test_start - self.initial_train_size
            else:  # expanding
                train_start = 0

            folds.append(
                FoldSpec(
                    fold_index=fold_idx,
                    train_start=train_start,
                    train_end=test_start,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            fold_idx += 1
            test_start += self.step_size

        return folds

    def min_samples_required(self, min_folds: int = 1) -> int:
        """Calculate minimum data points required for a given number of folds.

        Args:
            min_folds: Minimum number of folds desired.

        Returns:
            Minimum number of data points needed.
        """
        if min_folds < 1:
            raise ValueError(f"min_folds must be >= 1, got {min_folds}")

        return self.initial_train_size + self.test_size + (min_folds - 1) * self.step_size
