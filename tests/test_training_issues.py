"""
Test Training Issues

This test suite reproduces and validates fixes for the issues found in basic_usage.py:
1. PatchTST parameter mismatch (validation_split)
2. Data processing creating zero training records
3. Full pipeline integration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from currency_predictor.models.patchtst import PatchTST
from currency_predictor.data_processor import DataProcessor
from currency_predictor.prediction.predictor import CurrencyPredictor
from currency_predictor.config.manager import ConfigManager
from currency_predictor.prediction import PredictionPipeline


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_currency_data():
    """Create realistic currency data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=260, freq='D')

    np.random.seed(42)
    base_price = 30.0
    close_prices = base_price + np.cumsum(np.random.randn(260) * 0.1)

    data = {
        'Open': close_prices + np.random.randn(260) * 0.05,
        'High': close_prices + np.abs(np.random.randn(260) * 0.1),
        'Low': close_prices - np.abs(np.random.randn(260) * 0.1),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 260)
    }

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def small_training_data():
    """Create minimal training data that PatchTST can use"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    data = {
        'feature_1': np.random.randn(200),
        'feature_2': np.random.randn(200),
        'feature_3': np.random.randn(200),
    }

    X = pd.DataFrame(data, index=dates)
    y = pd.Series(np.random.randn(200), index=dates, name='target')

    return X, y


# ============================================================================
# Problem 1: PatchTST Parameter Mismatch Tests
# ============================================================================

class TestPatchTSTParameterHandling:
    """Test PatchTST model parameter handling"""

    def test_patchtst_fit_with_valid_parameters(self, small_training_data):
        """Test that PatchTST.fit() works with correct parameters"""
        X, y = small_training_data
        model = PatchTST(seq_len=20, pred_len=5, patch_len=5, stride=3)

        # Should not raise error
        model.fit(X, y)
        assert model.is_fitted

    def test_patchtst_fit_ignores_validation_split(self, small_training_data):
        """Test that PatchTST.fit() accepts but ignores validation_split via **kwargs"""
        X, y = small_training_data
        model = PatchTST(seq_len=20, pred_len=5, patch_len=5, stride=3)

        # PatchTSTSklearn.fit accepts **kwargs, so validation_split is silently ignored
        model.fit(X, y, validation_split=0.2)
        assert model.is_fitted

    def test_patchtst_fit_accepts_validation_data(self, small_training_data):
        """Test that PatchTST.fit() accepts validation_data parameter"""
        X, y = small_training_data
        X_val, y_val = X.iloc[:20], y.iloc[:20]

        model = PatchTST(seq_len=20, pred_len=5, patch_len=5, stride=3)

        # Should not raise error with validation_data
        model.fit(X, y, validation_data=(X_val, y_val))
        assert model.is_fitted

    def test_patchtst_signature_documentation(self):
        """Document PatchTST.fit() expected signature"""
        model = PatchTST()

        # Check method signature
        import inspect
        sig = inspect.signature(model.fit)

        # Should have these parameters
        params = list(sig.parameters.keys())
        assert 'X' in params
        assert 'y' in params
        assert 'validation_data' in params

        # Should NOT have validation_split
        assert 'validation_split' not in params


# ============================================================================
# Problem 2: Data Processing Zero Records Tests
# ============================================================================

class TestDataProcessorNaNHandling:
    """Test data processor NaN creation and handling"""

    def test_technical_indicators_preserve_data(self, sample_currency_data):
        """Test that technical indicators don't remove all records"""
        processor = DataProcessor()

        initial_count = len(sample_currency_data)
        df = processor.create_technical_indicators(sample_currency_data)

        # Should still have data (though some NaN is expected)
        assert len(df) == initial_count, "Technical indicators shouldn't drop rows"

        # Count NaN values
        nan_count = df.isnull().sum().sum()
        print(f"\nNaN after technical indicators: {nan_count}")

    def test_lagged_features_data_loss(self, sample_currency_data):
        """Test how many records are lost to lagged features"""
        processor = DataProcessor()

        # First create technical indicators
        df = processor.create_technical_indicators(sample_currency_data)

        initial_count = len(df)
        print(f"\nRecords before lagged features: {initial_count}")

        # Create lagged features (this is where the problem occurs)
        df_lagged = processor.create_lagged_features(df)

        final_count = len(df_lagged)
        print(f"Records after lagged features: {final_count}")
        print(f"Records lost: {initial_count - final_count}")

        # Lagged feature creation must retain rows (not drop everything).
        assert final_count > 0, "All records were dropped by lagged features!"

    def test_full_feature_engineering_pipeline(self, sample_currency_data):
        """Test complete feature engineering pipeline"""
        processor = DataProcessor()

        print(f"\nInitial records: {len(sample_currency_data)}")

        # Step 1: Clean
        df = processor.clean_data(sample_currency_data)
        print(f"After clean_data: {len(df)}")

        # Step 2: Technical indicators
        df = processor.create_technical_indicators(df)
        print(f"After technical indicators: {len(df)}")
        print(f"  NaN count: {df.isnull().sum().sum()}")

        # Step 3: Lagged features
        df = processor.create_lagged_features(df)
        print(f"After lagged features: {len(df)}")

        # Step 4: Prepare features/target
        X, y = processor.prepare_features_target(df)
        print(f"Final training samples: {len(X)}")

        # After fixes, this should pass
        assert len(X) > 0, "Feature engineering removed all training data!"
        assert len(X) > 50, f"Too much data lost: {len(sample_currency_data)} → {len(X)}"

    def test_dropna_vs_fillna_comparison(self, sample_currency_data):
        """Compare dropna() vs fillna() approaches"""
        processor = DataProcessor()

        # Create features
        df1 = processor.create_technical_indicators(sample_currency_data.copy())

        # Count NaN before drop/fill
        nan_count_before = df1.isnull().sum().sum()

        # Approach 1: dropna (current approach)
        df1_dropped = df1.dropna()

        # Approach 2: fillna
        df2_filled = df1.fillna(method='ffill').fillna(method='bfill')

        print(f"\nNaN count: {nan_count_before}")
        print(f"dropna() result: {len(df1_dropped)} records")
        print(f"fillna() result: {len(df2_filled)} records")

        # fillna should preserve more data
        assert len(df2_filled) >= len(df1_dropped)

    def test_reduced_feature_windows(self, sample_currency_data):
        """Test that smaller rolling windows preserve more data"""
        df = sample_currency_data.copy()

        # Large window (current implementation)
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        nan_count_large = df['MA_50'].isnull().sum()

        # Smaller window (proposed fix)
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        nan_count_small = df['MA_20'].isnull().sum()

        print(f"\nNaN with MA_50: {nan_count_large}")
        print(f"NaN with MA_20: {nan_count_small}")

        # Smaller window creates fewer NaN
        assert nan_count_small < nan_count_large


# ============================================================================
# Problem 3: CurrencyPredictor Integration Tests
# ============================================================================

class TestCurrencyPredictorTraining:
    """Test CurrencyPredictor with realistic data"""

    def test_predictor_train_with_validation_split(self, tmp_path):
        """Test that predictor handles validation_split correctly.

        CurrencyPredictor.train_model now internally converts validation_split
        to validation_data, so it should not raise an error.
        """
        # Create test data directory with raw subdirectory
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True)

        dates = pd.date_range(start='2024-01-01', periods=260, freq='D')
        df = pd.DataFrame({
            'Open': 30 + np.random.randn(260) * 0.1,
            'High': 30 + np.random.randn(260) * 0.1,
            'Low': 30 + np.random.randn(260) * 0.1,
            'Close': 30 + np.random.randn(260) * 0.1,
            'Volume': np.random.randint(1000000, 10000000, 260)
        }, index=dates)

        # Save with correct naming for DataStorage.load_raw_data
        data_file = raw_dir / "TEST_1y_20240101_000000.csv"
        df.to_csv(data_file)

        # Create predictor with correct kwarg name
        predictor = CurrencyPredictor(
            model_name="patchtst_sklearn",
            model_params={
                'seq_len': 20,
                'pred_len': 5,
                'patch_len': 5,
                'stride': 3
            },
            data_storage_path=str(data_dir)
        )

        # CurrencyPredictor now handles validation_split internally
        result = predictor.train_model(
            symbol="TEST",
            period="1y",
            validation_split=0.2
        )

        assert isinstance(result, dict)
        assert 'training_completed' in result


# ============================================================================
# Problem 4: Full Pipeline Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Test full prediction pipeline"""

    @pytest.mark.slow
    def test_basic_usage_pipeline_reproduction(self, tmp_path):
        """Reproduce the exact scenario from basic_usage.py"""

        # Setup
        config_manager = ConfigManager()
        config = config_manager.get_config()

        # Override with test settings
        config['data_dir'] = str(tmp_path / "data")
        config['models_dir'] = str(tmp_path / "models")
        config['results_dir'] = str(tmp_path / "results")

        # Create necessary directories
        for dir_path in [config['data_dir'], config['models_dir'], config['results_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=260, freq='D')
        df = pd.DataFrame({
            'Open': 30 + np.cumsum(np.random.randn(260) * 0.1),
            'High': 30 + np.cumsum(np.random.randn(260) * 0.1),
            'Low': 30 + np.cumsum(np.random.randn(260) * 0.1),
            'Close': 30 + np.cumsum(np.random.randn(260) * 0.1),
            'Volume': np.random.randint(1000000, 10000000, 260)
        }, index=dates)

        data_file = Path(config['data_dir']) / "USDTWD=X.csv"
        df.to_csv(data_file)

        # Create pipeline
        pipeline = PredictionPipeline(config, output_dir=config['results_dir'])

        # The pipeline now works end-to-end (see test_e2e_pipeline_output).
        # save_results=True so the results_saved stage runs; otherwise
        # success = all(pipeline_status.values()) can never be True.
        results = pipeline.run_full_pipeline(
            symbols=["USDTWD=X"],
            prediction_horizon=7,
            save_results=True,
            force_retrain=True,
        )

        assert results is not None
        assert results["success"] is True


# ============================================================================
# Utility Tests
# ============================================================================

class TestDataQualityChecks:
    """Tests to ensure data quality throughout pipeline"""

    def test_nan_detection_in_features(self, sample_currency_data):
        """Test that we can detect NaN in processed features"""
        processor = DataProcessor()

        df = processor.create_technical_indicators(sample_currency_data)

        # Check each column for NaN
        nan_per_column = df.isnull().sum()
        nan_columns = nan_per_column[nan_per_column > 0]

        print("\nColumns with NaN:")
        for col, count in nan_columns.items():
            print(f"  {col}: {count} NaN values")

        # Document which features create the most NaN
        if len(nan_columns) > 0:
            worst_column = nan_columns.idxmax()
            worst_count = nan_columns.max()
            print(f"\nWorst offender: {worst_column} with {worst_count} NaN")

    def test_minimum_data_requirements(self):
        """Test what minimum data is needed for PatchTST"""
        model = PatchTST(seq_len=20, pred_len=5, patch_len=5, stride=3)

        # Calculate minimum required records
        min_required = model.seq_len + model.pred_len

        print(f"\nPatchTST requirements:")
        print(f"  seq_len: {model.seq_len}")
        print(f"  pred_len: {model.pred_len}")
        print(f"  Minimum records needed: {min_required}")

        # After feature engineering, we need even more
        # MA_50 needs 50, lag_10 needs 10, etc.
        estimated_loss = 50 + 10  # MA_50 + max lag
        recommended_minimum = min_required + estimated_loss

        print(f"  Estimated feature engineering loss: {estimated_loss}")
        print(f"  Recommended minimum input: {recommended_minimum}")

        # 260 records - 60 lost = 200 remaining should be enough
        assert 260 > recommended_minimum, "Test data should have enough records"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
