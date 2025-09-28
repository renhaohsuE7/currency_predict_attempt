"""
Currency Prediction Main Script

This script demonstrates how to use the currency_predictor package
to collect data, process it, train models, and make predictions.
"""

import logging
import json
from pathlib import Path

from src.currency_predictor.prediction import CurrencyPredictor, PredictionPipeline
from src.currency_predictor.utils import setup_logging


def main():
    """Main function to run currency prediction pipeline."""
    
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Currency Prediction Pipeline")
    
    # Ensure required directories exist
    for dir_path in ['data', 'models', 'results']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config()
        
        # Step 1: Create and run prediction pipeline
        logger.info("Step 1: Creating prediction pipeline...")
        pipeline = PredictionPipeline(config, output_dir="results")
        
        # Currency pairs to predict
        symbols = config.get('symbols', ['USDTWD=X', 'EURUSD=X'])
        prediction_horizon = config.get('prediction_horizon', 7)
        
        logger.info(f"Predicting {len(symbols)} currency pairs: {symbols}")
        
        # Run complete pipeline
        results = pipeline.run_full_pipeline(
            symbols=symbols,
            prediction_horizon=prediction_horizon,
            save_results=True,
            force_retrain=False
        )
        
        # Print results summary
        logger.info(f"Pipeline execution completed, success: {results.get('success', False)}")
        logger.info(f"Processed currency pairs: {len(symbols)}")
        
        # Display stage status
        status = results.get('pipeline_status', {})
        logger.info(f"Data collection: {'✅' if status.get('data_collection') else '❌'}")
        logger.info(f"Model training: {'✅' if status.get('model_training') else '❌'}")
        logger.info(f"Prediction execution: {'✅' if status.get('prediction') else '❌'}")
        logger.info(f"Results saved: {'✅' if status.get('results_saved', False) else '❌'}")
        
        # Display prediction results
        predictions = results.get('predictions', [])
        for prediction in predictions:
            symbol = prediction['symbol']
            if not prediction.get('error'):
                last_value = prediction.get('last_known_value', 0)
                first_pred = prediction.get('predictions', [0])[0] if prediction.get('predictions') else 0
                change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
                logger.info(f"{symbol}: Predicted change {change:+.2f}%")
            else:
                logger.error(f"{symbol}: Prediction failed - {prediction.get('error', '')}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("EXECUTION SUMMARY")
        logger.info("="*50)
        
        overall_success = results.get('success', False)
        logger.info(f"Overall execution: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        
        if overall_success:
            logger.info("\nCheck the following directories for results:")
            logger.info("- 'results/' : Prediction results and reports")
            logger.info("- 'models/'  : Trained models") 
            logger.info("- 'data/'    : Collected currency data")
        
        return 0 if overall_success else 1
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


def load_config():
    """Load configuration from file or return default config."""
    config_file = Path("config.json")
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "model_name": "PatchTST",
            "model_params": {
                "seq_len": 168,
                "pred_len": 24,
                "patch_len": 12,
                "stride": 6,
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "data_storage_path": "data",
            "log_level": "INFO",
            "data_collection": {
                "period": "1y",
                "interval": "1d",
                "force_update": False
            },
            "model_training": {
                "period": "1y",
                "target_column": "Close",
                "feature_columns": None,
                "train_params": {
                    "validation_split": 0.2
                }
            },
            "prediction": {
                "period": "1y",
                "return_uncertainty": True
            },
            "symbols": [
                "USDTWD=X",
                "EURUSD=X",
                "GBPUSD=X"
            ],
            "prediction_horizon": 7
        }
    
    return config


if __name__ == "__main__":
    """Entry point of the application."""
    exit_code = main()
    exit(exit_code)
