"""
Currency Prediction Main Script

This script demonstrates how to use the currency_predictor package
to collect data, process it, train models, and make predictions.
"""

import logging
from src.currency_predictor import (
    CurrencyDataCollector, 
    DataProcessor, 
    CurrencyPredictor,
    setup_logging,
    ensure_directories,
    get_default_config
)


def main():
    """Main function to run currency prediction pipeline."""
    
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Currency Prediction Pipeline")
    
    # Ensure required directories exist
    ensure_directories(['data', 'models', 'results'])
    
    try:
        # Load configuration
        config = get_default_config()
        
        # Step 1: Collect Data
        logger.info("Step 1: Collecting currency data...")
        collector = CurrencyDataCollector()
        
        # Get data for EUR/USD pair as example
        currency_pair = "EURUSD=X"
        data = collector.get_yahoo_finance_data(currency_pair, period="1y")
        
        if data.empty:
            logger.error("No data collected. Exiting.")
            return
        
        logger.info(f"Collected {len(data)} records for {currency_pair}")
        
        # Save raw data
        collector.save_data(data, f"data/{currency_pair}_raw.csv")
        
        # Step 2: Process Data
        logger.info("Step 2: Processing data...")
        processor = DataProcessor()
        
        # Clean data
        clean_data = processor.clean_data(data)
        
        # Create technical indicators
        processed_data = processor.create_technical_indicators(clean_data)
        
        # Create lagged features
        final_data = processor.create_lagged_features(processed_data)
        
        # Prepare features and target
        X, y = processor.prepare_features_target(final_data)
        
        if len(X) == 0:
            logger.error("No features prepared. Check data processing steps.")
            return
        
        # Train/test split
        X_train, X_test, y_train, y_test = processor.train_test_split(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
        
        # Step 3: Train Models
        logger.info("Step 3: Training prediction models...")
        predictor = CurrencyPredictor()
        predictor.setup_default_models()
        
        # Train all models
        predictor.train_all_models(X_train_scaled, y_train)
        
        # Step 4: Evaluate Models
        logger.info("Step 4: Evaluating models...")
        results = predictor.evaluate_all_models(X_test_scaled, y_test)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*50)
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE:  {metrics['mae']:.4f}")
            logger.info(f"  R²:   {metrics['r2']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        # Find best model
        best_model = predictor.get_best_model(results, metric='rmse')
        logger.info(f"\nBest performing model: {best_model}")
        
        # Show feature importance for best model
        if best_model in predictor.feature_importance:
            importance = predictor.get_feature_importance(best_model, top_n=5)
            logger.info(f"\nTop 5 features for {best_model}:")
            for feature, score in importance.items():
                logger.info(f"  {feature}: {score:.4f}")
        
        # Save best model
        model_path = f"models/{best_model}_{currency_pair}.pkl"
        predictor.save_model(best_model, model_path)
        
        logger.info("\nCurrency prediction pipeline completed successfully!")
        logger.info(f"Best model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
