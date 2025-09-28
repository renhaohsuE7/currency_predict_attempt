"""Machine Learning Models Module

This module contains various machine learning models for currency prediction
including linear regression, random forest, and neural networks.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, Optional
import logging
import pickle

logger = logging.getLogger(__name__)


class CurrencyPredictor:
    """Main class for currency prediction using various ML models."""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.feature_importance = {}
        
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to the predictor.
        
        Args:
            name: Name identifier for the model
            model: Sklearn model instance
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def setup_default_models(self) -> None:
        """Setup default models for currency prediction."""
        
        # Linear models
        self.add_model('linear_regression', LinearRegression())
        self.add_model('ridge', Ridge(alpha=1.0))
        self.add_model('lasso', Lasso(alpha=0.1))
        
        # Tree-based models
        self.add_model('random_forest', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
        
        self.add_model('gradient_boosting', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ))
        
        logger.info(f"Setup {len(self.models)} default models")
    
    def train_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> None:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = dict(zip(
                X_train.columns, 
                model.feature_importances_
            ))
        elif hasattr(model, 'coef_'):
            self.feature_importance[model_name] = dict(zip(
                X_train.columns, 
                abs(model.coef_)
            ))
        
        logger.info(f"Model {model_name} trained successfully")
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> None:
        """Train all available models."""
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
    
    def predict(
        self, 
        model_name: str, 
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        
        logger.info(f"Generated {len(predictions)} predictions using {model_name}")
        return predictions
    
    def evaluate_model(
        self, 
        model_name: str, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a trained model's performance.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: True test values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(model_name, X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        metrics['mape'] = mape
        
        logger.info(f"{model_name} evaluation - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        return metrics
    
    def evaluate_all_models(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        results = {}
        
        for model_name in self.trained_models.keys():
            try:
                results[model_name] = self.evaluate_model(model_name, X_test, y_test)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        return results
    
    def get_best_model(
        self, 
        evaluation_results: Dict[str, Dict[str, float]], 
        metric: str = 'rmse'
    ) -> str:
        """
        Find the best performing model based on a metric.
        
        Args:
            evaluation_results: Results from evaluate_all_models
            metric: Metric to use for comparison ('rmse', 'mae', 'r2', 'mape')
            
        Returns:
            Name of the best model
        """
        if metric in ['rmse', 'mae', 'mse', 'mape']:
            # Lower is better
            best_model = min(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x][metric])
        else:
            # Higher is better (r2)
            best_model = max(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x][metric])
        
        logger.info(f"Best model by {metric}: {best_model}")
        return best_model
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features and their importance
        """
        if model_name not in self.feature_importance:
            return {}
        
        importance = self.feature_importance[model_name]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.trained_models:
            logger.error(f"Model {model_name} not trained yet")
            return False
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.trained_models[model_name], f)
            logger.info(f"Model {model_name} saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False
    
    def load_model(self, model_name: str, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            self.trained_models[model_name] = model
            logger.info(f"Model loaded as {model_name} from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            return False