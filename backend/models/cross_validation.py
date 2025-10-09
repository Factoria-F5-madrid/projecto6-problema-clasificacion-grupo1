"""
Cross-validation system for ML model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from ..config.model_config import ModelConfig

class CrossValidator:
    """
    Cross-validation system for evaluating ML models
    Controls overfitting and provides comprehensive metrics
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42, 
                 stratified: bool = True, shuffle: bool = True):
        """
        Initialize cross-validator
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            stratified: Whether to use stratified CV (recommended for classification)
            shuffle: Whether to shuffle data before splitting
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.stratified = stratified
        self.shuffle = shuffle
        
        # Initialize CV strategy
        if stratified:
            self.cv_strategy = StratifiedKFold(
                n_splits=cv_folds, 
                shuffle=shuffle, 
                random_state=random_state
            )
        else:
            self.cv_strategy = KFold(
                n_splits=cv_folds, 
                shuffle=shuffle, 
                random_state=random_state
            )
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a model using cross-validation
        
        Args:
            model: Trained ML model
            X: Feature matrix
            y: Target vector
            metrics: List of metrics to evaluate (default: all available)
            
        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = ModelConfig.EVALUATION_METRICS
        
        results = {}
        
        # Calculate cross-validation scores for each metric
        for metric in metrics:
            try:
                scores = cross_val_score(
                    model, X, y, 
                    cv=self.cv_strategy, 
                    scoring=metric,
                    n_jobs=-1
                )
                results[f'{metric}_mean'] = np.mean(scores)
                results[f'{metric}_std'] = np.std(scores)
                results[f'{metric}_scores'] = scores
            except Exception as e:
                print(f"Warning: Could not calculate {metric}: {e}")
                results[f'{metric}_mean'] = np.nan
                results[f'{metric}_std'] = np.nan
                results[f'{metric}_scores'] = []
        
        return results
    
    def check_overfitting(self, train_scores: np.ndarray, 
                         val_scores: np.ndarray, 
                         threshold: float = 0.05) -> Dict[str, Any]:
        """
        Check for overfitting by comparing train and validation scores
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            threshold: Maximum acceptable difference (default: 5%)
            
        Returns:
            Dictionary with overfitting analysis
        """
        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        difference = abs(train_mean - val_mean)
        
        is_overfitting = difference > threshold
        
        return {
            'train_mean': train_mean,
            'val_mean': val_mean,
            'difference': difference,
            'threshold': threshold,
            'is_overfitting': is_overfitting,
            'overfitting_percentage': (difference / train_mean) * 100
        }
    
    def detailed_evaluation(self, model, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed evaluation with train/validation split
        
        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Comprehensive evaluation results
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        # Check for overfitting
        overfitting_analysis = self.check_overfitting(
            np.array([train_metrics['accuracy']]),
            np.array([val_metrics['accuracy']])
        )
        
        return {
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'overfitting_analysis': overfitting_analysis,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        try:
            # For binary classification, use average='binary'
            # For multiclass, use average='macro'
            average = 'binary' if len(np.unique(y_true)) == 2 else 'macro'
            
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
                'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
                'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
            }
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation
        
        Args:
            models: Dictionary of model_name -> model_instance
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            cv_results = self.evaluate_model(model, X, y)
            
            result_row = {'model': model_name}
            for metric in ModelConfig.EVALUATION_METRICS:
                if f'{metric}_mean' in cv_results:
                    result_row[f'{metric}_mean'] = cv_results[f'{metric}_mean']
                    result_row[f'{metric}_std'] = cv_results[f'{metric}_std']
            
            results.append(result_row)
        
        return pd.DataFrame(results)
