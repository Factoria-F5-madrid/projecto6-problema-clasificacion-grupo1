"""
Hyperparameter tuning system for ML models
Compatible with any sklearn model
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator

class HyperparameterTuner:
    """
    Hyperparameter tuning system for ML models
    Compatible with any sklearn model
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42, 
                 n_jobs: int = -1, verbose: int = 1):
        """
        Initialize hyperparameter tuner
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs (-1 = use all cores)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def grid_search(self, model: BaseEstimator, param_grid: Dict[str, List], 
                   X: np.ndarray, y: np.ndarray, 
                   scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform grid search to find the best hyperparameters
        
        Args:
            model: The ML model to optimize (can be any sklearn model)
            param_grid: Dictionary with hyperparameters to test
            X: Training features
            y: Training targets
            scoring: Metric to optimize (default: accuracy)
            
        Returns:
            Dictionary with optimization results
        """
        # Create GridSearchCV object with our settings
        grid_search = GridSearchCV(
            estimator=model,           # The model to optimize
            param_grid=param_grid,     # Hyperparameters to test
            cv=self.cv_folds,         # Number of cross-validation folds
            scoring=scoring,          # Metric to optimize
            n_jobs=self.n_jobs,       # Use all CPU cores
            verbose=self.verbose      # Show progress
        )
        
        # Fit the grid search to find best parameters
        print(f"Starting grid search for {model.__class__.__name__}...")
        grid_search.fit(X, y)
        
        # Return results in a clear format
        results = {
            'best_model': grid_search.best_estimator_,      # The optimized model
            'best_params': grid_search.best_params_,        # Best hyperparameters found
            'best_score': grid_search.best_score_,          # Best cross-validation score
            'cv_results': grid_search.cv_results_,          # All results for analysis
            'n_trials': len(grid_search.cv_results_['params'])  # Number of combinations tested
        }
        
        print(f"Grid search completed! Best score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return results
    
    def randomized_search(self, model: BaseEstimator, param_distributions: Dict[str, List], 
                         X: np.ndarray, y: np.ndarray, n_iter: int = 100,
                         scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform randomized search to find the best hyperparameters (faster than grid search)
        
        Args:
            model: The ML model to optimize (can be any sklearn model)
            param_distributions: Dictionary with hyperparameters to test
            X: Training features
            y: Training targets
            n_iter: Number of random combinations to try (default: 100)
            scoring: Metric to optimize (default: accuracy)
            
        Returns:
            Dictionary with optimization results
        """
        # Create RandomizedSearchCV object with our settings
        random_search = RandomizedSearchCV(
            estimator=model,                    # The model to optimize
            param_distributions=param_distributions,  # Hyperparameters to test
            n_iter=n_iter,                     # Number of random combinations to try
            cv=self.cv_folds,                  # Number of cross-validation folds
            scoring=scoring,                   # Metric to optimize
            n_jobs=self.n_jobs,                # Use all CPU cores
            verbose=self.verbose,              # Show progress
            random_state=self.random_state     # For reproducibility
        )
        
        # Fit the random search to find best parameters
        print(f"Starting randomized search for {model.__class__.__name__}...")
        print(f"Testing {n_iter} random combinations...")
        random_search.fit(X, y)
        
        # Return results in a clear format
        results = {
            'best_model': random_search.best_estimator_,      # The optimized model
            'best_params': random_search.best_params_,        # Best hyperparameters found
            'best_score': random_search.best_score_,          # Best cross-validation score
            'cv_results': random_search.cv_results_,          # All results for analysis
            'n_trials': n_iter                                # Number of combinations tested
        }
        
        print(f"Randomized search completed! Best score: {random_search.best_score_:.4f}")
        print(f"Best parameters: {random_search.best_params_}")
        
        return results
    
    def compare_methods(self, model: BaseEstimator, param_grid: Dict[str, List], 
                       X: np.ndarray, y: np.ndarray, n_iter: int = 100,
                       scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Compare GridSearchCV vs RandomizedSearchCV on the same model
        
        Args:
            model: The ML model to optimize
            param_grid: Dictionary with hyperparameters to test
            X: Training features
            y: Training targets
            n_iter: Number of random combinations for RandomizedSearch (default: 100)
            scoring: Metric to optimize (default: accuracy)
            
        Returns:
            Dictionary with comparison results
        """
        print("=" * 60)
        print("COMPARING GRID SEARCH vs RANDOMIZED SEARCH")
        print("=" * 60)
        
        # Run Grid Search
        print("\n1. Running Grid Search...")
        grid_results = self.grid_search(model, param_grid, X, y, scoring)
        
        # Run Randomized Search
        print("\n2. Running Randomized Search...")
        random_results = self.randomized_search(model, param_grid, X, y, n_iter, scoring)
        
        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        # Create comparison summary
        comparison = {
            'grid_search': {
                'best_score': grid_results['best_score'],
                'best_params': grid_results['best_params'],
                'n_trials': grid_results['n_trials'],
                'method': 'GridSearchCV'
            },
            'randomized_search': {
                'best_score': random_results['best_score'],
                'best_params': random_results['best_params'],
                'n_trials': random_results['n_trials'],
                'method': 'RandomizedSearchCV'
            }
        }
        
        # Print comparison table
        print(f"\n{'Method':<20} {'Best Score':<12} {'Trials':<8} {'Winner':<10}")
        print("-" * 50)
        
        grid_score = grid_results['best_score']
        random_score = random_results['best_score']
        
        winner_grid = "✓" if grid_score >= random_score else ""
        winner_random = "✓" if random_score >= grid_score else ""
        
        print(f"{'GridSearchCV':<20} {grid_score:<12.4f} {grid_results['n_trials']:<8} {winner_grid:<10}")
        print(f"{'RandomizedSearchCV':<20} {random_score:<12.4f} {random_results['n_trials']:<8} {winner_random:<10}")
        
        # Determine winner
        if grid_score > random_score:
            print(f"\n🏆 GridSearchCV WINS! (Score: {grid_score:.4f})")
            best_method = 'grid_search'
        elif random_score > grid_score:
            print(f"\n🏆 RandomizedSearchCV WINS! (Score: {random_score:.4f})")
            best_method = 'randomized_search'
        else:
            print(f"\n🤝 TIE! Both methods achieved {grid_score:.4f}")
            best_method = 'tie'
        
        # Add winner info to results
        comparison['winner'] = best_method
        comparison['score_difference'] = abs(grid_score - random_score)
        
        print(f"\nScore difference: {comparison['score_difference']:.4f}")
        print("=" * 60)
        
        return comparison
