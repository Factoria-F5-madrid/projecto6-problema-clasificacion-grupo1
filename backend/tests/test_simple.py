"""
Simple test script to verify that the ML optimization framework works
"""

import sys
import os
sys.path.append('backend')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner
from config.model_config import ModelConfig

def test_simple():
    """Simple test of the framework"""
    
    print("🧪 TESTING ML FRAMEWORK - SIMPLE TEST")
    print("=" * 50)
    
    # 1. Create small dataset
    print("\n1. Creating test dataset...")
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"   ✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Test CrossValidator
    print("\n2. Testing CrossValidator...")
    validator = CrossValidator(cv_folds=3)  # Small CV for testing
    model = LogisticRegression(random_state=42)
    
    results = validator.evaluate_model(model, X_train, y_train)
    print(f"   ✅ Cross-validation: {results['accuracy_mean']:.3f}")
    
    # 3. Test HyperparameterTuner
    print("\n3. Testing HyperparameterTuner...")
    tuner = HyperparameterTuner(cv_folds=3, verbose=0)
    
    # Simple parameters for testing (compatible combinations)
    params = {'C': [0.1, 1], 'penalty': ['l2']}  # Only l2 to avoid solver conflicts
    
    grid_results = tuner.grid_search(model, params, X_train, y_train)
    print(f"   ✅ GridSearch: {grid_results['best_score']:.3f}")
    
    # 4. Test overfitting check
    print("\n4. Testing overfitting control...")
    overfitting = validator.check_overfitting(
        np.array([0.95]), np.array([0.90])
    )
    print(f"   ✅ Overfitting check: {overfitting['is_overfitting']}")
    
    print("\n" + "=" * 50)
    print("🎉 FRAMEWORK IS WORKING!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_simple()
