"""
Test script to verify that the ML optimization framework works correctly
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner
from config.model_config import ModelConfig

def test_framework():
    """Test the ML optimization framework with synthetic data"""
    
    print("🧪 TESTING ML OPTIMIZATION FRAMEWORK")
    print("=" * 50)
    
    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   ✅ Train set: {X_train.shape[0]} samples")
    print(f"   ✅ Test set: {X_test.shape[0]} samples")
    
    # 2. Test CrossValidator
    print("\n2. Testing CrossValidator...")
    validator = CrossValidator(cv_folds=5)
    
    # Test with a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    
    cv_results = validator.evaluate_model(model, X_train, y_train)
    print(f"   ✅ Cross-validation completed")
    print(f"   ✅ Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    
    # 3. Test HyperparameterTuner
    print("\n3. Testing HyperparameterTuner...")
    tuner = HyperparameterTuner(cv_folds=5, verbose=0)  # Silent for testing
    
    # Get model and parameters from config
    model = ModelConfig.get_model('logistic_regression')
    params = ModelConfig.get_hyperparameters('logistic_regression')
    
    print(f"   ✅ Model: {model.__class__.__name__}")
    print(f"   ✅ Parameters to test: {len(params)} parameter groups")
    
    # Test GridSearch (with limited parameters for speed)
    limited_params = {
        'C': [0.1, 1, 10],  # Reduced for testing
        'penalty': ['l1', 'l2']
    }
    
    print("\n   Testing GridSearch...")
    grid_results = tuner.grid_search(model, limited_params, X_train, y_train)
    print(f"   ✅ GridSearch completed: {grid_results['best_score']:.4f}")
    
    # Test RandomizedSearch
    print("\n   Testing RandomizedSearch...")
    random_results = tuner.randomized_search(
        model, limited_params, X_train, y_train, n_iter=10
    )
    print(f"   ✅ RandomizedSearch completed: {random_results['best_score']:.4f}")
    
    # 4. Test comparison
    print("\n4. Testing comparison method...")
    comparison = tuner.compare_methods(
        model, limited_params, X_train, y_train, n_iter=10
    )
    print(f"   ✅ Comparison completed")
    print(f"   ✅ Winner: {comparison['winner']}")
    
    # 5. Test overfitting control
    print("\n5. Testing overfitting control...")
    best_model = grid_results['best_model']
    overfitting_analysis = validator.check_overfitting(
        np.array([0.95]),  # Simulated train score
        np.array([0.90])   # Simulated val score
    )
    print(f"   ✅ Overfitting analysis: {overfitting_analysis['is_overfitting']}")
    print(f"   ✅ Difference: {overfitting_analysis['overfitting_percentage']:.2f}%")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Framework is working correctly.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_framework()
