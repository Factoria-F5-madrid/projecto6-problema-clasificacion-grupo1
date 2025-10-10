"""
Test script to verify that the ML optimization framework works with real data
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner
from config.model_config import ModelConfig

# Import baseline models
from models.baseline_models import load_and_clean_data, train_baseline_models

def test_with_real_data():
    """Test the framework with real tweet data"""
    
    print("🧪 TESTING ML FRAMEWORK WITH REAL DATA")
    print("=" * 60)
    
    # 1. Load real data
    print("\n1. Loading real tweet data...")
    try:
        df = load_and_clean_data('backend/data/processed/cleaned_tweets.csv')
        print(f"   ✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"   ✅ Columns: {list(df.columns)}")
        print(f"   ✅ Target distribution:")
        print(df['class_label'].value_counts())
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return False
    
    # 2. Prepare data for ML
    print("\n2. Preparing data for ML...")
    try:
        # Use a subset for testing (faster)
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        # Prepare features and target
        X_text = df_sample['clean_tweet_improved'].fillna('')
        y = df_sample['class']  # 0: Hate Speech, 1: Offensive, 2: Neither
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(X_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   ✅ Text vectorized: {X.shape[1]} features")
        print(f"   ✅ Train set: {X_train.shape[0]} samples")
        print(f"   ✅ Test set: {X_test.shape[0]} samples")
        print(f"   ✅ Target classes: {np.unique(y)}")
        
    except Exception as e:
        print(f"   ❌ Error preparing data: {e}")
        return False
    
    # 3. Test CrossValidator
    print("\n3. Testing CrossValidator with real data...")
    try:
        validator = CrossValidator(cv_folds=3)  # Small CV for testing
        
        # Test with Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_results = validator.evaluate_model(model, X_train, y_train)
        print(f"   ✅ Cross-validation completed")
        print(f"   ✅ Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error in CrossValidator: {e}")
        return False
    
    # 4. Test HyperparameterTuner
    print("\n4. Testing HyperparameterTuner with real data...")
    try:
        tuner = HyperparameterTuner(cv_folds=3, verbose=0)
        
        # Simple parameters for testing
        params = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'max_iter': [100, 500]
        }
        
        grid_results = tuner.grid_search(model, params, X_train, y_train)
        print(f"   ✅ GridSearch completed")
        print(f"   ✅ Best score: {grid_results['best_score']:.3f}")
        print(f"   ✅ Best params: {grid_results['best_params']}")
        
    except Exception as e:
        print(f"   ❌ Error in HyperparameterTuner: {e}")
        return False
    
    # 5. Test overfitting control
    print("\n5. Testing overfitting control...")
    try:
        # Train the best model
        best_model = grid_results['best_model']
        best_model.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate scores
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        # Check overfitting
        overfitting_analysis = validator.check_overfitting(
            np.array([train_score]), np.array([test_score])
        )
        
        print(f"   ✅ Train score: {train_score:.3f}")
        print(f"   ✅ Test score: {test_score:.3f}")
        print(f"   ✅ Overfitting: {overfitting_analysis['is_overfitting']}")
        print(f"   ✅ Difference: {overfitting_analysis['overfitting_percentage']:.2f}%")
        
    except Exception as e:
        print(f"   ❌ Error in overfitting check: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 FRAMEWORK WORKS WITH REAL DATA!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_with_real_data()
