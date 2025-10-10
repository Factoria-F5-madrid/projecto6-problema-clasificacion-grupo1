"""
Test framework with real data - step by step
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner

def test_framework_step_by_step():
    """Test framework step by step with real data"""
    
    print("🧪 TESTING FRAMEWORK WITH REAL DATA - STEP BY STEP")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    print(f"   ✅ Loaded {df.shape[0]} samples")
    
    # Step 2: Prepare data (small sample for testing)
    print("\n2. Preparing data...")
    df_sample = df.sample(n=500, random_state=42)  # Small sample for testing
    
    X_text = df_sample['clean_tweet_improved'].fillna('')
    y = df_sample['class']  # 0: Hate Speech, 1: Offensive, 2: Neither
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(X_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Vectorized: {X.shape[1]} features")
    print(f"   ✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"   ✅ Classes: {np.unique(y)}")
    
    # Step 3: Test CrossValidator
    print("\n3. Testing CrossValidator...")
    validator = CrossValidator(cv_folds=3)
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    cv_results = validator.evaluate_model(model, X_train, y_train)
    print(f"   ✅ CV Accuracy: {cv_results['accuracy_mean']:.3f}")
    
    # Step 4: Test HyperparameterTuner
    print("\n4. Testing HyperparameterTuner...")
    tuner = HyperparameterTuner(cv_folds=3, verbose=0)
    
    params = {'C': [0.1, 1], 'penalty': ['l2']}
    grid_results = tuner.grid_search(model, params, X_train, y_train)
    print(f"   ✅ Best score: {grid_results['best_score']:.3f}")
    
    # Step 5: Test overfitting
    print("\n5. Testing overfitting control...")
    best_model = grid_results['best_model']
    best_model.fit(X_train, y_train)
    
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    overfitting = validator.check_overfitting(
        np.array([train_score]), np.array([test_score])
    )
    
    print(f"   ✅ Train: {train_score:.3f}, Test: {test_score:.3f}")
    print(f"   ✅ Overfitting: {overfitting['is_overfitting']}")
    print(f"   ✅ Difference: {overfitting['overfitting_percentage']:.2f}%")
    
    print("\n" + "=" * 60)
    print("🎉 FRAMEWORK WORKS WITH REAL DATA!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_framework_step_by_step()
