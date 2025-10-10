"""
Script to optimize overfitting and reduce it below 5%
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner
from config.model_config import ModelConfig

def optimize_overfitting():
    """Optimize overfitting to get it below 5%"""
    
    print("🎯 OPTIMIZING OVERFITTING - TARGET: <5%")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Use larger sample for better results
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    
    X_text = df_sample['clean_tweet_improved'].fillna('')
    y = df_sample['class']
    
    # Vectorize with more features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(X_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   ✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 2. Test different models and regularization
    print("\n2. Testing different models and regularization...")
    
    models_to_test = {
        'LogisticRegression_L1': LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42, max_iter=1000),
        'LogisticRegression_L2': LogisticRegression(penalty='l2', C=0.1, random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
        'SVM': SVC(C=0.1, kernel='linear', random_state=42),
        'NaiveBayes': MultinomialNB(alpha=1.0)
    }
    
    validator = CrossValidator(cv_folds=5)
    tuner = HyperparameterTuner(cv_folds=5, verbose=0)
    
    best_model = None
    best_score = 0
    best_overfitting = float('inf')
    
    for name, model in models_to_test.items():
        print(f"\n   Testing {name}...")
        
        try:
            # Cross-validation
            cv_results = validator.evaluate_model(model, X_train, y_train)
            cv_score = cv_results['accuracy_mean']
            
            # Train and test
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Check overfitting
            overfitting_analysis = validator.check_overfitting(
                np.array([train_score]), np.array([test_score])
            )
            overfitting_pct = overfitting_analysis['overfitting_percentage']
            
            print(f"      CV Score: {cv_score:.3f}")
            print(f"      Train: {train_score:.3f}, Test: {test_score:.3f}")
            print(f"      Overfitting: {overfitting_pct:.2f}%")
            
            # Check if this is the best model (lowest overfitting with good score)
            if overfitting_pct < best_overfitting and test_score > 0.7:
                best_model = model
                best_score = test_score
                best_overfitting = overfitting_pct
                print(f"      ✅ NEW BEST MODEL!")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue
    
    # 3. Optimize the best model with hyperparameter tuning
    if best_model is not None:
        print(f"\n3. Optimizing best model (overfitting: {best_overfitting:.2f}%)...")
        
        # Get model name and create parameter grid
        model_name = type(best_model).__name__
        
        if model_name == 'LogisticRegression':
            params = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'max_iter': [1000, 2000]
            }
        elif model_name == 'RandomForestClassifier':
            params = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'GradientBoostingClassifier':
            params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            params = {'C': [0.1, 1, 10]}
        
        # Create new instance for tuning
        if model_name == 'LogisticRegression':
            model_for_tuning = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'RandomForestClassifier':
            model_for_tuning = RandomForestClassifier(random_state=42)
        elif model_name == 'GradientBoostingClassifier':
            model_for_tuning = GradientBoostingClassifier(random_state=42)
        else:
            model_for_tuning = best_model
        
        # Hyperparameter tuning
        grid_results = tuner.grid_search(model_for_tuning, params, X_train, y_train)
        
        # Test optimized model
        optimized_model = grid_results['best_model']
        optimized_model.fit(X_train, y_train)
        
        train_score_opt = optimized_model.score(X_train, y_train)
        test_score_opt = optimized_model.score(X_test, y_test)
        
        overfitting_opt = validator.check_overfitting(
            np.array([train_score_opt]), np.array([test_score_opt])
        )
        
        print(f"   ✅ Optimized model:")
        print(f"      Best params: {grid_results['best_params']}")
        print(f"      Train: {train_score_opt:.3f}, Test: {test_score_opt:.3f}")
        print(f"      Overfitting: {overfitting_opt['overfitting_percentage']:.2f}%")
        
        # Final result
        if overfitting_opt['overfitting_percentage'] < 5.0:
            print(f"\n🎉 SUCCESS! Overfitting reduced to {overfitting_opt['overfitting_percentage']:.2f}% (<5%)")
        else:
            print(f"\n⚠️  Overfitting still {overfitting_opt['overfitting_percentage']:.2f}% (>5%)")
            print("   Consider: More regularization, more data, or different model")
    
    else:
        print("\n❌ No suitable model found")
    
    print("\n" + "=" * 60)
    return best_model, best_overfitting

if __name__ == "__main__":
    optimize_overfitting()
