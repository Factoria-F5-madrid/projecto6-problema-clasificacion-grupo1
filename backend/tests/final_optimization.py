"""
Final optimization using the best models found
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner

def final_optimization():
    """Final optimization using best models"""
    
    print("🎯 FINAL OPTIMIZATION - USING BEST MODELS")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Use larger sample for better results
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)
    
    X_text = df_sample['clean_tweet_improved'].fillna('')
    y = df_sample['class']
    
    # Vectorize with more features
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(X_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   ✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 2. Test the best models with proper parameters
    print("\n2. Testing best models with optimized parameters...")
    
    # Best models from previous test
    best_models = {
        'LogisticRegression_L1': LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=0.1, 
            random_state=42, 
            max_iter=1000
        ),
        'LogisticRegression_L2': LogisticRegression(
            penalty='l2', 
            C=0.1, 
            random_state=42, 
            max_iter=1000
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            random_state=42
        ),
        'SVM': SVC(
            C=0.1, 
            kernel='linear', 
            random_state=42
        ),
        'NaiveBayes': MultinomialNB(
            alpha=1.0
        )
    }
    
    validator = CrossValidator(cv_folds=5)
    
    results = []
    
    for name, model in best_models.items():
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
            
            # Store results
            results.append({
                'name': name,
                'model': model,
                'cv_score': cv_score,
                'train_score': train_score,
                'test_score': test_score,
                'overfitting': overfitting_pct
            })
            
            # Check if meets requirements
            if overfitting_pct < 5.0 and test_score > 0.7:
                print(f"      ✅ MEETS REQUIREMENTS! (<5% overfitting, >70% accuracy)")
            else:
                print(f"      ⚠️  Does not meet requirements")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue
    
    # 3. Find the best model
    print("\n3. Finding the best model...")
    
    # Filter models that meet requirements
    valid_models = [r for r in results if r['overfitting'] < 5.0 and r['test_score'] > 0.7]
    
    if valid_models:
        # Sort by test score (descending)
        valid_models.sort(key=lambda x: x['test_score'], reverse=True)
        best_result = valid_models[0]
        
        print(f"\n🏆 BEST MODEL: {best_result['name']}")
        print(f"   ✅ Test Score: {best_result['test_score']:.3f}")
        print(f"   ✅ Overfitting: {best_result['overfitting']:.2f}%")
        print(f"   ✅ CV Score: {best_result['cv_score']:.3f}")
        
        # 4. Final evaluation
        print(f"\n4. Final evaluation of best model...")
        
        best_model = best_result['model']
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n   Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance (if available)
        if hasattr(best_model, 'coef_'):
            print(f"\n   Feature Importance (top 10):")
            feature_names = vectorizer.get_feature_names_out()
            coef = best_model.coef_[0] if len(best_model.coef_.shape) > 1 else best_model.coef_
            top_features = np.argsort(np.abs(coef))[-10:][::-1]
            for i, idx in enumerate(top_features):
                print(f"      {i+1:2d}. {feature_names[idx]:20s} {coef[idx]:8.4f}")
        
        print(f"\n🎉 SUCCESS! Found model with {best_result['overfitting']:.2f}% overfitting (<5%)")
        return best_model, best_result
        
    else:
        print(f"\n❌ No model meets the requirements (<5% overfitting, >70% accuracy)")
        print(f"   Best overfitting: {min([r['overfitting'] for r in results]):.2f}%")
        return None, None

if __name__ == "__main__":
    final_optimization()
