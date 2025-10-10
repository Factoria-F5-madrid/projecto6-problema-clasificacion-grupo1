"""
Script to improve model metrics while keeping overfitting < 5%
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Import our framework
from models.cross_validation import CrossValidator
from models.hyperparameter_tuning import HyperparameterTuner

def improve_metrics():
    """Improve model metrics while keeping overfitting < 5%"""
    
    print("🚀 IMPROVING MODEL METRICS")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Use larger sample for better results
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    
    X_text = df_sample['clean_tweet_improved'].fillna('')
    y = df_sample['class']
    
    # Enhanced vectorization
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1,3),  # Include trigrams
        min_df=2,  # Remove rare words
        max_df=0.95  # Remove very common words
    )
    X = vectorizer.fit_transform(X_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   ✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 2. Test enhanced models
    print("\n2. Testing enhanced models...")
    
    enhanced_models = {
        'LogisticRegression_L1_Enhanced': LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=0.01,  # More regularization
            random_state=42, 
            max_iter=2000,
            class_weight='balanced'  # Handle class imbalance
        ),
        'LogisticRegression_L2_Enhanced': LogisticRegression(
            penalty='l2', 
            C=0.01,  # More regularization
            random_state=42, 
            max_iter=2000,
            class_weight='balanced'
        ),
        'RandomForest_Enhanced': RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting_Enhanced': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,  # Lower learning rate
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'NaiveBayes_Enhanced': MultinomialNB(
            alpha=0.1  # Less smoothing
        )
    }
    
    validator = CrossValidator(cv_folds=5)
    
    results = []
    
    for name, model in enhanced_models.items():
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
            
            # Get detailed metrics
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Calculate weighted averages
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
            
            print(f"      CV Score: {cv_score:.3f}")
            print(f"      Train: {train_score:.3f}, Test: {test_score:.3f}")
            print(f"      Overfitting: {overfitting_pct:.2f}%")
            print(f"      Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Store results
            results.append({
                'name': name,
                'model': model,
                'cv_score': cv_score,
                'train_score': train_score,
                'test_score': test_score,
                'overfitting': overfitting_pct,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            # Check if meets requirements
            if overfitting_pct < 5.0 and test_score > 0.8:  # Higher threshold
                print(f"      ✅ EXCELLENT! (<5% overfitting, >80% accuracy)")
            elif overfitting_pct < 5.0 and test_score > 0.75:
                print(f"      ✅ GOOD! (<5% overfitting, >75% accuracy)")
            else:
                print(f"      ⚠️  Needs improvement")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue
    
    # 3. Find the best model
    print("\n3. Finding the best enhanced model...")
    
    # Filter models that meet requirements
    valid_models = [r for r in results if r['overfitting'] < 5.0 and r['test_score'] > 0.75]
    
    if valid_models:
        # Sort by F1 score (better balance of precision and recall)
        valid_models.sort(key=lambda x: x['f1'], reverse=True)
        best_result = valid_models[0]
        
        print(f"\n🏆 BEST ENHANCED MODEL: {best_result['name']}")
        print(f"   ✅ Test Score: {best_result['test_score']:.3f}")
        print(f"   ✅ Overfitting: {best_result['overfitting']:.2f}%")
        print(f"   ✅ Precision: {best_result['precision']:.3f}")
        print(f"   ✅ Recall: {best_result['recall']:.3f}")
        print(f"   ✅ F1-Score: {best_result['f1']:.3f}")
        
        # 4. Final evaluation
        print(f"\n4. Final evaluation of best enhanced model...")
        
        best_model = best_result['model']
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print(f"\n   Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # ROC AUC (if possible)
        try:
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                print(f"\n   ROC AUC: {roc_auc:.3f}")
        except:
            print(f"\n   ROC AUC: Not available for this model")
        
        print(f"\n🎉 SUCCESS! Enhanced model with {best_result['overfitting']:.2f}% overfitting")
        return best_model, best_result
        
    else:
        print(f"\n❌ No enhanced model meets the requirements")
        print(f"   Best overfitting: {min([r['overfitting'] for r in results]):.2f}%")
        return None, None

if __name__ == "__main__":
    improve_metrics()
