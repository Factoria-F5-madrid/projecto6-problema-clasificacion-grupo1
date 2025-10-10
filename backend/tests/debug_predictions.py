"""
Script to debug why certain words are not being classified correctly
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def debug_predictions():
    """Debug why certain words are not being classified correctly"""
    
    print("🔍 DEBUGGING PREDICTIONS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Prepare data
    X_text = df['clean_tweet_improved'].fillna('')
    y = df['class']
    
    # Create vectorizer with same parameters as our optimized model
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1,3),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(X_text)
    
    # Create optimized model
    model = LogisticRegression(
        penalty='l2', 
        C=0.01,
        random_state=42, 
        max_iter=2000,
        class_weight='balanced'
    )
    
    # Train the model
    model.fit(X, y)
    
    # Test words
    test_words = [
        "asshole",
        "fuck you",
        "you are stupid",
        "hate you",
        "kill yourself",
        "you are beautiful",
        "I love you",
        "good morning",
        "fucking idiot",
        "bitch"
    ]
    
    print("\n1. Testing individual words/phrases:")
    print("-" * 60)
    
    for word in test_words:
        # Transform text
        X_test = vectorizer.transform([word])
        
        # Make prediction
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        
        # Get class names
        class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        
        print(f"Text: '{word}'")
        print(f"Prediction: {class_names[prediction]}")
        print(f"Probabilities: {probabilities}")
        print(f"Confidence: {probabilities[prediction]:.3f}")
        print("-" * 40)
    
    # 2. Check what features are being used
    print("\n2. Analyzing feature importance:")
    print("-" * 60)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients for each class
    coefficients = model.coef_
    
    # Find most important features for each class
    for class_idx, class_name in class_names.items():
        print(f"\n{class_name} (Class {class_idx}):")
        class_coef = coefficients[class_idx]
        
        # Get top 10 most important features
        top_indices = np.argsort(class_coef)[-10:][::-1]
        
        for i, idx in enumerate(top_indices):
            feature = feature_names[idx]
            coef = class_coef[idx]
            print(f"  {i+1:2d}. {feature:<20} (coef: {coef:.3f})")
    
    # 3. Check if our test words are in the vocabulary
    print("\n3. Checking vocabulary:")
    print("-" * 60)
    
    for word in test_words:
        # Check if word is in vocabulary
        if word in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[word]
            print(f"'{word}' is in vocabulary at index {idx}")
        else:
            print(f"'{word}' is NOT in vocabulary")
            
            # Check if parts of the word are in vocabulary
            words = word.split()
            for w in words:
                if w in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[w]
                    print(f"  - '{w}' is in vocabulary at index {idx}")
                else:
                    print(f"  - '{w}' is NOT in vocabulary")
    
    # 4. Check the training data distribution
    print("\n4. Training data distribution:")
    print("-" * 60)
    
    class_counts = df['class'].value_counts()
    print("Class distribution:")
    for class_idx, count in class_counts.items():
        print(f"  {class_names[class_idx]}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # 5. Check some examples from each class
    print("\n5. Sample examples from each class:")
    print("-" * 60)
    
    for class_idx, class_name in class_names.items():
        print(f"\n{class_name} examples:")
        class_samples = df[df['class'] == class_idx]['clean_tweet_improved'].head(3)
        for i, sample in enumerate(class_samples, 1):
            print(f"  {i}. {sample}")
    
    # 6. Test with more explicit examples
    print("\n6. Testing with more explicit examples:")
    print("-" * 60)
    
    explicit_examples = [
        "you are an asshole",
        "fuck you bitch",
        "I hate you so much",
        "you are fucking stupid",
        "kill yourself you idiot",
        "you are a fucking moron",
        "I want to kill you",
        "you are a piece of shit"
    ]
    
    for example in explicit_examples:
        X_test = vectorizer.transform([example])
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        
        print(f"Text: '{example}'")
        print(f"Prediction: {class_names[prediction]}")
        print(f"Probabilities: {probabilities}")
        print(f"Confidence: {probabilities[prediction]:.3f}")
        print("-" * 40)

if __name__ == "__main__":
    debug_predictions()
