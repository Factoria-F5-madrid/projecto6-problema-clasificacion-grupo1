"""
Create an improved model that better handles offensive language
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def create_improved_model():
    """Create an improved model that better handles offensive language"""
    
    print("🚀 CREATING IMPROVED MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Prepare data
    X_text = df['clean_tweet_improved'].fillna('')
    y = df['class']
    
    print(f"Dataset: {len(df)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Create improved vectorizer
    print("\n1. Creating improved vectorizer...")
    
    # Custom stop words - keep important words
    custom_stop_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 
        'our', 'ours', 'ourselves', 'it', 'its', 'itself', 'they', 'them', 
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
        'whose', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just'
    ]
    
    vectorizer = TfidfVectorizer(
        max_features=3000,  # More features
        stop_words=custom_stop_words,  # Custom stop words
        ngram_range=(1,3),  # Keep trigrams
        min_df=1,  # Lower threshold
        max_df=0.95,
        lowercase=True,
        strip_accents='unicode'
    )
    
    X = vectorizer.fit_transform(X_text)
    print(f"Features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create improved model
    print("\n2. Creating improved model...")
    
    model = LogisticRegression(
        penalty='l2', 
        C=0.1,  # Less regularization
        random_state=42, 
        max_iter=2000,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n3. Evaluating model...")
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfitting = (train_score - test_score) * 100
    
    print(f"Train Score: {train_score:.3f}")
    print(f"Test Score: {test_score:.3f}")
    print(f"Overfitting: {overfitting:.2f}%")
    
    # Test predictions
    print("\n4. Testing predictions...")
    
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
    
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    
    print("\nTest Results:")
    print("-" * 60)
    
    for word in test_words:
        X_test_word = vectorizer.transform([word])
        prediction = model.predict(X_test_word)[0]
        probabilities = model.predict_proba(X_test_word)[0]
        
        print(f"Text: '{word}'")
        print(f"Prediction: {class_names[prediction]}")
        print(f"Probabilities: {probabilities}")
        print(f"Confidence: {probabilities[prediction]:.3f}")
        print("-" * 40)
    
    # Check vocabulary
    print("\n5. Checking vocabulary...")
    
    for word in test_words:
        if word in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[word]
            print(f"'{word}' is in vocabulary at index {idx}")
        else:
            print(f"'{word}' is NOT in vocabulary")
            # Check parts
            words = word.split()
            for w in words:
                if w in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[w]
                    print(f"  - '{w}' is in vocabulary at index {idx}")
    
    # Get feature importance
    print("\n6. Feature importance for Offensive Language:")
    print("-" * 60)
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[1]  # Offensive Language class
    
    # Get top 20 most important features for Offensive Language
    top_indices = np.argsort(coefficients)[-20:][::-1]
    
    for i, idx in enumerate(top_indices):
        feature = feature_names[idx]
        coef = coefficients[idx]
        print(f"{i+1:2d}. {feature:<20} (coef: {coef:.3f})")
    
    return model, vectorizer, overfitting, test_score

if __name__ == "__main__":
    create_improved_model()
