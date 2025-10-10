"""
Create a model specifically trained to handle offensive words better
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

def create_offensive_words_model():
    """Create a model specifically trained to handle offensive words better"""
    
    print("🚀 CREATING OFFENSIVE WORDS MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Prepare data
    X_text = df['clean_tweet_improved'].fillna('')
    y = df['class']
    
    print(f"Dataset: {len(df)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Create vectorizer with NO stop words for offensive words
    print("\n1. Creating vectorizer with NO stop words...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,  # More features
        stop_words=None,  # NO stop words - keep everything
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
    
    # Create model with different class weights
    print("\n2. Creating model with custom class weights...")
    
    # Calculate class weights to balance the dataset
    class_counts = y.value_counts()
    total = len(y)
    class_weights = {
        0: total / (3 * class_counts[0]),  # Hate Speech - more weight
        1: total / (3 * class_counts[1]),  # Offensive Language - less weight
        2: total / (3 * class_counts[2])   # Neither - more weight
    }
    
    print(f"Class weights: {class_weights}")
    
    model = LogisticRegression(
        penalty='l2', 
        C=0.1,  # Less regularization
        random_state=42, 
        max_iter=2000,
        class_weight=class_weights  # Custom class weights
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
        "bitch",
        "you are an asshole",
        "fuck you bitch",
        "you are a fucking moron",
        "you are a piece of shit"
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
    
    # Get feature importance for each class
    print("\n6. Feature importance by class:")
    print("-" * 60)
    
    feature_names = vectorizer.get_feature_names_out()
    
    for class_idx, class_name in class_names.items():
        print(f"\n{class_name} (Class {class_idx}):")
        coefficients = model.coef_[class_idx]
        
        # Get top 10 most important features
        top_indices = np.argsort(coefficients)[-10:][::-1]
        
        for i, idx in enumerate(top_indices):
            feature = feature_names[idx]
            coef = coefficients[idx]
            print(f"  {i+1:2d}. {feature:<20} (coef: {coef:.3f})")
    
    # Test with more examples
    print("\n7. Testing with more examples...")
    
    more_examples = [
        "you are an asshole",
        "fuck you bitch",
        "you are a fucking moron",
        "you are a piece of shit",
        "you are stupid",
        "you are dumb",
        "you are an idiot",
        "you are a loser",
        "you are a failure",
        "you are worthless"
    ]
    
    print("\nMore Examples:")
    print("-" * 60)
    
    for example in more_examples:
        X_test_word = vectorizer.transform([example])
        prediction = model.predict(X_test_word)[0]
        probabilities = model.predict_proba(X_test_word)[0]
        
        print(f"Text: '{example}'")
        print(f"Prediction: {class_names[prediction]}")
        print(f"Probabilities: {probabilities}")
        print(f"Confidence: {probabilities[prediction]:.3f}")
        print("-" * 40)
    
    return model, vectorizer, overfitting, test_score

if __name__ == "__main__":
    create_offensive_words_model()
