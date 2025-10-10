"""
Create a hybrid model that combines ML with rule-based classification for offensive words
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

def create_rule_based_model():
    """Create a hybrid model that combines ML with rule-based classification"""
    
    print("🚀 CREATING RULE-BASED HYBRID MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
    
    # Prepare data
    X_text = df['clean_tweet_improved'].fillna('')
    y = df['class']
    
    print(f"Dataset: {len(df)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1,3),
        min_df=2,
        max_df=0.95,
        lowercase=True
    )
    
    X = vectorizer.fit_transform(X_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create ML model
    model = LogisticRegression(
        penalty='l2', 
        C=0.01,
        random_state=42, 
        max_iter=2000,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Define offensive words rules
    offensive_words = {
        'hate_speech': [
            'faggot', 'fag', 'faggots', 'fags', 'nigger', 'nigga', 'niggas', 'niggers',
            'dyke', 'dykes', 'tranny', 'trannies', 'faggy', 'faggoty', 'niggah',
            'white trash', 'cracker', 'crackers', 'chink', 'chinks', 'gook', 'gooks',
            'wetback', 'wetbacks', 'spic', 'spics', 'kike', 'kikes', 'towelhead',
            'towelheads', 'raghead', 'ragheads', 'sand nigger', 'sand niggers'
        ],
        'offensive_language': [
            'bitch', 'bitches', 'hoes', 'hoe', 'pussy', 'pussies', 'ass', 'asshole',
            'assholes', 'fuck', 'fucking', 'fucked', 'fucker', 'fuckers', 'shit',
            'shits', 'shitty', 'damn', 'damned', 'hell', 'crap', 'crapper', 'dumb',
            'dumbass', 'dumbasses', 'stupid', 'idiot', 'idiots', 'moron', 'morons',
            'loser', 'losers', 'failure', 'failures', 'worthless', 'pathetic',
            'disgusting', 'gross', 'nasty', 'ugly', 'fat', 'fats', 'skinny', 'skinnies'
        ]
    }
    
    def rule_based_classification(text):
        """Apply rule-based classification for offensive words"""
        text_lower = text.lower()
        
        # Check for hate speech words
        hate_speech_count = sum(1 for word in offensive_words['hate_speech'] if word in text_lower)
        offensive_count = sum(1 for word in offensive_words['offensive_language'] if word in text_lower)
        
        if hate_speech_count > 0:
            return 0, 0.9  # Hate Speech with high confidence
        elif offensive_count > 0:
            return 1, 0.8  # Offensive Language with high confidence
        else:
            return None, 0.0  # No rule-based classification
    
    def hybrid_predict(text, model, vectorizer):
        """Hybrid prediction combining rules and ML"""
        # First try rule-based classification
        rule_pred, rule_conf = rule_based_classification(text)
        
        if rule_pred is not None:
            # Use rule-based classification
            return rule_pred, rule_conf
        else:
            # Use ML model
            X_test = vectorizer.transform([text])
            ml_pred = model.predict(X_test)[0]
            ml_conf = model.predict_proba(X_test)[0][ml_pred]
            return ml_pred, ml_conf
    
    # Test the hybrid model
    print("\n1. Testing hybrid model...")
    
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
        "you are a piece of shit",
        "faggot",
        "nigger",
        "you are a faggot",
        "you are a nigger"
    ]
    
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    
    print("\nHybrid Model Results:")
    print("-" * 60)
    
    for word in test_words:
        prediction, confidence = hybrid_predict(word, model, vectorizer)
        
        print(f"Text: '{word}'")
        print(f"Prediction: {class_names[prediction]}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Method: {'Rule-based' if confidence > 0.7 else 'ML'}")
        print("-" * 40)
    
    # Test with more examples
    print("\n2. Testing with more examples...")
    
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
        "you are worthless",
        "you are a faggot",
        "you are a nigger",
        "you are a dyke",
        "you are a tranny"
    ]
    
    print("\nMore Examples:")
    print("-" * 60)
    
    for example in more_examples:
        prediction, confidence = hybrid_predict(example, model, vectorizer)
        
        print(f"Text: '{example}'")
        print(f"Prediction: {class_names[prediction]}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Method: {'Rule-based' if confidence > 0.7 else 'ML'}")
        print("-" * 40)
    
    # Evaluate the hybrid model
    print("\n3. Evaluating hybrid model...")
    
    # Test on a sample of the test set
    test_sample = X_test[:100]
    y_test_sample = y_test[:100]
    
    hybrid_predictions = []
    for i in range(len(test_sample)):
        text = X_text.iloc[i]
        pred, conf = hybrid_predict(text, model, vectorizer)
        hybrid_predictions.append(pred)
    
    # Calculate accuracy
    hybrid_accuracy = np.mean(hybrid_predictions == y_test_sample)
    print(f"Hybrid Model Accuracy: {hybrid_accuracy:.3f}")
    
    # Compare with pure ML
    ml_predictions = model.predict(test_sample)
    ml_accuracy = np.mean(ml_predictions == y_test_sample)
    print(f"Pure ML Accuracy: {ml_accuracy:.3f}")
    
    return model, vectorizer, hybrid_predict

if __name__ == "__main__":
    create_rule_based_model()
