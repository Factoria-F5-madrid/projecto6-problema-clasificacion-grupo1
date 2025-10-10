"""
Baseline Models for Tweet Classification

This module contains all the core functionality for training and evaluating
baseline models for hate speech detection in tweets.

"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV and perform basic data cleaning.
    Args:
        file_path (str): Path to the CSV file containing the tweet data
    Returns:
        pd.DataFrame: Cleaned dataset ready for processing
    """
    print("Loading datos...")
    df = pd.read_csv(file_path)
    
    print("Cleaning datos and null values...")
    df_clean = df.copy()
    
    # CHeck for null values before cleaning
    nulls_before = df_clean['clean_tweet_improved'].isnull().sum()
    print(f"\t- Valores nulos encontrados: {nulls_before:,}")
    
    # Delete rows with null values in the text column
    df_clean = df_clean.dropna(subset=['clean_tweet_improved'])
    
    # Convert to a string and remove empty texts
    df_clean['clean_tweet_improved'] = df_clean['clean_tweet_improved'].astype(str)
    df_clean = df_clean[df_clean['clean_tweet_improved'].str.strip() != '']
    df_clean = df_clean[df_clean['clean_tweet_improved'] != 'nan']
    
    nulls_after = len(df) - len(df_clean)
    print(f"\t-Filas eliminadas: {nulls_after:,}")
    print(f"\t- Dataset final: {len(df_clean):,} filas")
    
    return df_clean


def prepare_features(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Prepare features using TF-IDF vectorization and split data.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        test_size (float): Proportion of dataset for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple: (X_train_vec, X_test_vec, y_train, y_test, tfidf_vectorizer)
    """
    print("Preparando características TF-IDF...")
    
    # Extraer features y target
    X = df['clean_tweet_improved']
    y = df['class']
    
    # Train/Test Split estratificado
    print(f"Splitting data ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Vectorización TF-IDF
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)    
    return X_train_vec, X_test_vec, y_train, y_test, tfidf


def get_baseline_models(class_weights: Dict) -> Dict:
    """
    Create and configure baseline models.
    Args:
        class_weights (Dict): Class weights for handling imbalanced data
    Returns:
        Dict: Dictionary containing configured models
    """
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight=class_weights,
            max_iter=1000,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(
            alpha=1.0
        ),
        'SVM Linear': SVC(
            kernel='linear',
            class_weight=class_weights,
            probability=True,
            random_state=42
        )
    }
    return models


def train_model(model, X_train, y_train) -> Tuple[Any, float]:
    """
    Train a single model and measure training time.
    Args:
        model: Sklearn model to train
        X_train: Training features
        y_train: Training labels
    Returns:
        Tuple: (trained_model, training_time)
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


def evaluate_model(model, X_test, y_test, class_names: list) -> Dict:
    """
    Evaluate a trained model and return comprehensive metrics.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        class_names (list): Names of the classes for reporting
        
    Returns:
        Dict: Dictionary containing evaluation metrics
    """
    # Predicciones
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report como diccionario
    class_report = classification_report(
        y_test, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'classification_report': class_report
    }


def train_and_evaluate_models(df: pd.DataFrame, class_weights: Dict) -> Tuple[Dict, Tuple]:
    """
    Main function that orchestrates the complete training and evaluation pipeline.
    Args:
        df (pd.DataFrame): Input dataset
        class_weights (Dict): Class weights for handling imbalanced data 
    Returns:
        Tuple: (results_dict, test_data_tuple)
    """
    
    print("STARTING AUTOMATED MODELING PIPELINE")
    print("\n")
    
    # 1. Preparing features
    X_train_vec, X_test_vec, y_train, y_test, tfidf = prepare_features(df)

    # 2. Getting baseline models
    models = get_baseline_models(class_weights)

    # 3. Training and evaluating each model
    results = {}
    class_names = ['Hate Speech', 'Offensive Language', 'Neither']
    
    for model_name, model in models.items():
        print(f"\n")
        print(f"Training: {model_name.upper()}")
        print("\n")
        
        # TRAINING
        trained_model, training_time = train_model(model, X_train_vec, y_train)

        # Evaluating model
        eval_results = evaluate_model(trained_model, X_test_vec, y_test, class_names)

        # Combining results                         
        results[model_name] = {
            'model': trained_model,
            'training_time': training_time,
            **eval_results
        }
        
        # Printing results
        accuracy = eval_results['accuracy']
        print(f"ccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Training time: {training_time:.2f} segundos")
        print(f"\nClassification Report:")
        print(classification_report(y_test, eval_results['predictions'], target_names=class_names))

    # 4. Final comparison
    print(f"\n")
    print("FINAL MODEL COMPARISON")
    print("\n")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Modelo': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Accuracy %': f"{result['accuracy']*100:.2f}%",
            'Tiempo (s)': f"{result['training_time']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Identifying best model...
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']

    print(f"\nBEST MODEL: {best_model_name}")
    print(f"\tAccuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return results, (X_test_vec, y_test, tfidf)


def save_results(results: Dict, output_path: str = None) -> None:
    """
    Save model results to disk.

    Args:
        results (Dict): Results dictionary from train_and_evaluate_models
        output_path (str): Path where to save the results
    """
    if output_path is None:
        output_path = "baseline_results.json"
    
    # Prepare results for serialization (remove non-serializable objects)
    serializable_results = {}
    for model_name, result in results.items():
        serializable_results[model_name] = {
            'accuracy': result['accuracy'],
            'training_time': result['training_time'],
            'classification_report': result['classification_report']
        }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Resultados guardados en: {output_path}")


def compute_class_weights(df: pd.DataFrame) -> Dict:
    """
    Compute class weights for handling imbalanced dataset.
    
    Args:
        df (pd.DataFrame): Dataset with 'class' column
        
    Returns:
        Dict: Class weights dictionary
    """
    classes = np.unique(df['class'])
    class_weights_array = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=df['class']
    )
    class_weights = dict(enumerate(class_weights_array))
    return class_weights


def run_baseline_experiment(data_path: str, save_results_path: str = None) -> Tuple[Dict, Tuple]:
    """
    Complete baseline experiment pipeline.
    
    Args:
        data_path (str): Path to the input CSV file
        save_results_path (str): Path to save results (optional)
    Returns:
        Tuple: (results_dict, test_data_tuple)
    """
    
    # 1. Load and clean data
    df = load_and_clean_data(data_path)
    
    # 2. Calculating class weights...
    class_weights = compute_class_weights(df)

    # 3. Running complete pipeline
    results, test_data = train_and_evaluate_models(df, class_weights)

    # 4. Save results if specified
    if save_results_path:
        save_results(results, save_results_path)
    
    return results, test_data


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/cleaned_tweets.csv"
    results, test_data = run_baseline_experiment(data_path)
    print("\nExperiment completed!")