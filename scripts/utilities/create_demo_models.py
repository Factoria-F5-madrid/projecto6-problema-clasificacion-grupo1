#!/usr/bin/env python3
"""
Crear modelos de demostración con scores diferentes para mostrar reemplazo
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

def create_demo_models():
    """Crear modelos de demostración con diferentes rendimientos"""
    
    print("🚀 Creando modelos de demostración...")
    
    # Crear directorio si no existe
    os.makedirs('backend/models/saved', exist_ok=True)
    
    # Datos de entrenamiento simulados
    texts = [
        # Ofensivos
        "I hate you so much", "You are stupid", "Go die", "F*** you",
        "You're an idiot", "I hate this", "You suck", "Kill yourself",
        "You're worthless", "I hate everyone", "You're a moron", "Die already",
        "You're trash", "I hate this place", "You're dumb", "F*** off",
        
        # Neutros
        "Hello how are you", "This is nice", "Thank you", "Good morning",
        "I like this", "Have a great day", "This is good", "I'm happy",
        "Nice weather", "Good job", "I love this", "This is great",
        "Amazing work", "Perfect", "Excellent", "Wonderful"
    ]
    
    labels = [
        # Ofensivos (16)
        "offensive", "offensive", "offensive", "offensive",
        "offensive", "offensive", "offensive", "offensive",
        "offensive", "offensive", "offensive", "offensive",
        "offensive", "offensive", "offensive", "offensive",
        # Neutros (16)
        "neither", "neither", "neither", "neither",
        "neither", "neither", "neither", "neither",
        "neither", "neither", "neither", "neither",
        "neither", "neither", "neither", "neither"
    ]
    
    # Vectorizar
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts)
    y = labels
    
    # Modelo A: Random Forest (mejor rendimiento)
    print("📊 Entrenando Modelo A (Random Forest)...")
    model_a = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=2
    )
    model_a.fit(X, y)
    
    # Modelo B: Logistic Regression (rendimiento medio)
    print("📊 Entrenando Modelo B (Logistic Regression)...")
    model_b = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=0.1  # Regularización más fuerte = peor rendimiento
    )
    model_b.fit(X, y)
    
    # Guardar modelos
    print("💾 Guardando modelos...")
    joblib.dump(model_a, 'backend/models/saved/demo_model_a.pkl')
    joblib.dump(model_b, 'backend/models/saved/demo_model_b.pkl')
    joblib.dump(vectorizer, 'backend/models/saved/demo_vectorizer.pkl')
    
    # Evaluar modelos
    print("🔍 Evaluando modelos...")
    
    # Datos de prueba
    test_texts = [
        "I hate you", "You are amazing", "This sucks", "Great job",
        "You're stupid", "I love this", "F*** off", "Thank you"
    ]
    test_labels = [
        "offensive", "neither", "offensive", "neither",
        "offensive", "neither", "offensive", "neither"
    ]
    
    X_test = vectorizer.transform(test_texts)
    
    # Predicciones
    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, f1_score
    
    acc_a = accuracy_score(test_labels, pred_a)
    f1_a = f1_score(test_labels, pred_a, average='weighted')
    
    acc_b = accuracy_score(test_labels, pred_b)
    f1_b = f1_score(test_labels, pred_b, average='weighted')
    
    print(f"✅ Modelo A - Accuracy: {acc_a:.3f}, F1: {f1_a:.3f}")
    print(f"✅ Modelo B - Accuracy: {acc_b:.3f}, F1: {f1_b:.3f}")
    
    # Crear metadata
    metadata_a = {
        "model_type": "RandomForest",
        "accuracy": float(acc_a),
        "f1_score": float(f1_a),
        "overall_score": float((acc_a + f1_a) / 2),
        "created_at": "2025-10-13T10:00:00"
    }
    
    metadata_b = {
        "model_type": "LogisticRegression", 
        "accuracy": float(acc_b),
        "f1_score": float(f1_b),
        "overall_score": float((acc_b + f1_b) / 2),
        "created_at": "2025-10-13T10:00:00"
    }
    
    import json
    with open('backend/models/saved/demo_model_a_metadata.json', 'w') as f:
        json.dump(metadata_a, f, indent=2)
    
    with open('backend/models/saved/demo_model_b_metadata.json', 'w') as f:
        json.dump(metadata_b, f, indent=2)
    
    print("🎉 Modelos de demostración creados exitosamente!")
    print("📁 Archivos creados:")
    print("   - demo_model_a.pkl (Random Forest)")
    print("   - demo_model_b.pkl (Logistic Regression)")
    print("   - demo_vectorizer.pkl")
    print("   - demo_model_a_metadata.json")
    print("   - demo_model_b_metadata.json")

if __name__ == "__main__":
    create_demo_models()
