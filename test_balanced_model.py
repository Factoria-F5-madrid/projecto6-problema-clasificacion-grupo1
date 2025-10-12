#!/usr/bin/env python3
"""
Test del modelo balanceado
"""

import sys
import os
sys.path.append('backend')

import joblib
import numpy as np

def test_balanced_model():
    """Probar el modelo balanceado"""
    print("🧪 TEST DEL MODELO BALANCEADO")
    print("=" * 50)
    
    # Cargar modelo balanceado
    print("📊 Cargando modelo balanceado...")
    model = joblib.load("backend/models/saved/balanced_model.pkl")
    vectorizer = joblib.load("backend/models/saved/balanced_vectorizer.pkl")
    
    # Casos de prueba
    test_cases = [
        "@sshole",
        "asshole", 
        "fuck",
        "shit",
        "stupid",
        "idiot",
        "Women are inferior to men",
        "This is fucking stupid",
        "I hate all immigrants",
        "Hello, how are you?"
    ]
    
    print("\n🔍 PROBANDO CASOS:")
    print("-" * 50)
    
    for text in test_cases:
        print(f"\n📝 Texto: '{text}'")
        
        # Vectorizar
        X = vectorizer.transform([text])
        
        # Predecir
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Mapear predicción
        class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        prediction_text = class_mapping.get(prediction, 'Unknown')
        confidence = max(probabilities) * 100
        
        print(f"   Predicción: {prediction_text}")
        print(f"   Confianza: {confidence:.1f}%")
        print(f"   Probabilidades:")
        for i, (class_name, prob) in enumerate(zip(class_mapping.values(), probabilities)):
            print(f"     - {class_name}: {prob:.1%}")

if __name__ == "__main__":
    test_balanced_model()
