#!/usr/bin/env python3
"""
Test del preprocesamiento con modelo balanceado
"""

import sys
import os
sys.path.append('backend')

import joblib
import numpy as np
from utils.robust_preprocessor import RobustPreprocessor

def test_preprocessing_with_model():
    """Probar preprocesamiento con modelo balanceado"""
    print("🧪 TEST DE PREPROCESAMIENTO CON MODELO BALANCEADO")
    print("=" * 60)
    
    # Cargar modelo balanceado
    print("📊 Cargando modelo balanceado...")
    model = joblib.load("backend/models/saved/balanced_model.pkl")
    vectorizer = joblib.load("backend/models/saved/balanced_vectorizer.pkl")
    
    # Inicializar preprocesador
    preprocessor = RobustPreprocessor()
    
    # Casos de prueba
    test_cases = [
        "@sshole",
        "F*ck you",
        "H*ll no",
        "st*pid",
        "1d10t"
    ]
    
    print("\n🔍 PROBANDO CASOS CON PREPROCESAMIENTO:")
    print("-" * 60)
    
    for text in test_cases:
        print(f"\n📝 Texto original: '{text}'")
        
        # Preprocesar
        preprocessed = preprocessor.preprocess_text(
            text,
            normalize_evasions=True,
            clean_text=True,
            extract_features=True
        )
        
        processed_text = preprocessed['processed_text']
        evasions = preprocessed['evasions_found']
        
        print(f"   Texto procesado: '{processed_text}'")
        print(f"   Evasiones detectadas: {evasions}")
        
        # Vectorizar texto procesado
        X = vectorizer.transform([processed_text])
        
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
        
        # Comparar con texto sin preprocesar
        print(f"\n   🔄 COMPARACIÓN (sin preprocesar):")
        X_original = vectorizer.transform([text])
        prediction_original = model.predict(X_original)[0]
        probabilities_original = model.predict_proba(X_original)[0]
        prediction_text_original = class_mapping.get(prediction_original, 'Unknown')
        confidence_original = max(probabilities_original) * 100
        
        print(f"     Predicción: {prediction_text_original}")
        print(f"     Confianza: {confidence_original:.1f}%")
        
        # Mostrar mejora
        if prediction != prediction_original:
            print(f"     ✅ MEJORA: {prediction_text_original} → {prediction_text}")
        else:
            print(f"     ⚠️ Sin cambio en predicción")

if __name__ == "__main__":
    test_preprocessing_with_model()
