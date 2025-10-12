#!/usr/bin/env python3
"""
Debuggear la lógica de la app
"""

import sys
import os
sys.path.append('backend')

import joblib
import numpy as np
from utils.robust_preprocessor import RobustPreprocessor
from utils.confidence_booster import ConfidenceBooster

def debug_app_logic():
    """Debuggear la lógica completa de la app"""
    print("🔍 DEBUGGEANDO LÓGICA DE LA APP")
    print("=" * 50)
    
    # Cargar componentes
    model = joblib.load("backend/models/saved/balanced_model.pkl")
    vectorizer = joblib.load("backend/models/saved/balanced_vectorizer.pkl")
    preprocessor = RobustPreprocessor()
    confidence_booster = ConfidenceBooster()
    
    # Texto de prueba
    text = "@sshole"
    print(f"📝 Texto original: '{text}'")
    
    # Paso 1: Preprocesar
    preprocessed = preprocessor.preprocess_text(
        text,
        normalize_evasions=True,
        clean_text=True,
        extract_features=True
    )
    processed_text = preprocessed['processed_text']
    print(f"📝 Texto procesado: '{processed_text}'")
    
    # Paso 2: Vectorizar
    X = vectorizer.transform([processed_text])
    print(f"📊 Vector shape: {X.shape}")
    
    # Paso 3: Predecir
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    print(f"🤖 Predicción numérica: {prediction}")
    print(f"📊 Probabilidades originales: {probabilities}")
    
    # Paso 4: Aplicar booster de confianza
    classes = ['Hate Speech', 'Offensive Language', 'Neither']
    boosted_probs, explanation = confidence_booster.boost_confidence(
        processed_text, probabilities, classes
    )
    print(f"🚀 Probabilidades mejoradas: {boosted_probs}")
    print(f"💡 Explicación: {explanation}")
    
    # Paso 5: Obtener predicción final
    final_prediction = np.argmax(boosted_probs)
    final_probability = boosted_probs[final_prediction]
    print(f"🎯 Predicción final: {final_prediction}")
    print(f"🎯 Probabilidad final: {final_probability:.1%}")
    
    # Paso 6: Mapear a nombres
    class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    final_prediction_text = class_mapping.get(final_prediction, 'Unknown')
    print(f"🏷️ Clasificación final: {final_prediction_text}")
    
    # Comparar con predicción original
    original_prediction_text = class_mapping.get(prediction, 'Unknown')
    original_probability = probabilities[prediction]
    print(f"\n🔄 COMPARACIÓN:")
    print(f"   Sin booster: {original_prediction_text} ({original_probability:.1%})")
    print(f"   Con booster: {final_prediction_text} ({final_probability:.1%})")
    
    if prediction != final_prediction:
        print(f"   ✅ BOOSTER CAMBIÓ LA PREDICCIÓN")
    else:
        print(f"   ⚠️ Booster no cambió la predicción")

if __name__ == "__main__":
    debug_app_logic()
