#!/usr/bin/env python3
"""
Debuggear específicamente el caso "F*ck you"
"""

import sys
import os
sys.path.append('backend')

import joblib
import numpy as np
from utils.robust_preprocessor import RobustPreprocessor
from utils.confidence_booster import ConfidenceBooster

def debug_fuck_you():
    """Debuggear específicamente "F*ck you" """
    print("🔍 DEBUGGEANDO 'F*ck you'")
    print("=" * 50)
    
    # Cargar componentes
    model = joblib.load("backend/models/saved/balanced_model.pkl")
    vectorizer = joblib.load("backend/models/saved/balanced_vectorizer.pkl")
    preprocessor = RobustPreprocessor()
    confidence_booster = ConfidenceBooster()
    
    # Texto de prueba
    text = "F*ck you"
    print(f"📝 Texto original: '{text}'")
    
    # Paso 1: Preprocesar
    preprocessed = preprocessor.preprocess_text(
        text,
        normalize_evasions=True,
        clean_text=True,
        extract_features=True
    )
    processed_text = preprocessed['processed_text']
    evasions = preprocessed['evasions_found']
    
    print(f"📝 Texto procesado: '{processed_text}'")
    print(f"🚫 Evasiones detectadas: {evasions}")
    
    # Paso 2: Verificar si "fuck" está en el vocabulario
    feature_names = vectorizer.get_feature_names_out()
    print(f"🔍 'fuck' en vocabulario: {'fuck' in feature_names}")
    print(f"🔍 'you' en vocabulario: {'you' in feature_names}")
    
    # Buscar palabras similares
    fuck_similar = [word for word in feature_names if 'fuck' in word.lower()]
    print(f"🔍 Palabras con 'fuck': {fuck_similar[:10]}")  # Mostrar solo las primeras 10
    
    # Paso 3: Vectorizar
    X = vectorizer.transform([processed_text])
    print(f"📊 Vector shape: {X.shape}")
    print(f"📊 Non-zero features: {X.nnz}")
    
    # Ver qué features están activas
    feature_names = vectorizer.get_feature_names_out()
    active_features = X.nonzero()[1]
    active_words = [feature_names[i] for i in active_features]
    print(f"🔍 Palabras activas: {active_words}")
    
    # Paso 4: Predecir
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    print(f"🤖 Predicción numérica: {prediction}")
    print(f"📊 Probabilidades: {probabilities}")
    
    # Paso 5: Aplicar booster
    classes = ['Hate Speech', 'Offensive Language', 'Neither']
    boosted_probs, explanation = confidence_booster.boost_confidence(
        processed_text, probabilities, classes
    )
    print(f"🚀 Probabilidades mejoradas: {boosted_probs}")
    print(f"💡 Explicación: {explanation}")
    
    # Paso 6: Resultado final
    final_prediction = np.argmax(boosted_probs)
    final_probability = boosted_probs[final_prediction]
    class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    final_prediction_text = class_mapping.get(final_prediction, 'Unknown')
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"   Predicción: {final_prediction_text}")
    print(f"   Confianza: {final_probability:.1%}")
    
    # Comparar con casos que funcionan
    print(f"\n🔄 COMPARACIÓN CON CASOS QUE FUNCIONAN:")
    
    working_cases = ["asshole", "stupid", "idiot"]
    for case in working_cases:
        X_case = vectorizer.transform([case])
        pred_case = model.predict(X_case)[0]
        prob_case = model.predict_proba(X_case)[0]
        pred_text_case = class_mapping.get(pred_case, 'Unknown')
        conf_case = max(prob_case) * 100
        
        print(f"   '{case}': {pred_text_case} ({conf_case:.1%})")
    
    # Verificar si el problema está en el preprocesamiento
    print(f"\n🔧 VERIFICANDO PREPROCESAMIENTO:")
    print(f"   Patrones de evasión disponibles:")
    for pattern in preprocessor.evasion_patterns.keys():
        if 'fuck' in pattern.lower():
            print(f"     - {pattern} → {preprocessor.evasion_patterns[pattern]}")

if __name__ == "__main__":
    debug_fuck_you()
