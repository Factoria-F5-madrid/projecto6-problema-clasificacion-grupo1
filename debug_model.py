#!/usr/bin/env python3
"""
Script de diagnóstico para debuggear el modelo
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def debug_model():
    """Debuggear el modelo para entender por qué falla"""
e 
    
    df = pd.read_csv("backend/data/processed/cleaned_tweets.csv")
    print(f"Total de datos: {len(df)}")
    print(f"Distribución de clases:")
    print(df['class'].value_counts())
    print()
    
    # Preparar datos
    X_text = df['clean_tweet_improved'].fillna('')
    y = df['class']
    
    # Crear vectorizador
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1,3),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(X_text)
    
    # Entrenar modelo
    print("🤖 Entrenando modelo...")
    model = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        solver='liblinear',
        random_state=42
    )
    model.fit(X, y)
    
    # Probar con casos específicos
    test_cases = [
        "asshole",
        "fuck",
        "shit",
        "stupid",
        "idiot",
        "hello",
        "thank you"
    ]
    
    print("\n🧪 PROBANDO CASOS ESPECÍFICOS:")
    print("-" * 50)
    
    for text in test_cases:
        print(f"\n📝 Texto: '{text}'")
        
        # Vectorizar
        X_test = vectorizer.transform([text])
        
        # Predecir
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        
        # Mapear predicción
        class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        prediction_text = class_mapping.get(prediction, 'Unknown')
        
        print(f"   Predicción: {prediction_text}")
        print(f"   Probabilidades: {dict(zip(class_mapping.values(), probabilities))}")
        
        # Verificar si la palabra está en el vocabulario
        feature_names = vectorizer.get_feature_names_out()
        if text in feature_names:
            print(f"   ✅ Palabra '{text}' encontrada en vocabulario")
        else:
            print(f"   ❌ Palabra '{text}' NO encontrada en vocabulario")
    
    # Verificar vocabulario
    print(f"\n📚 VOCABULARIO DEL MODELO:")
    print(f"Total de palabras: {len(vectorizer.get_feature_names_out())}")
    
    # Buscar palabras ofensivas en el vocabulario
    offensive_words = ['asshole', 'fuck', 'shit', 'stupid', 'idiot']
    print(f"\n🔍 PALABRAS OFENSIVAS EN VOCABULARIO:")
    for word in offensive_words:
        if word in vectorizer.get_feature_names_out():
            print(f"   ✅ '{word}' - SÍ está en vocabulario")
        else:
            print(f"   ❌ '{word}' - NO está en vocabulario")
    
    # Verificar datos de entrenamiento
    print(f"\n📊 ANÁLISIS DE DATOS DE ENTRENAMIENTO:")
    
    # Buscar ejemplos de "asshole" en los datos
    asshole_examples = df[df['clean_tweet_improved'].str.contains('asshole', case=False, na=False)]
    print(f"Ejemplos con 'asshole': {len(asshole_examples)}")
    if len(asshole_examples) > 0:
        print("Clasificaciones de ejemplos con 'asshole':")
        print(asshole_examples['class'].value_counts())
        print("\nEjemplos:")
        for i, row in asshole_examples.head(3).iterrows():
            print(f"  - '{row['clean_tweet_improved']}' → {row['class']}")
    
    # Buscar ejemplos de "fuck" en los datos
    fuck_examples = df[df['clean_tweet_improved'].str.contains('fuck', case=False, na=False)]
    print(f"\nEjemplos con 'fuck': {len(fuck_examples)}")
    if len(fuck_examples) > 0:
        print("Clasificaciones de ejemplos con 'fuck':")
        print(fuck_examples['class'].value_counts())
    
    # Verificar si el problema está en el preprocesamiento
    print(f"\n🔧 VERIFICANDO PREPROCESAMIENTO:")
    from utils.robust_preprocessor import RobustPreprocessor
    
    preprocessor = RobustPreprocessor()
    result = preprocessor.preprocess_text("@sshole", normalize_evasions=True, clean_text=True)
    print(f"Texto original: '@sshole'")
    print(f"Texto procesado: '{result['processed_text']}'")
    print(f"Evasiones detectadas: {result['evasions_found']}")
    
    # Probar con texto procesado
    processed_text = result['processed_text']
    X_processed = vectorizer.transform([processed_text])
    prediction_processed = model.predict(X_processed)[0]
    probabilities_processed = model.predict_proba(X_processed)[0]
    
    print(f"\nCon texto procesado '{processed_text}':")
    print(f"Predicción: {class_mapping.get(prediction_processed, 'Unknown')}")
    print(f"Probabilidades: {dict(zip(class_mapping.values(), probabilities_processed))}")

if __name__ == "__main__":
    debug_model()
