#!/usr/bin/env python3
"""
Verificar el mapeo de clases del modelo balanceado
"""

import sys
import os
sys.path.append('backend')

import joblib
import numpy as np

def check_class_mapping():
    """Verificar el mapeo de clases"""
    print("🔍 VERIFICANDO MAPEO DE CLASES")
    print("=" * 50)
    
    # Cargar modelo balanceado
    model = joblib.load("backend/models/saved/balanced_model.pkl")
    vectorizer = joblib.load("backend/models/saved/balanced_vectorizer.pkl")
    
    # Verificar las clases del modelo
    print(f"Clases del modelo: {model.classes_}")
    print(f"Tipo de modelo: {type(model)}")
    
    # Probar con "asshole"
    X = vectorizer.transform(["asshole"])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    print(f"\nProbando con 'asshole':")
    print(f"Predicción numérica: {prediction}")
    print(f"Probabilidades: {probabilities}")
    
    # Mapeo actual en la app
    class_mapping_app = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    print(f"\nMapeo actual en la app: {class_mapping_app}")
    print(f"Predicción según app: {class_mapping_app.get(prediction, 'Unknown')}")
    
    # Verificar si el mapeo es correcto
    print(f"\n¿Es correcto el mapeo?")
    print(f"Clase 0: {model.classes_[0]} → {class_mapping_app[0]}")
    print(f"Clase 1: {model.classes_[1]} → {class_mapping_app[1]}")
    print(f"Clase 2: {model.classes_[2]} → {class_mapping_app[2]}")

if __name__ == "__main__":
    check_class_mapping()
