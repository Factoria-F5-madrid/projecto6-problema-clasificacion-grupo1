#!/usr/bin/env python3
"""
Script para implementar reglas más fuertes - VERSIÓN CORREGIDA
Prioriza reglas sobre ML para palabras ofensivas obvias
"""

import re
import pandas as pd
import numpy as np
import joblib

def create_strong_rules():
    """Crear reglas más fuertes para palabras ofensivas"""
    print("🔧 Creando reglas más fuertes...")
    
    # Palabras ofensivas directas (ALTA PRIORIDAD)
    offensive_words = [
        'fuck', 'shit', 'asshole', 'bastard', 'damn', 'hell',
        'stupid', 'idiot', 'moron', 'dumb', 'retard', 'crap',
        'bitch', 'whore', 'slut', 'fag', 'nigger', 'kike'
    ]
    
    # Patrones de hate speech (ALTA PRIORIDAD)
    hate_patterns = [
        r'\bwomen\s+are\s+inferior\b',
        r'\ball\s+\w+\s+are\s+\w+\b',
        r'\b\w+\s+people\s+are\s+\w+\b',
        r'\bsuperior\s+to\s+\w+\b',
        r'\binferior\s+to\s+\w+\b'
    ]
    
    # Palabras ofensivas con evasiones
    evasion_patterns = [
        r'f[\W_]*u[\W_]*c[\W_]*k',
        r's[\W_]*h[\W_]*i[\W_]*t',
        r'a[\W_]*s[\W_]*s[\W_]*h[\W_]*o[\W_]*l[\W_]*e',
        r's[\W_]*t[\W_]*u[\W_]*p[\W_]*i[\W_]*d',
        r'i[\W_]*d[\W_]*i[\W_]*o[\W_]*t'
    ]
    
    def classify_with_strong_rules(text):
        """Clasificar con reglas fuertes"""
        text_lower = text.lower()
        
        # 1. Verificar hate speech (PRIORIDAD MÁXIMA)
        for pattern in hate_patterns:
            if re.search(pattern, text_lower):
                return 'Hate Speech', 0.9
        
        # 2. Verificar palabras ofensivas directas
        for word in offensive_words:
            if word in text_lower:
                return 'Offensive Language', 0.8
        
        # 3. Verificar evasiones
        for pattern in evasion_patterns:
            if re.search(pattern, text_lower):
                return 'Offensive Language', 0.7
        
        # 4. Si no encuentra nada, usar ML
        return None, 0.0
    
    return classify_with_strong_rules

def test_strong_rules():
    """Probar reglas fuertes"""
    print("🧪 Probando reglas fuertes...")
    
    classify_func = create_strong_rules()
    
    test_cases = [
        "fuck you",
        "F*ck you", 
        "asshole",
        "@sshole",
        "stupid",
        "st*pid",
        "Women are inferior to men",
        "This is fucking stupid",
        "Hello, how are you?",
        "Thank you very much"
    ]
    
    print("\n📊 Resultados con reglas fuertes:")
    print("-" * 50)
    
    for text in test_cases:
        prediction, confidence = classify_func(text)
        
        if prediction:
            print(f"'{text}' → {prediction} ({confidence:.1%}) [REGLAS]")
        else:
            print(f"'{text}' → Usar ML (reglas no detectaron nada)")

class StrongRulesHybrid:
    """Sistema híbrido con reglas fuertes"""
    
    def __init__(self):
        self.classify_rules = create_strong_rules()
        self.class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        
        # Cargar modelo ML como fallback
        try:
            self.model = joblib.load("backend/models/saved/improved_model.pkl")
            self.vectorizer = joblib.load("backend/models/saved/improved_vectorizer.pkl")
            self.ml_available = True
            print("✅ Modelo ML cargado como fallback")
        except Exception as e:
            print(f"⚠️ No se pudo cargar modelo ML: {e}")
            self.ml_available = False
    
    def predict(self, text):
        """Hacer predicción con reglas fuertes + ML"""
        # 1. Intentar reglas fuertes primero
        rule_pred, rule_conf = self.classify_rules(text)
        
        if rule_pred:
            return {
                'prediction': rule_pred,
                'confidence': rule_conf,
                'method': 'strong_rules',
                'explanation': 'Clasificado por reglas fuertes (palabras ofensivas detectadas)'
            }
        
        # 2. Si no hay reglas, usar ML
        if self.ml_available:
            try:
                X = self.vectorizer.transform([text])
                probabilities = self.model.predict_proba(X)[0]
                prediction = np.argmax(probabilities)
                confidence = max(probabilities)
                
                return {
                    'prediction': self.class_names[prediction],
                    'confidence': confidence,
                    'method': 'ml_fallback',
                    'explanation': 'Clasificado por ML (reglas no detectaron patrones)'
                }
            except Exception as e:
                print(f"⚠️ Error en ML: {e}")
        
        # 3. Fallback final
        return {
            'prediction': 'Neither',
            'confidence': 0.5,
            'method': 'fallback',
            'explanation': 'Clasificación por defecto'
        }

def main():
    """Función principal"""
    print("🔧 IMPLEMENTANDO REGLAS MÁS FUERTES - VERSIÓN CORREGIDA")
    print("=" * 60)
    
    # Probar reglas fuertes
    test_strong_rules()
    
    # Crear sistema híbrido
    print("\n🚀 Creando sistema híbrido con reglas fuertes...")
    hybrid_system = StrongRulesHybrid()
    
    # Probar sistema híbrido
    print("\n🧪 Probando sistema híbrido con reglas fuertes:")
    print("-" * 50)
    
    test_cases = [
        "fuck you",
        "F*ck you", 
        "asshole",
        "@sshole",
        "stupid",
        "st*pid",
        "Women are inferior to men",
        "This is fucking stupid",
        "Hello, how are you?",
        "Thank you very much"
    ]
    
    for text in test_cases:
        result = hybrid_system.predict(text)
        print(f"'{text}' → {result['prediction']} ({result['confidence']:.1%}) [{result['method']}]")
    
    # Guardar sistema híbrido
    joblib.dump(hybrid_system, "backend/models/saved/strong_rules_hybrid.pkl")
    
    print(f"\n✅ Sistema híbrido con reglas fuertes guardado!")
    print(f"📁 Archivo: backend/models/saved/strong_rules_hybrid.pkl")
    
    print(f"\n🎯 Ahora las palabras ofensivas se detectan por reglas (más confiable)")
    print(f"💡 'fuck you' ahora debería clasificarse como 'Offensive Language' con 80% de confianza")

if __name__ == "__main__":
    main()
