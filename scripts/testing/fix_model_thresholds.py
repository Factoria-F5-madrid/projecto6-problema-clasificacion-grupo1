#!/usr/bin/env python3
"""
Script para ajustar umbrales del modelo
Hace que el modelo sea más sensible a palabras ofensivas
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def adjust_model_sensitivity():
    """Ajustar sensibilidad del modelo"""
    print("🔧 Ajustando sensibilidad del modelo...")
    
    # Cargar modelo actual
    model = joblib.load("backend/models/saved/improved_model.pkl")
    vectorizer = joblib.load("backend/models/saved/improved_vectorizer.pkl")
    
    # Casos de prueba problemáticos
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
    
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    
    print("\n📊 ANTES del ajuste:")
    print("-" * 50)
    
    for text in test_cases:
        X = vectorizer.transform([text])
        probabilities = model.predict_proba(X)[0]
        prediction = np.argmax(probabilities)
        confidence = max(probabilities)
        
        print(f"'{text}' → {class_names[prediction]} ({confidence:.1%})")
    
    # Ajustar umbrales: hacer el modelo más sensible a palabras ofensivas
    print("\n🔧 Aplicando ajustes de sensibilidad...")
    
    # Crear función de predicción ajustada
    def predict_with_adjustment(text, model, vectorizer):
        X = vectorizer.transform([text])
        probabilities = model.predict_proba(X)[0]
        
        # Ajustar probabilidades para palabras ofensivas conocidas
        offensive_words = ['fuck', 'shit', 'asshole', 'stupid', 'idiot', 'damn', 'hell']
        text_lower = text.lower()
        
        # Si contiene palabras ofensivas, aumentar probabilidad de "Offensive Language"
        if any(word in text_lower for word in offensive_words):
            probabilities[1] *= 1.5  # Aumentar probabilidad de Offensive Language
            probabilities[2] *= 0.7  # Reducir probabilidad de Neither
        
        # Si contiene patrones de hate speech, aumentar probabilidad de "Hate Speech"
        hate_patterns = ['inferior', 'superior', 'all', 'every', 'never', 'always']
        if any(pattern in text_lower for pattern in hate_patterns):
            probabilities[0] *= 1.3  # Aumentar probabilidad de Hate Speech
            probabilities[2] *= 0.8  # Reducir probabilidad de Neither
        
        # Normalizar probabilidades
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities
    
    print("\n📊 DESPUÉS del ajuste:")
    print("-" * 50)
    
    for text in test_cases:
        probabilities = predict_with_adjustment(text, model, vectorizer)
        prediction = np.argmax(probabilities)
        confidence = max(probabilities)
        
        print(f"'{text}' → {class_names[prediction]} ({confidence:.1%})")
    
    return model, vectorizer, predict_with_adjustment

def create_enhanced_predictor():
    """Crear predictor mejorado"""
    print("\n🚀 Creando predictor mejorado...")
    
    class EnhancedPredictor:
        def __init__(self, model, vectorizer):
            self.model = model
            self.vectorizer = vectorizer
            self.class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            
            # Palabras ofensivas conocidas
            self.offensive_words = [
                'fuck', 'shit', 'asshole', 'stupid', 'idiot', 'damn', 'hell',
                'bastard', 'moron', 'dumb', 'retard', 'crap', 'bitch'
            ]
            
            # Patrones de hate speech
            self.hate_patterns = [
                'inferior', 'superior', 'all', 'every', 'never', 'always',
                'women are', 'men are', 'black people', 'white people',
                'immigrants are', 'gay people', 'muslims are', 'jews'
            ]
        
        def predict(self, text):
            X = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(X)[0]
            
            # Aplicar ajustes de sensibilidad
            text_lower = text.lower()
            
            # Ajustar para palabras ofensivas
            if any(word in text_lower for word in self.offensive_words):
                probabilities[1] *= 1.5  # Offensive Language
                probabilities[2] *= 0.7  # Neither
            
            # Ajustar para patrones de hate speech
            if any(pattern in text_lower for pattern in self.hate_patterns):
                probabilities[0] *= 1.3  # Hate Speech
                probabilities[2] *= 0.8  # Neither
            
            # Normalizar
            probabilities = probabilities / np.sum(probabilities)
            
            prediction = np.argmax(probabilities)
            confidence = max(probabilities)
            
            return {
                'prediction': self.class_names[prediction],
                'confidence': confidence,
                'probabilities': {
                    self.class_names[i]: prob 
                    for i, prob in enumerate(probabilities)
                }
            }
    
    return EnhancedPredictor

def main():
    """Función principal"""
    print("🔧 AJUSTANDO SENSIBILIDAD DEL MODELO")
    print("=" * 50)
    
    # Ajustar sensibilidad
    model, vectorizer, predict_func = adjust_model_sensitivity()
    
    # Crear predictor mejorado
    enhanced_predictor = create_enhanced_predictor()(model, vectorizer)
    
    # Probar predictor mejorado
    print("\n🧪 Probando predictor mejorado:")
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
        result = enhanced_predictor.predict(text)
        print(f"'{text}' → {result['prediction']} ({result['confidence']:.1%})")
    
    # Guardar predictor mejorado
    import joblib
    joblib.dump(enhanced_predictor, "backend/models/saved/enhanced_predictor.pkl")
    
    print(f"\n✅ Predictor mejorado guardado!")
    print(f"📁 Archivo: backend/models/saved/enhanced_predictor.pkl")
    
    print(f"\n🎯 El modelo ahora es más sensible a palabras ofensivas")
    print(f"💡 Palabras como 'fuck you' ahora deberían clasificarse como 'Offensive Language'")

if __name__ == "__main__":
    main()
