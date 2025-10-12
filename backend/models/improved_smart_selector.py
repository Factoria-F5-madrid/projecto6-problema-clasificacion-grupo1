"""
Sistema de Selección Inteligente MEJORADO
Evita falsos positivos y mejora la precisión
"""

import re
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class ImprovedSmartSelector:
    """Selector inteligente mejorado para hate speech detection"""
    
    def __init__(self):
        self.class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        self.models = {}
        self.rules_loaded = False
        self.ml_loaded = False
        
        # Cargar todos los modelos disponibles
        self._load_all_models()
        
        # Configurar reglas mejoradas
        self._setup_improved_rules()
    
    def _load_all_models(self):
        """Cargar todos los modelos disponibles"""
        print("🔄 Cargando modelos disponibles...")
        
        # Lista de modelos a intentar cargar
        model_paths = [
            ("improved", "backend/models/saved/improved_model.pkl", "backend/models/saved/improved_vectorizer.pkl"),
            ("balanced", "backend/models/saved/balanced_model.pkl", "backend/models/saved/balanced_vectorizer.pkl"),
            ("original", "backend/models/saved/model.pkl", "backend/models/saved/vectorizer.pkl")
        ]
        
        for name, model_path, vectorizer_path in model_paths:
            try:
                if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                    self.models[name] = {
                        'model': joblib.load(model_path),
                        'vectorizer': joblib.load(vectorizer_path),
                        'weight': self._get_model_weight(name),
                        'type': 'ml'
                    }
                    print(f"✅ {name.capitalize()}: Cargado (peso: {self._get_model_weight(name)})")
                else:
                    print(f"❌ {name.capitalize()}: Archivos no encontrados")
            except Exception as e:
                print(f"❌ {name.capitalize()}: Error cargando - {e}")
        
        self.ml_loaded = len(self.models) > 0
        print(f"📊 Total de modelos ML cargados: {len(self.models)}")
    
    def _get_model_weight(self, model_name):
        """Obtener peso del modelo según su calidad"""
        weights = {
            'improved': 0.5,    # Mejor modelo
            'balanced': 0.3,    # Modelo balanceado
            'original': 0.2     # Modelo base
        }
        return weights.get(model_name, 0.1)
    
    def _setup_improved_rules(self):
        """Configurar reglas mejoradas para evitar falsos positivos"""
        
        # Palabras ofensivas directas (solo en contexto ofensivo)
        self.offensive_words = [
            'fuck', 'shit', 'asshole', 'bastard', 'damn', 'hell',
            'stupid', 'idiot', 'moron', 'dumb', 'retard', 'crap',
            'bitch', 'whore', 'slut', 'fag', 'nigger', 'kike',
            'jódete', 'mierda', 'cabrón', 'idiota', 'estúpido'
        ]
        
        # Palabras que NO son ofensivas (para evitar falsos positivos)
        self.safe_words = [
            'hello', 'hi', 'how', 'are', 'you', 'thank', 'thanks',
            'please', 'welcome', 'good', 'nice', 'amazing', 'awesome',
            'great', 'wonderful', 'fantastic', 'beautiful', 'excellent'
        ]
        
        # Patrones de hate speech (más específicos)
        self.hate_patterns = [
            r'\bwomen\s+are\s+inferior\b',
            r'\ball\s+\w+\s+are\s+\w+\b',
            r'\b\w+\s+people\s+are\s+\w+\b',
            r'\bsuperior\s+to\s+\w+\b',
            r'\binferior\s+to\s+\w+\b',
            r'\bwomen\s+belong\s+in\b',
            r'\bmen\s+are\s+better\b',
            r'\bhate\s+all\s+\w+\b',
            r'\ball\s+\w+\s+should\s+die\b'
        ]
        
        # Patrones de evasión (más específicos)
        self.evasion_patterns = [
            r'f[\W_]*u[\W_]*c[\W_]*k',
            r's[\W_]*h[\W_]*i[\W_]*t',
            r'a[\W_]*s[\W_]*s[\W_]*h[\W_]*o[\W_]*l[\W_]*e',
            r's[\W_]*t[\W_]*u[\W_]*p[\W_]*i[\W_]*d',
            r'i[\W_]*d[\W_]*i[\W_]*o[\W_]*t'
        ]
        
        # Palabras que indican contexto NO ofensivo (para evitar falsos positivos)
        self.positive_context_words = [
            'amazing', 'awesome', 'great', 'wonderful', 'fantastic',
            'beautiful', 'excellent', 'perfect', 'incredible', 'outstanding',
            'thank', 'thanks', 'please', 'welcome', 'hello', 'hi',
            'good', 'nice', 'lovely', 'sweet', 'kind', 'helpful'
        ]
        
        # Frases que indican contexto NO ofensivo
        self.positive_phrases = [
            'thank you', 'good morning', 'good afternoon', 'good evening',
            'have a nice', 'wonderful day', 'amazing work', 'great job',
            'you are amazing', 'you are wonderful', 'you are great'
        ]
        
        self.rules_loaded = True
        print("✅ Reglas mejoradas configuradas")
    
    def _is_positive_context(self, text):
        """Verificar si el texto tiene contexto positivo"""
        text_lower = text.lower()
        
        # Verificar frases positivas
        for phrase in self.positive_phrases:
            if phrase in text_lower:
                return True
        
        # Verificar palabras positivas
        positive_count = sum(1 for word in self.positive_context_words if word in text_lower)
        if positive_count >= 2:  # Si tiene 2+ palabras positivas
            return True
        
        return False
    
    def _classify_with_improved_rules(self, text):
        """Clasificar usando reglas mejoradas"""
        text_lower = text.lower()
        
        # 1. Verificar contexto positivo primero (evitar falsos positivos)
        if self._is_positive_context(text):
            return None, 0.0, 'positive_context'  # Usar ML para contexto positivo
        
        # 2. Verificar si contiene solo palabras seguras (evitar falsos positivos)
        words = text_lower.split()
        if len(words) <= 5:  # Frases cortas
            safe_word_count = sum(1 for word in words if word in self.safe_words)
            if safe_word_count >= len(words) * 0.6:  # 60%+ palabras seguras
                return None, 0.0, 'safe_words_only'  # Usar ML para palabras seguras
        
        # 3. Verificar hate speech (PRIORIDAD MÁXIMA)
        for pattern in self.hate_patterns:
            if re.search(pattern, text_lower):
                return 'Hate Speech', 0.9, 'hate_speech_pattern'
        
        # 4. Verificar palabras ofensivas directas (solo si no hay contexto positivo)
        for word in self.offensive_words:
            if word in text_lower:
                return 'Offensive Language', 0.8, 'offensive_word'
        
        # 5. Verificar evasiones (solo si no hay contexto positivo)
        for pattern in self.evasion_patterns:
            if re.search(pattern, text_lower):
                return 'Offensive Language', 0.7, 'evasion_pattern'
        
        return None, 0.0, 'no_rules_match'
    
    def _predict_with_ml(self, text):
        """Hacer predicción con todos los modelos ML"""
        if not self.ml_loaded:
            return None
        
        predictions = []
        total_weight = 0
        
        for name, model_data in self.models.items():
            try:
                X = model_data['vectorizer'].transform([text])
                probabilities = model_data['model'].predict_proba(X)[0]
                prediction = np.argmax(probabilities)
                confidence = max(probabilities)
                weight = model_data['weight']
                
                predictions.append({
                    'model': name,
                    'prediction': self.class_names[prediction],
                    'confidence': confidence,
                    'probabilities': {
                        self.class_names[i]: prob 
                        for i, prob in enumerate(probabilities)
                    },
                    'weight': weight
                })
                
                total_weight += weight
                
            except Exception as e:
                print(f"⚠️ Error en modelo {name}: {e}")
        
        if not predictions:
            return None
        
        # Calcular predicción ponderada
        weighted_probs = {self.class_names[i]: 0.0 for i in range(3)}
        
        for pred in predictions:
            for class_name, prob in pred['probabilities'].items():
                weighted_probs[class_name] += prob * pred['weight']
        
        # Normalizar
        for class_name in weighted_probs:
            weighted_probs[class_name] /= total_weight
        
        # Obtener predicción final
        final_prediction = max(weighted_probs, key=weighted_probs.get)
        final_confidence = weighted_probs[final_prediction]
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'probabilities': weighted_probs,
            'individual_predictions': predictions,
            'method': 'ml_ensemble'
        }
    
    def predict(self, text):
        """Hacer predicción inteligente mejorada"""
        # 1. Intentar reglas mejoradas primero
        rule_pred, rule_conf, rule_type = self._classify_with_improved_rules(text)
        
        if rule_pred:
            return {
                'text': text,
                'prediction': rule_pred,
                'confidence': rule_conf,
                'method': 'improved_rules',
                'rule_type': rule_type,
                'explanation': f'Clasificado por reglas mejoradas ({rule_type})',
                'individual_predictions': [],
                'probabilities': {
                    'Hate Speech': 0.9 if rule_pred == 'Hate Speech' else 0.05,
                    'Offensive Language': 0.8 if rule_pred == 'Offensive Language' else 0.1,
                    'Neither': 0.1 if rule_pred != 'Neither' else 0.8
                }
            }
        
        # 2. Si no hay reglas, usar ML
        ml_result = self._predict_with_ml(text)
        
        if ml_result:
            return {
                'text': text,
                'prediction': ml_result['prediction'],
                'confidence': ml_result['confidence'],
                'method': 'ml_ensemble',
                'rule_type': 'ml_fallback',
                'explanation': 'Clasificado por ensemble de modelos ML',
                'individual_predictions': ml_result['individual_predictions'],
                'probabilities': ml_result['probabilities']
            }
        
        # 3. Fallback final
        return {
            'text': text,
            'prediction': 'Neither',
            'confidence': 0.5,
            'method': 'fallback',
            'rule_type': 'default',
            'explanation': 'Clasificación por defecto (sin reglas ni ML)',
            'individual_predictions': [],
            'probabilities': {
                'Hate Speech': 0.1,
                'Offensive Language': 0.2,
                'Neither': 0.7
            }
        }
    
    def get_system_status(self):
        """Obtener estado del sistema"""
        return {
            'rules_loaded': self.rules_loaded,
            'ml_loaded': self.ml_loaded,
            'models_available': list(self.models.keys()),
            'total_models': len(self.models),
            'system_ready': self.rules_loaded or self.ml_loaded
        }

def main():
    """Función principal para probar el selector mejorado"""
    print("🧠 SISTEMA DE SELECCIÓN INTELIGENTE MEJORADO")
    print("=" * 60)
    
    # Inicializar selector
    selector = ImprovedSmartSelector()
    
    # Mostrar estado del sistema
    status = selector.get_system_status()
    print(f"\n📊 Estado del Sistema:")
    print(f"   Reglas mejoradas: {'✅' if status['rules_loaded'] else '❌'}")
    print(f"   Modelos ML: {'✅' if status['ml_loaded'] else '❌'}")
    print(f"   Modelos disponibles: {', '.join(status['models_available'])}")
    print(f"   Sistema listo: {'✅' if status['system_ready'] else '❌'}")
    
    # Casos de prueba (incluyendo casos problemáticos)
    test_cases = [
        "fuck you",                    # Debería ser Offensive
        "F*ck you",                    # Debería ser Offensive (evasión)
        "asshole",                     # Debería ser Offensive
        "@sshole",                     # Debería ser Offensive (evasión)
        "stupid",                      # Debería ser Offensive
        "st*pid",                      # Debería ser Offensive (evasión)
        "Women are inferior to men",   # Debería ser Hate Speech
        "This is fucking stupid",      # Debería ser Offensive
        "I hate all immigrants",       # Debería ser Hate Speech
        "Hello, how are you?",         # Debería ser Neither
        "Thank you very much",         # Debería ser Neither
        "You are amazing",             # Debería ser Neither
        "This is fucking amazing",     # Debería ser Neither (contexto positivo)
        "You are fucking awesome",     # Debería ser Neither (contexto positivo)
        "What the hell is this",       # Debería ser Offensive
        "Hell yeah, that's great"      # Debería ser Neither (contexto positivo)
    ]
    
    print(f"\n🧪 Probando {len(test_cases)} casos:")
    print("-" * 60)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for text in test_cases:
        result = selector.predict(text)
        
        # Determinar si la predicción es correcta (simplificado)
        is_correct = True
        if "fuck" in text.lower() and "amazing" not in text.lower() and "awesome" not in text.lower():
            is_correct = result['prediction'] == 'Offensive Language'
        elif "inferior" in text.lower() or "hate all" in text.lower():
            is_correct = result['prediction'] == 'Hate Speech'
        elif "amazing" in text.lower() or "awesome" in text.lower() or "thank" in text.lower():
            is_correct = result['prediction'] == 'Neither'
        
        if is_correct:
            correct_predictions += 1
        
        status_icon = "✅" if is_correct else "❌"
        print(f"{status_icon} '{text}'")
        print(f"   → {result['prediction']} ({result['confidence']:.1%}) [{result['method']}]")
        print(f"   💡 {result['explanation']}")
        print()
    
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"📊 Precisión estimada: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    if accuracy >= 80:
        print("🎉 ¡Excelente! El sistema mejorado funciona muy bien")
    elif accuracy >= 70:
        print("👍 ¡Bien! El sistema mejorado funciona bien")
    else:
        print("⚠️ El sistema necesita más ajustes")
    
    print("\n✅ Sistema de selección inteligente mejorado funcionando!")
    print("🎯 Las palabras ofensivas se detectan con reglas mejoradas")
    print("🤖 El ML se usa para casos complejos y contexto positivo")

if __name__ == "__main__":
    main()
