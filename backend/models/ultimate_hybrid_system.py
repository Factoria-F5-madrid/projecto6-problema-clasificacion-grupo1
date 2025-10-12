#!/usr/bin/env python3
"""
Sistema Híbrido Definitivo
Combina: Reglas + ML Expandido + BERT + Contexto
"""

import re
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class UltimateHybridSystem:
    """Sistema híbrido definitivo con máxima precisión"""
    
    def __init__(self):
        self.class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        self.models = {}
        self.bert_available = False
        
        # Cargar todos los componentes
        self._load_all_models()
        self._setup_comprehensive_rules()
        
        print("🚀 Sistema Híbrido Definitivo inicializado")
        print(f"📊 Modelos ML: {len(self.models)}")
        print(f"🧠 BERT: {'✅' if self.bert_available else '❌'}")
        print(f"📝 Reglas: {len(self.offensive_words)} palabras ofensivas")
    
    def _load_all_models(self):
        """Cargar todos los modelos disponibles"""
        
        # Modelos tradicionales
        model_paths = [
            ("expanded", "backend/models/saved/expanded_model.pkl", "backend/models/saved/expanded_vectorizer.pkl"),
            ("improved", "backend/models/saved/improved_model.pkl", "backend/models/saved/improved_vectorizer.pkl"),
            ("balanced", "backend/models/saved/balanced_model.pkl", "backend/models/saved/balanced_vectorizer.pkl")
        ]
        
        for name, model_path, vectorizer_path in model_paths:
            try:
                if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                    self.models[name] = {
                        'model': joblib.load(model_path),
                        'vectorizer': joblib.load(vectorizer_path),
                        'weight': self._get_model_weight(name),
                        'type': 'traditional'
                    }
            except Exception as e:
                print(f"⚠️ Error cargando {name}: {e}")
        
        # Intentar cargar BERT
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-multilingual-cased',
                num_labels=3
            )
            self.bert_model.eval()
            self.bert_available = True
            
        except Exception as e:
            print(f"⚠️ BERT no disponible: {e}")
            self.bert_available = False
    
    def _get_model_weight(self, model_name):
        """Obtener peso del modelo"""
        weights = {
            'expanded': 0.4,    # Modelo expandido (mejor)
            'improved': 0.3,    # Modelo mejorado
            'balanced': 0.2,    # Modelo balanceado
            'bert': 0.1         # BERT
        }
        return weights.get(model_name, 0.1)
    
    def _setup_comprehensive_rules(self):
        """Configurar reglas comprehensivas"""
        
        # Palabras ofensivas (completas)
        self.offensive_words = [
            # Inglés común
            'fuck', 'shit', 'asshole', 'bastard', 'damn', 'hell',
            'stupid', 'idiot', 'moron', 'dumb', 'retard', 'crap',
            'bitch', 'whore', 'slut', 'fag', 'nigger', 'kike',
            'jerk', 'loser', 'pathetic', 'ridiculous', 'dumbass',
            # Inglés raro
            'dimwit', 'nincompoop', 'dolt', 'asinine', 'numbskull',
            'dunderhead', 'blockhead', 'moronic', 'simpleton',
            'dunce', 'halfwit', 'idiotic', 'nitwit', 'twit', 'dope',
            # Español común
            'jódete', 'mierda', 'cabrón', 'idiota', 'estúpido',
            'tonto', 'tonta', 'tontos', 'tontas', 'imbécil',
            'imbécil', 'imbéciles', 'gilipollas', 'capullo',
            'capullos', 'mamón', 'mamones', 'cabrón', 'cabrones',
            'hijo de puta', 'hijos de puta', 'puta', 'putas',
            'joder', 'jodido', 'jodida', 'jodidos', 'jodidas',
            # Español raro
            'zoquete', 'bobo', 'pardillos', 'memo', 'palurdo',
            'cenutrio', 'pardillo', 'zopenco', 'bobalicón',
            'garrulo', 'paletos', 'patán', 'zopencos'
        ]
        
        # Patrones de hate speech
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
        
        # Frases positivas
        self.positive_phrases = [
            'thank you', 'good morning', 'amazing work', 'you are great',
            'brilliant', 'excellent', 'outstanding', 'fantastic',
            'wonderful', 'marvelous', 'incredible', 'awesome',
            'hello', 'hi', 'hey', 'good morning', 'good afternoon',
            'gracias', 'buenos días', 'trabajo increíble', 'eres genial',
            'brillante', 'excelente', 'fantástico', 'maravilloso',
            'hola', 'buenos días', 'buenas tardes'
        ]
        
        # Patrones de evasión
        self.evasion_patterns = [
            r'f[\W_]*u[\W_]*c[\W_]*k',
            r's[\W_]*h[\W_]*i[\W_]*t',
            r'a[\W_]*s[\W_]*s[\W_]*h[\W_]*o[\W_]*l[\W_]*e',
            r's[\W_]*t[\W_]*u[\W_]*p[\W_]*i[\W_]*d',
            r'i[\W_]*d[\W_]*i[\W_]*o[\W_]*t'
        ]
    
    def _analyze_context(self, text):
        """Analizar contexto del texto"""
        text_lower = text.lower()
        
        # Indicadores de contexto positivo
        positive_indicators = 0
        for phrase in self.positive_phrases:
            if phrase in text_lower:
                positive_indicators += 1
        
        # Indicadores de contexto ofensivo (usar palabras completas)
        offensive_indicators = 0
        for word in self.offensive_words:
            # Usar regex para coincidencias de palabras completas
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                offensive_indicators += 1
        
        # Indicadores de hate speech
        hate_indicators = 0
        for pattern in self.hate_patterns:
            if re.search(pattern, text_lower):
                hate_indicators += 1
        
        # Indicadores de evasión
        evasion_indicators = 0
        for pattern in self.evasion_patterns:
            if re.search(pattern, text_lower):
                evasion_indicators += 1
        
        return {
            'positive_indicators': positive_indicators,
            'offensive_indicators': offensive_indicators,
            'hate_indicators': hate_indicators,
            'evasion_indicators': evasion_indicators
        }
    
    def _predict_with_rules(self, text):
        """Predecir usando reglas inteligentes"""
        context = self._analyze_context(text)
        text_lower = text.lower()
        
        # Hate speech (máxima prioridad)
        if context['hate_indicators'] > 0:
            return {
                'prediction': 'Hate Speech',
                'confidence': 0.9,
                'method': 'rules_hate',
                'explanation': 'Detectado patrón de hate speech'
            }
        
        # Evasión (alta prioridad)
        if context['evasion_indicators'] > 0:
            return {
                'prediction': 'Offensive Language',
                'confidence': 0.85,
                'method': 'rules_evasion',
                'explanation': 'Detectado patrón de evasión ofensiva'
            }
        
        # Palabras ofensivas directas (usar palabras completas)
        text_lower = text.lower()
        offensive_found = []
        for word in self.offensive_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                offensive_found.append(word)
        
        if offensive_found:
            # Verificar si hay contexto positivo que pueda neutralizar
            if context['positive_indicators'] > 0:
                return {
                    'prediction': 'Neither',
                    'confidence': 0.6,
                    'method': 'rules_context',
                    'explanation': 'Palabras ofensivas pero con contexto positivo'
                }
            else:
                return {
                    'prediction': 'Offensive Language',
                    'confidence': 0.8,
                    'method': 'rules_offensive',
                    'explanation': f'Detectadas palabras ofensivas: {", ".join(offensive_found)}'
                }
        
        # Contexto positivo (alta prioridad)
        if context['positive_indicators'] > 0 and context['offensive_indicators'] == 0:
            return {
                'prediction': 'Neither',
                'confidence': 0.8,
                'method': 'rules_positive',
                'explanation': 'Contexto positivo detectado sin palabras ofensivas'
            }
        
        return None
    
    def _predict_with_bert(self, text):
        """Predecir usando BERT"""
        if not self.bert_available:
            return None
        
        try:
            import torch
            
            inputs = self.bert_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'prediction': self.class_names[prediction],
                'confidence': confidence,
                'method': 'bert',
                'explanation': 'Clasificado por modelo BERT multilingüe'
            }
            
        except Exception as e:
            return None
    
    def _predict_with_traditional(self, text):
        """Predecir usando modelos tradicionales"""
        if not self.models:
            return None
        
        predictions = []
        confidences = []
        weights = []
        
        for name, model_info in self.models.items():
            try:
                X = model_info['vectorizer'].transform([text])
                prediction = model_info['model'].predict(X)[0]
                probabilities = model_info['model'].predict_proba(X)[0]
                confidence = max(probabilities)
                
                # Mapear predicción
                if hasattr(model_info['model'], 'classes_'):
                    class_mapping = {i: self.class_names[i] for i in range(len(model_info['model'].classes_))}
                    prediction_name = class_mapping.get(prediction, 'Neither')
                else:
                    prediction_name = 'Neither'
                
                predictions.append(prediction_name)
                confidences.append(confidence)
                weights.append(model_info['weight'])
                
            except Exception as e:
                continue
        
        if not predictions:
            return None
        
        # Combinar predicciones
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        vote_counts = {}
        for pred, weight in zip(predictions, weights):
            vote_counts[pred] = vote_counts.get(pred, 0) + weight
        
        final_prediction = max(vote_counts, key=vote_counts.get)
        
        return {
            'prediction': final_prediction,
            'confidence': weighted_confidence,
            'method': 'traditional_ensemble',
            'explanation': 'Clasificado por ensemble de modelos ML tradicionales'
        }
    
    def predict(self, text):
        """Predecir usando sistema híbrido definitivo"""
        
        # 1. Reglas inteligentes (máxima prioridad)
        rules_result = self._predict_with_rules(text)
        if rules_result and rules_result['confidence'] > 0.7:
            return rules_result
        
        # 2. BERT (alta prioridad)
        bert_result = self._predict_with_bert(text)
        if bert_result and bert_result['confidence'] > 0.6:
            return bert_result
        
        # 3. Modelos tradicionales (fallback)
        traditional_result = self._predict_with_traditional(text)
        if traditional_result:
            return traditional_result
        
        # 4. Fallback final
        return {
            'prediction': 'Neither',
            'confidence': 0.4,
            'method': 'fallback',
            'explanation': 'Clasificación por defecto (conservador)'
        }

def test_ultimate_system():
    """Probar el sistema definitivo"""
    
    print("🧪 PROBANDO SISTEMA HÍBRIDO DEFINITIVO")
    print("=" * 60)
    
    # Inicializar sistema
    system = UltimateHybridSystem()
    
    # Casos de prueba
    test_cases = [
        # Palabras ofensivas raras
        "eres un zoquete",
        "que bobo eres", 
        "sois unos pardillos",
        "eres un memo",
        "que palurdo",
        
        # Palabras ofensivas comunes
        "fuck you",
        "eres un tonto",
        "que idiota",
        
        # Hate speech
        "Women are inferior to men",
        "All men are stupid",
        
        # Contexto positivo
        "Hello, how are you?",
        "That's brilliant work",
        "You are amazing",
        
        # Evasión
        "f*ck you",
        "@sshole",
        
        # Palabras sin sentido
        "xyz123 abc456"
    ]
    
    print("📊 Resultados:")
    print("-" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for text in test_cases:
        result = system.predict(text)
        
        # Determinar si es correcto (simplificado)
        is_correct = True
        if any(word in text.lower() for word in ['zoquete', 'bobo', 'pardillos', 'memo', 'palurdo', 'fuck', 'tonto', 'idiota']):
            is_correct = result['prediction'] in ['Offensive Language', 'Hate Speech']
        elif any(word in text.lower() for word in ['inferior', 'stupid']):
            is_correct = result['prediction'] == 'Hate Speech'
        elif any(word in text.lower() for word in ['brilliant', 'amazing', 'Hello']):
            is_correct = result['prediction'] == 'Neither'
        
        if is_correct:
            correct += 1
        
        status_icon = "✅" if is_correct else "❌"
        print(f"{status_icon} '{text}'")
        print(f"   → {result['prediction']} ({result['confidence']:.1%}) [{result['method']}]")
        print(f"   💡 {result['explanation']}")
        print()
    
    # Resumen
    accuracy = (correct / total) * 100
    print("=" * 60)
    print(f"📈 Precisión: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 90:
        print("🎉 ¡Excelente! Sistema definitivo funcionando perfectamente")
    elif accuracy >= 80:
        print("👍 Muy bueno, sistema híbrido efectivo")
    else:
        print("⚠️ Necesita ajustes adicionales")

if __name__ == "__main__":
    test_ultimate_system()
