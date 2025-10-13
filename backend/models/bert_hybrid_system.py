#!/usr/bin/env python3
"""
Sistema Híbrido con BERT
Opción 3: Usar modelos pre-entrenados
"""

import re
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class BERTHybridSystem:
    """Sistema híbrido que combina BERT con reglas y ML tradicional"""
    
    def __init__(self):
        self.class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        self.models = {}
        self.bert_available = False
        
        # Cargar modelos tradicionales
        self._load_traditional_models()
        
        # Intentar cargar BERT
        self._load_bert_model()
        
        # Configurar reglas
        self._setup_rules()
    
    def _load_traditional_models(self):
        """Cargar modelos ML tradicionales"""
        print("🔄 Cargando modelos tradicionales...")
        
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
                    print(f"✅ {name.capitalize()}: Cargado")
                else:
                    print(f"❌ {name.capitalize()}: No encontrado")
            except Exception as e:
                print(f"❌ {name.capitalize()}: Error - {e}")
        
        print(f"📊 Modelos tradicionales cargados: {len(self.models)}")
    
    def _load_bert_model(self):
        """Intentar cargar modelo BERT"""
        print("🔄 Intentando cargar BERT...")
        
        try:
            # Intentar importar transformers
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Cargar modelo BERT multilingüe
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-multilingual-cased',
                num_labels=3
            )
            
            # Configurar para inferencia
            self.bert_model.eval()
            self.bert_available = True
            
            print("✅ BERT cargado correctamente")
            
        except ImportError:
            print("❌ BERT no disponible - transformers no instalado")
            print("💡 Instala con: pip install transformers torch")
            self.bert_available = False
        except Exception as e:
            print(f"❌ Error cargando BERT: {e}")
            self.bert_available = False
    
    def _get_model_weight(self, model_name):
        """Obtener peso del modelo"""
        weights = {
            'expanded': 0.4,    # Modelo expandido (mejor)
            'improved': 0.3,    # Modelo mejorado
            'balanced': 0.2,    # Modelo balanceado
            'bert': 0.1         # BERT (peso bajo por ahora)
        }
        return weights.get(model_name, 0.1)
    
    def _setup_rules(self):
        """Configurar reglas para palabras ofensivas"""
        
        # Palabras ofensivas (expandidas)
        self.offensive_words = [
            # Inglés
            'fuck', 'shit', 'asshole', 'bastard', 'damn', 'hell',
            'stupid', 'idiot', 'moron', 'dumb', 'retard', 'crap',
            'bitch', 'whore', 'slut', 'fag', 'nigger', 'kike',
            'jerk', 'loser', 'pathetic', 'ridiculous', 'dumbass',
            'dimwit', 'nincompoop', 'dolt', 'asinine', 'numbskull',
            'dunderhead', 'blockhead', 'moronic', 'simpleton',
            'dunce', 'halfwit', 'idiotic', 'nitwit', 'twit', 'dope',
            # Español
            'jódete', 'mierda', 'cabrón', 'idiota', 'estúpido',
            'tonto', 'tonta', 'tontos', 'tontas', 'imbécil',
            'imbécil', 'imbéciles', 'gilipollas', 'capullo',
            'capullos', 'mamón', 'mamones', 'cabrón', 'cabrones',
            'hijo de puta', 'hijos de puta', 'puta', 'putas',
            'joder', 'jodido', 'jodida', 'jodidos', 'jodidas',
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
        
        # Frases positivas (para evitar falsos positivos)
        self.positive_phrases = [
            'thank you', 'good morning', 'amazing work', 'you are great',
            'brilliant', 'excellent', 'outstanding', 'fantastic',
            'wonderful', 'marvelous', 'incredible', 'awesome',
            'gracias', 'buenos días', 'trabajo increíble', 'eres genial',
            'brillante', 'excelente', 'fantástico', 'maravilloso'
        ]
    
    def _predict_with_bert(self, text):
        """Predecir usando BERT"""
        if not self.bert_available:
            return None
        
        try:
            import torch
            
            # Tokenizar
            inputs = self.bert_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Predecir
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'prediction': self.class_names[prediction],
                'confidence': confidence,
                'method': 'bert'
            }
            
        except Exception as e:
            print(f"❌ Error en BERT: {e}")
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
                # Vectorizar
                X = model_info['vectorizer'].transform([text])
                
                # Predecir
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
                print(f"❌ Error en {name}: {e}")
                continue
        
        if not predictions:
            return None
        
        # Combinar predicciones con pesos
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        # Votación ponderada
        vote_counts = {}
        for pred, weight in zip(predictions, weights):
            vote_counts[pred] = vote_counts.get(pred, 0) + weight
        
        final_prediction = max(vote_counts, key=vote_counts.get)
        
        return {
            'prediction': final_prediction,
            'confidence': weighted_confidence,
            'method': 'traditional_ensemble'
        }
    
    def _predict_with_rules(self, text):
        """Predecir usando reglas"""
        text_lower = text.lower()
        
        # Verificar palabras ofensivas
        for word in self.offensive_words:
            if word in text_lower:
                return {
                    'prediction': 'Offensive Language',
                    'confidence': 0.8,
                    'method': 'rules'
                }
        
        # Verificar patrones de hate speech
        for pattern in self.hate_patterns:
            if re.search(pattern, text_lower):
                return {
                    'prediction': 'Hate Speech',
                    'confidence': 0.9,
                    'method': 'rules'
                }
        
        # Verificar frases positivas
        for phrase in self.positive_phrases:
            if phrase in text_lower:
                return {
                    'prediction': 'Neither',
                    'confidence': 0.7,
                    'method': 'rules'
                }
        
        return None
    
    def predict(self, text):
        """Predecir usando sistema híbrido"""
        
        # 1. Reglas primero (más rápido)
        rules_result = self._predict_with_rules(text)
        if rules_result and rules_result['confidence'] > 0.7:
            return rules_result
        
        # 2. BERT si está disponible (más inteligente)
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
            'method': 'fallback'
        }

def test_bert_hybrid():
    """Probar el sistema híbrido BERT"""
    
    print("🧪 PROBANDO SISTEMA HÍBRIDO BERT")
    print("=" * 50)
    
    # Inicializar sistema
    system = BERTHybridSystem()
    
    # Casos de prueba
    test_cases = [
        "eres un zoquete",           # Palabra rara ofensiva
        "que bobo eres",             # Palabra rara ofensiva
        "sois unos pardillos",       # Palabra rara ofensiva
        "Hello, how are you?",       # Texto limpio
        "That's brilliant work",     # Texto positivo
        "fuck you",                  # Palabra ofensiva común
        "Women are inferior to men", # Hate speech
        "xyz123 abc456"              # Palabras sin sentido
    ]
    
    print("📊 Resultados:")
    print("-" * 50)
    
    for text in test_cases:
        result = system.predict(text)
        print(f"'{text}'")
        print(f"  → {result['prediction']} ({result['confidence']:.1%}) [{result['method']}]")
        print()

if __name__ == "__main__":
    test_bert_hybrid()
