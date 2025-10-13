#!/usr/bin/env python3
"""
Sistema híbrido avanzado que combina múltiples modelos
Implementa A/B testing y comparación de modelos para nivel experto
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

class AdvancedHybridSystem:
    """Sistema híbrido avanzado con múltiples modelos"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        self.label_mapping = {'Hate Speech': 0, 'Offensive Language': 1, 'Neither': 2}
        
    def load_models(self):
        """Cargar todos los modelos disponibles"""
        print("🔄 Cargando modelos disponibles...")
        
        # Cargar XGBoost mejorado
        try:
            self.models['xgboost'] = {
                'model': joblib.load("backend/models/saved/improved_model.pkl"),
                'vectorizer': joblib.load("backend/models/saved/improved_vectorizer.pkl"),
                'type': 'xgboost',
                'weight': 0.6
            }
            print("✅ XGBoost mejorado cargado")
        except Exception as e:
            print(f"❌ Error cargando XGBoost: {e}")
        
        # Cargar modelo balanceado
        try:
            self.models['balanced'] = {
                'model': joblib.load("backend/models/saved/balanced_model.pkl"),
                'vectorizer': joblib.load("backend/models/saved/balanced_vectorizer.pkl"),
                'type': 'logistic',
                'weight': 0.3
            }
            print("✅ Modelo balanceado cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo balanceado: {e}")
        
        # Cargar modelo original
        try:
            self.models['original'] = {
                'model': joblib.load("backend/models/saved/model.pkl"),
                'vectorizer': joblib.load("backend/models/saved/vectorizer.pkl"),
                'type': 'logistic',
                'weight': 0.1
            }
            print("✅ Modelo original cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo original: {e}")
        
        print(f"📊 Total de modelos cargados: {len(self.models)}")
    
    def predict_single_model(self, text, model_name):
        """Hacer predicción con un modelo específico"""
        if model_name not in self.models:
            return None
        
        model_data = self.models[model_name]
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        
        try:
            # Vectorizar texto
            X = vectorizer.transform([text])
            
            # Hacer predicción
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            return {
                'model': model_name,
                'prediction': self.class_mapping[prediction],
                'confidence': float(max(probabilities)),
                'probabilities': {
                    self.class_mapping[j]: float(prob) 
                    for j, prob in enumerate(probabilities)
                }
            }
        except Exception as e:
            print(f"❌ Error en predicción con {model_name}: {e}")
            return None
    
    def predict_ensemble(self, text):
        """Predicción con ensemble de todos los modelos"""
        predictions = []
        
        # Obtener predicciones de todos los modelos
        for model_name in self.models:
            pred = self.predict_single_model(text, model_name)
            if pred:
                predictions.append(pred)
        
        if not predictions:
            return None
        
        # Calcular predicción ponderada
        weighted_probs = {self.class_mapping[i]: 0.0 for i in range(3)}
        total_weight = 0
        
        for pred in predictions:
            model_name = pred['model']
            weight = self.models[model_name]['weight']
            total_weight += weight
            
            for class_name, prob in pred['probabilities'].items():
                weighted_probs[class_name] += prob * weight
        
        # Normalizar probabilidades
        for class_name in weighted_probs:
            weighted_probs[class_name] /= total_weight
        
        # Obtener predicción final
        final_prediction = max(weighted_probs, key=weighted_probs.get)
        final_confidence = weighted_probs[final_prediction]
        
        return {
            'text': text,
            'prediction': final_prediction,
            'confidence': final_confidence,
            'probabilities': weighted_probs,
            'individual_predictions': predictions,
            'method': 'ensemble_weighted'
        }
    
    def compare_models(self, texts):
        """Comparar todos los modelos en casos de prueba"""
        print("🔄 COMPARACIÓN DE MODELOS - A/B TESTING")
        print("=" * 60)
        
        results = []
        
        for text in texts:
            print(f"\n📝 Texto: '{text}'")
            print("-" * 50)
            
            text_results = {'text': text, 'predictions': {}}
            
            # Predicción con ensemble
            ensemble_pred = self.predict_ensemble(text)
            if ensemble_pred:
                text_results['ensemble'] = ensemble_pred
                print(f"🎯 Ensemble: {ensemble_pred['prediction']} ({ensemble_pred['confidence']:.1%})")
            
            # Predicciones individuales
            for model_name in self.models:
                pred = self.predict_single_model(text, model_name)
                if pred:
                    text_results['predictions'][model_name] = pred
                    print(f"   {model_name.capitalize()}: {pred['prediction']} ({pred['confidence']:.1%})")
            
            results.append(text_results)
        
        return results
    
    def evaluate_models(self, test_data):
        """Evaluar rendimiento de todos los modelos"""
        print("📊 EVALUACIÓN DE MODELOS")
        print("=" * 40)
        
        # Preparar datos de prueba
        if isinstance(test_data, str):
            df = pd.read_csv(test_data)
        else:
            df = test_data
        
        df_clean = df.dropna(subset=['clean_tweet_improved', 'class'])
        df_clean = df_clean[df_clean['clean_tweet_improved'].str.strip() != '']
        df_clean['label'] = df_clean['class'].map(self.label_mapping)
        
        X_test = df_clean['clean_tweet_improved'].values
        y_test = df_clean['label'].values
        
        print(f"📈 Datos de prueba: {len(X_test)} ejemplos")
        
        # Evaluar cada modelo
        model_scores = {}
        
        for model_name in self.models:
            print(f"\n🔍 Evaluando {model_name}...")
            
            model_data = self.models[model_name]
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            
            try:
                # Vectorizar datos de prueba
                X_test_vec = vectorizer.transform(X_test)
                
                # Hacer predicciones
                y_pred = model.predict(X_test_vec)
                
                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='macro', zero_division=0
                )
                
                model_scores[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                print(f"   Accuracy: {accuracy:.3f}")
                print(f"   Precision: {precision:.3f}")
                print(f"   Recall: {recall:.3f}")
                print(f"   F1-Score: {f1:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluando {model_name}: {e}")
        
        return model_scores
    
    def get_model_recommendations(self, text):
        """Obtener recomendaciones de qué modelo usar"""
        predictions = []
        
        for model_name in self.models:
            pred = self.predict_single_model(text, model_name)
            if pred:
                predictions.append((model_name, pred))
        
        if not predictions:
            return "No hay modelos disponibles"
        
        # Ordenar por confianza
        predictions.sort(key=lambda x: x[1]['confidence'], reverse=True)
        
        best_model = predictions[0]
        recommendations = {
            'best_model': best_model[0],
            'best_prediction': best_model[1]['prediction'],
            'best_confidence': best_model[1]['confidence'],
            'all_predictions': predictions,
            'agreement': len(set(p[1]['prediction'] for p in predictions)) == 1
        }
        
        return recommendations

def main():
    """Función principal para probar el sistema híbrido avanzado"""
    print("🚀 SISTEMA HÍBRIDO AVANZADO - NIVEL EXPERTO")
    print("=" * 60)
    
    # Inicializar sistema
    system = AdvancedHybridSystem()
    
    # Cargar modelos
    system.load_models()
    
    if not system.models:
        print("❌ No se pudieron cargar modelos")
        return
    
    # Casos de prueba
    test_cases = [
        "fuck you",
        "F*ck you", 
        "asshole",
        "@sshole",
        "stupid",
        "st*pid",
        "Women are inferior to men",
        "This is fucking stupid",
        "I hate all immigrants",
        "Hello, how are you?",
        "Thank you very much",
        "You are amazing"
    ]
    
    # Comparar modelos
    results = system.compare_models(test_cases)
    
    # Evaluar con datos reales
    print("\n" + "="*60)
    model_scores = system.evaluate_models("backend/data/processed/cleaned_tweets.csv")
    
    # Mostrar recomendaciones
    print("\n🎯 RECOMENDACIONES DE MODELOS:")
    print("-" * 40)
    
    for text in test_cases[:5]:  # Solo primeros 5 para no saturar
        rec = system.get_model_recommendations(text)
        print(f"\n📝 '{text}'")
        print(f"   Mejor modelo: {rec['best_model']}")
        print(f"   Predicción: {rec['best_prediction']} ({rec['best_confidence']:.1%})")
        print(f"   Acuerdo entre modelos: {'✅' if rec['agreement'] else '❌'}")
    
    print("\n🎉 SISTEMA HÍBRIDO AVANZADO COMPLETADO")
    print("🚀 Nivel experto alcanzado con A/B testing y ensemble")

if __name__ == "__main__":
    main()
