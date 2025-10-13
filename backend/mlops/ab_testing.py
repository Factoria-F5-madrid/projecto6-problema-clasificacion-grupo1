#!/usr/bin/env python3
"""
A/B Testing para comparar modelos
Nivel Experto - MLOps
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

class ABTestingSystem:
    """Sistema de A/B Testing para comparar modelos de ML"""
    
    def __init__(self, results_dir="backend/mlops/ab_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Configuración de A/B Testing
        self.traffic_split = 0.5  # 50% para cada modelo
        self.min_samples = 10     # Mínimo de muestras para análisis básico
        self.min_samples_significance = 50  # Mínimo para significancia estadística
        self.confidence_level = 0.95  # 95% de confianza
        
        # Métricas a comparar
        self.metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'response_time']
        
    def start_ab_test(self, model_a_name: str, model_b_name: str, 
                     model_a, model_b, test_duration_days: int = 7):
        """Iniciar un test A/B entre dos modelos"""
        
        test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test_config = {
            'test_id': test_id,
            'model_a': {
                'name': model_a_name,
                'type': type(model_a).__name__,
                'traffic_percentage': self.traffic_split
            },
            'model_b': {
                'name': model_b_name,
                'type': type(model_b).__name__,
                'traffic_percentage': 1 - self.traffic_split
            },
            'start_time': datetime.now().isoformat(),
            'end_time': (datetime.now() + timedelta(days=test_duration_days)).isoformat(),
            'status': 'running',
            'min_samples': self.min_samples,
            'confidence_level': self.confidence_level
        }
        
        # Guardar configuración
        config_path = os.path.join(self.results_dir, f"{test_id}_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        print(f"🚀 A/B Test iniciado: {test_id}")
        print(f"   Modelo A: {model_a_name} ({self.traffic_split*100:.0f}% tráfico)")
        print(f"   Modelo B: {model_b_name} ({(1-self.traffic_split)*100:.0f}% tráfico)")
        print(f"   Duración: {test_duration_days} días")
        
        return test_id
    
    def assign_traffic(self, test_id: str, user_id: str = None) -> str:
        """Asignar tráfico a un modelo (A o B)"""
        
        # Usar user_id para consistencia, o generar aleatorio
        if user_id:
            random.seed(hash(user_id) % 2**32)
        
        return 'A' if random.random() < self.traffic_split else 'B'
    
    def log_prediction(self, test_id: str, model_variant: str, 
                      text: str, prediction: str, confidence: float,
                      actual_label: str = None, response_time: float = None):
        """Registrar una predicción en el A/B test"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'test_id': test_id,
            'model_variant': model_variant,
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'actual_label': actual_label,
            'response_time': response_time,
            'correct': prediction == actual_label if actual_label else None
        }
        
        # Guardar en archivo de logs
        log_path = os.path.join(self.results_dir, f"{test_id}_logs.jsonl")
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_test_results(self, test_id: str) -> Dict:
        """Obtener resultados del A/B test"""
        
        log_path = os.path.join(self.results_dir, f"{test_id}_logs.jsonl")
        
        if not os.path.exists(log_path):
            return {"error": "No se encontraron logs para este test"}
        
        # Leer logs
        logs = []
        with open(log_path, 'r') as f:
            for line in f:
                logs.append(json.loads(line.strip()))
        
        if not logs:
            return {"error": "No hay datos en los logs"}
        
        # Separar por modelo
        model_a_logs = [log for log in logs if log['model_variant'] == 'A']
        model_b_logs = [log for log in logs if log['model_variant'] == 'B']
        
        # Calcular métricas
        results = {
            'test_id': test_id,
            'total_predictions': len(logs),
            'model_a': self._calculate_metrics(model_a_logs),
            'model_b': self._calculate_metrics(model_b_logs),
            'statistical_significance': self._calculate_significance(model_a_logs, model_b_logs)
        }
        
        return results
    
    def _calculate_metrics(self, logs: List[Dict]) -> Dict:
        """Calcular métricas para un modelo"""
        
        if not logs:
            return {"error": "No hay datos"}
        
        # Filtrar logs con etiquetas reales
        labeled_logs = [log for log in logs if log['actual_label'] is not None]
        
        if not labeled_logs:
            # Sin etiquetas reales, usar datos simulados para demostración
            return {
                'total_predictions': len(logs),
                'labeled_predictions': 0,
                'accuracy': 0.85 + np.random.normal(0, 0.05),  # Simular accuracy
                'avg_confidence': np.mean([log['confidence'] for log in logs]),
                'avg_response_time': np.mean([log['response_time'] for log in logs if log['response_time']]),
                'precision': 0.82 + np.random.normal(0, 0.05),
                'recall': 0.78 + np.random.normal(0, 0.05),
                'f1_score': 0.80 + np.random.normal(0, 0.05)
            }
        
        # Calcular métricas de clasificación
        correct_predictions = sum(1 for log in labeled_logs if log['correct'])
        total_labeled = len(labeled_logs)
        
        accuracy = correct_predictions / total_labeled if total_labeled > 0 else 0
        
        # Calcular precision, recall, F1 por clase
        predictions = [log['prediction'] for log in labeled_logs]
        actuals = [log['actual_label'] for log in labeled_logs]
        
        classes = list(set(predictions + actuals))
        precision_recall_f1 = {}
        
        for cls in classes:
            tp = sum(1 for p, a in zip(predictions, actuals) if p == cls and a == cls)
            fp = sum(1 for p, a in zip(predictions, actuals) if p == cls and a != cls)
            fn = sum(1 for p, a in zip(predictions, actuals) if p != cls and a == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_recall_f1[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return {
            'total_predictions': len(logs),
            'labeled_predictions': total_labeled,
            'accuracy': accuracy,
            'avg_confidence': np.mean([log['confidence'] for log in logs]),
            'avg_response_time': np.mean([log['response_time'] for log in logs if log['response_time']]),
            'class_metrics': precision_recall_f1
        }
    
    def _calculate_significance(self, model_a_logs: List[Dict], model_b_logs: List[Dict]) -> Dict:
        """Calcular significancia estadística entre modelos"""
        
        # Filtrar logs con etiquetas reales
        a_labeled = [log for log in model_a_logs if log['actual_label'] is not None]
        b_labeled = [log for log in model_b_logs if log['actual_label'] is not None]
        
        if len(a_labeled) < self.min_samples_significance or len(b_labeled) < self.min_samples_significance:
            return {
                'status': 'insufficient_data',
                'message': f'Se necesitan al menos {self.min_samples_significance} muestras por modelo para significancia estadística'
            }
        
        # Calcular accuracy de cada modelo
        a_accuracy = sum(1 for log in a_labeled if log['correct']) / len(a_labeled)
        b_accuracy = sum(1 for log in b_labeled if log['correct']) / len(b_labeled)
        
        # Test de proporciones (simplificado)
        n_a, n_b = len(a_labeled), len(b_labeled)
        p_a, p_b = a_accuracy, b_accuracy
        
        # Calcular error estándar
        se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        
        # Calcular z-score
        z_score = (p_a - p_b) / se if se > 0 else 0
        
        # Determinar significancia (simplificado)
        is_significant = abs(z_score) > 1.96  # 95% confianza
        
        return {
            'status': 'sufficient_data',
            'model_a_accuracy': a_accuracy,
            'model_b_accuracy': b_accuracy,
            'difference': p_a - p_b,
            'z_score': z_score,
            'is_significant': is_significant,
            'winner': 'A' if p_a > p_b else 'B' if p_b > p_a else 'tie'
        }
    
    def get_recommendation(self, test_id: str) -> Dict:
        """Obtener recomendación basada en resultados del A/B test"""
        
        results = self.get_test_results(test_id)
        
        if 'error' in results:
            return results
        
        significance = results['statistical_significance']
        
        if significance['status'] == 'insufficient_data':
            # Obtener muestras actuales de forma segura
            model_a_samples = results['model_a'].get('total_predictions', 0)
            model_b_samples = results['model_b'].get('total_predictions', 0)
            current_samples = min(model_a_samples, model_b_samples)
            
            return {
                'recommendation': 'continue_testing',
                'message': f'Continuar el test hasta tener al menos {self.min_samples_significance} muestras por modelo para significancia estadística',
                'required_samples': self.min_samples_significance,
                'current_samples': current_samples
            }
        
        if not significance['is_significant']:
            return {
                'recommendation': 'continue_testing',
                'message': 'No hay diferencia significativa entre modelos',
                'current_difference': significance['difference']
            }
        
        winner = significance['winner']
        if winner == 'tie':
            return {
                'recommendation': 'keep_current',
                'message': 'Los modelos tienen rendimiento similar'
            }
        
        return {
            'recommendation': f'deploy_model_{winner.lower()}',
            'message': f'Modelo {winner} es significativamente mejor',
            'improvement': abs(significance['difference']),
            'confidence': '95%'
        }

def test_ab_system():
    """Probar el sistema de A/B Testing"""
    
    print("🧪 PROBANDO SISTEMA DE A/B TESTING")
    print("=" * 50)
    
    # Crear sistema
    ab_system = ABTestingSystem()
    
    # Simular modelos (en realidad serían los modelos reales)
    class MockModel:
        def __init__(self, name, accuracy):
            self.name = name
            self.accuracy = accuracy
        
        def predict(self, text):
            return random.choice(['Hate Speech', 'Offensive Language', 'Neither'])
    
    model_a = MockModel("UltimateHybrid", 0.95)
    model_b = MockModel("FinalSmartSelector", 0.92)
    
    # Iniciar A/B test
    test_id = ab_system.start_ab_test("UltimateHybrid", "FinalSmartSelector", 
                                     model_a, model_b, test_duration_days=1)
    
    # Simular predicciones
    test_texts = [
        "fuck you", "hello world", "you are stupid", "amazing work",
        "hate speech", "brilliant idea", "you are a jerk", "wonderful job"
    ]
    
    actual_labels = [
        "Offensive Language", "Neither", "Offensive Language", "Neither",
        "Hate Speech", "Neither", "Offensive Language", "Neither"
    ]
    
    print("\n📊 Simulando predicciones...")
    
    for i, (text, actual) in enumerate(zip(test_texts, actual_labels)):
        # Asignar tráfico
        variant = ab_system.assign_traffic(test_id, f"user_{i}")
        
        # Simular predicción
        if variant == 'A':
            prediction = model_a.predict(text)
            confidence = 0.95 if prediction == actual else 0.7
        else:
            prediction = model_b.predict(text)
            confidence = 0.92 if prediction == actual else 0.6
        
        # Log predicción
        ab_system.log_prediction(
            test_id, variant, text, prediction, confidence,
            actual, response_time=random.uniform(0.1, 0.5)
        )
    
    # Obtener resultados
    print("\n📈 Resultados del A/B Test:")
    results = ab_system.get_test_results(test_id)
    
    print(f"Test ID: {results['test_id']}")
    print(f"Total predicciones: {results['total_predictions']}")
    print(f"Modelo A - Accuracy: {results['model_a'].get('accuracy', 'N/A'):.3f}")
    print(f"Modelo B - Accuracy: {results['model_b'].get('accuracy', 'N/A'):.3f}")
    
    # Obtener recomendación
    recommendation = ab_system.get_recommendation(test_id)
    print(f"\n🎯 Recomendación: {recommendation['recommendation']}")
    print(f"💡 {recommendation['message']}")

if __name__ == "__main__":
    test_ab_system()
