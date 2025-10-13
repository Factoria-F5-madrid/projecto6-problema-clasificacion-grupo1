#!/usr/bin/env python3
"""
Auto-reemplazo de Modelos
Nivel Experto - MLOps
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AutoModelReplacement:
    """Sistema de auto-reemplazo de modelos basado en métricas de rendimiento"""
    
    def __init__(self, models_dir="backend/models/saved", results_dir="backend/mlops/replacement_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Configuración de reemplazo
        self.performance_threshold = 0.05  # 5% mejora mínima requerida
        self.min_evaluations = 10          # Mínimo de evaluaciones antes de considerar reemplazo
        self.evaluation_window = 7         # Días de ventana para evaluación
        self.confidence_threshold = 0.95   # 95% confianza para reemplazo
        
        # Estado actual
        self.current_model = None
        self.candidate_models = []
        self.evaluation_history = []
        
    def register_model(self, model_name: str, model_path: str, model_type: str = "hybrid") -> bool:
        """Registrar un nuevo modelo candidato"""
        
        try:
            # Verificar que el modelo existe
            if not os.path.exists(model_path):
                print(f"❌ Modelo no encontrado: {model_path}")
                return False
            
            # Solo verificar que el archivo existe, no cargarlo
            model_info = {
                'name': model_name,
                'path': model_path,
                'type': model_type,
                'registered_at': datetime.now().isoformat(),
                'status': 'candidate',
                'evaluations': [],
                'performance_metrics': {}
            }
            
            self.candidate_models.append(model_info)
            self._save_candidate_models()
            
            print(f"✅ Modelo registrado: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error registrando modelo {model_name}: {e}")
            return False
    
    def set_current_model(self, model_name: str) -> bool:
        """Establecer el modelo actual en producción"""
        
        # Buscar el modelo
        model = next((m for m in self.candidate_models if m['name'] == model_name), None)
        
        if not model:
            print(f"❌ Modelo no encontrado: {model_name}")
            return False
        
        # Actualizar estado
        for m in self.candidate_models:
            m['status'] = 'candidate'
        
        model['status'] = 'production'
        self.current_model = model
        
        self._save_candidate_models()
        print(f"✅ Modelo actual establecido: {model_name}")
        return True
    
    def evaluate_model_performance(self, model_name: str, test_data: List[str], 
                                 true_labels: List[str], predictions: List[str]) -> Dict:
        """Evaluar el rendimiento de un modelo"""
        
        try:
            # Calcular métricas básicas
            accuracy = self._calculate_accuracy(true_labels, predictions)
            precision = self._calculate_precision(true_labels, predictions)
            recall = self._calculate_recall(true_labels, predictions)
            f1_score = self._calculate_f1_score(true_labels, predictions)
            
            # Calcular confianza promedio
            confidence = self._calculate_confidence(predictions)
            
            # Calcular tiempo de respuesta (simulado)
            response_time = self._calculate_response_time(len(test_data))
            
            evaluation = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'test_samples': len(test_data),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confidence': confidence,
                'response_time': response_time,
                'overall_score': self._calculate_overall_score(accuracy, precision, recall, f1_score, confidence, response_time)
            }
            
            # Guardar evaluación
            self.evaluation_history.append(evaluation)
            self._save_evaluation_history()
            
            # Actualizar métricas del modelo
            self._update_model_metrics(model_name, evaluation)
            
            print(f"✅ Evaluación completada para {model_name}: Score = {evaluation['overall_score']:.3f}")
            return evaluation
            
        except Exception as e:
            print(f"❌ Error evaluando modelo {model_name}: {e}")
            return {}
    
    def _calculate_accuracy(self, true_labels: List[str], predictions: List[str]) -> float:
        """Calcular accuracy"""
        if len(true_labels) != len(predictions):
            return 0.0
        
        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        return correct / len(true_labels) if true_labels else 0.0
    
    def _calculate_precision(self, true_labels: List[str], predictions: List[str]) -> float:
        """Calcular precision promedio"""
        try:
            from sklearn.metrics import precision_score
            return precision_score(true_labels, predictions, average='weighted', zero_division=0)
        except:
            return 0.0
    
    def _calculate_recall(self, true_labels: List[str], predictions: List[str]) -> float:
        """Calcular recall promedio"""
        try:
            from sklearn.metrics import recall_score
            return recall_score(true_labels, predictions, average='weighted', zero_division=0)
        except:
            return 0.0
    
    def _calculate_f1_score(self, true_labels: List[str], predictions: List[str]) -> float:
        """Calcular F1 score promedio"""
        try:
            from sklearn.metrics import f1_score
            return f1_score(true_labels, predictions, average='weighted', zero_division=0)
        except:
            return 0.0
    
    def _calculate_confidence(self, predictions: List[str]) -> float:
        """Calcular confianza promedio (simulada)"""
        # Simular confianza basada en la consistencia de las predicciones
        if not predictions:
            return 0.0
        
        # Contar predicciones por clase
        class_counts = {}
        for pred in predictions:
            class_counts[pred] = class_counts.get(pred, 0) + 1
        
        # Calcular entropía (menor entropía = mayor confianza)
        total = len(predictions)
        entropy = 0
        for count in class_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Convertir entropía a confianza (0-1)
        max_entropy = np.log2(len(class_counts)) if len(class_counts) > 1 else 1
        confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
        
        return max(0, min(1, confidence))
    
    def _calculate_response_time(self, sample_count: int) -> float:
        """Calcular tiempo de respuesta (simulado)"""
        # Simular tiempo de respuesta basado en el número de muestras
        base_time = 0.1  # 100ms base
        per_sample_time = 0.001  # 1ms por muestra
        return base_time + (sample_count * per_sample_time)
    
    def _calculate_overall_score(self, accuracy: float, precision: float, recall: float, 
                               f1_score: float, confidence: float, response_time: float) -> float:
        """Calcular score general del modelo"""
        
        # Ponderación de métricas
        weights = {
            'accuracy': 0.25,
            'precision': 0.20,
            'recall': 0.20,
            'f1_score': 0.20,
            'confidence': 0.10,
            'response_time': 0.05
        }
        
        # Normalizar tiempo de respuesta (menor es mejor)
        normalized_response_time = max(0, 1 - (response_time * 10))  # Penalizar si > 100ms
        
        score = (
            weights['accuracy'] * accuracy +
            weights['precision'] * precision +
            weights['recall'] * recall +
            weights['f1_score'] * f1_score +
            weights['confidence'] * confidence +
            weights['response_time'] * normalized_response_time
        )
        
        return score
    
    def _load_model(self, model_path: str):
        """Cargar modelo desde archivo"""
        try:
            import joblib
            model = joblib.load(model_path)
            print(f"✅ Modelo cargado desde: {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error cargando modelo {model_path}: {e}")
            return None
    
    def _make_predictions(self, model, test_data: List[str]) -> List[str]:
        """Hacer predicciones con el modelo cargado"""
        try:
            predictions = []
            
            for text in test_data:
                # Intentar diferentes métodos de predicción según el tipo de modelo
                if hasattr(model, 'predict'):
                    # Modelo scikit-learn estándar - arreglar formato 2D
                    try:
                        import numpy as np
                        # Asegurar que el texto esté en formato 2D
                        text_2d = np.array([text]).reshape(1, -1)
                        pred = model.predict(text_2d)[0]
                    except:
                        # Si falla, intentar con formato original
                        pred = model.predict([text])[0]
                elif hasattr(model, 'detect_hate_speech'):
                    # Sistema híbrido personalizado
                    result = model.detect_hate_speech(text)
                    pred = result.get('classification', 'neither')
                elif hasattr(model, 'predict_ensemble'):
                    # Sistema de ensemble
                    result = model.predict_ensemble(text)
                    pred = result.get('classification', 'neither')
                else:
                    # Fallback
                    pred = 'neither'
                
                predictions.append(pred)
            
            print(f"✅ Predicciones generadas: {len(predictions)}")
            return predictions
            
        except Exception as e:
            print(f"❌ Error generando predicciones: {e}")
            # Retornar predicciones por defecto en caso de error
            return ['neither'] * len(test_data)
    
    def _update_model_metrics(self, model_name: str, evaluation: Dict):
        """Actualizar métricas de un modelo"""
        
        print(f"🔍 Actualizando métricas para {model_name}")
        model = next((m for m in self.candidate_models if m['name'] == model_name), None)
        if not model:
            print(f"❌ Modelo {model_name} no encontrado en candidatos")
            return
        
        # Inicializar evaluations si no existe
        if 'evaluations' not in model:
            model['evaluations'] = []
        
        # Agregar evaluación
        model['evaluations'].append(evaluation)
        print(f"✅ Evaluación agregada. Total: {len(model['evaluations'])}")
        
        # Calcular métricas promedio
        if len(model['evaluations']) > 0:
            recent_evaluations = model['evaluations'][-self.min_evaluations:]
            
            model['performance_metrics'] = {
                'avg_accuracy': np.mean([e['accuracy'] for e in recent_evaluations]),
                'avg_precision': np.mean([e['precision'] for e in recent_evaluations]),
                'avg_recall': np.mean([e['recall'] for e in recent_evaluations]),
                'avg_f1_score': np.mean([e['f1_score'] for e in recent_evaluations]),
                'avg_confidence': np.mean([e['confidence'] for e in recent_evaluations]),
                'avg_response_time': np.mean([e['response_time'] for e in recent_evaluations]),
                'avg_overall_score': np.mean([e['overall_score'] for e in recent_evaluations]),
                'total_evaluations': len(model['evaluations']),
                'last_evaluation': evaluation['timestamp']
            }
    
    def check_for_replacement(self) -> Optional[Dict]:
        """Verificar si hay un modelo candidato que deba reemplazar al actual"""
        
        if not self.current_model:
            print("⚠️ No hay modelo actual establecido")
            return None
        
        current_metrics = self.current_model.get('performance_metrics', {})
        current_score = current_metrics.get('avg_overall_score', 0)
        
        print(f"🔍 Verificando reemplazo. Score actual: {current_score:.3f}")
        
        # Buscar mejor modelo candidato
        best_candidate = None
        best_improvement = 0
        
        for model in self.candidate_models:
            if model['status'] == 'production':
                continue
            
            metrics = model.get('performance_metrics', {})
            if not metrics or metrics.get('total_evaluations', 0) < self.min_evaluations:
                continue
            
            candidate_score = metrics.get('avg_overall_score', 0)
            improvement = candidate_score - current_score
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = model
        
        # Verificar si el mejor candidato supera el umbral
        if best_candidate and best_improvement >= self.performance_threshold:
            replacement_decision = {
                'should_replace': True,
                'current_model': self.current_model['name'],
                'candidate_model': best_candidate['name'],
                'improvement': best_improvement,
                'current_score': current_score,
                'candidate_score': best_candidate['performance_metrics']['avg_overall_score'],
                'confidence': self._calculate_replacement_confidence(best_candidate),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Reemplazo recomendado: {best_candidate['name']} (+{best_improvement:.3f})")
            return replacement_decision
        else:
            print(f"ℹ️ No se recomienda reemplazo. Mejor mejora: {best_improvement:.3f}")
            return None
    
    def _calculate_replacement_confidence(self, candidate_model: Dict) -> float:
        """Calcular confianza en la decisión de reemplazo"""
        
        metrics = candidate_model.get('performance_metrics', {})
        evaluations = len(candidate_model.get('evaluations', []))
        
        # Confianza basada en número de evaluaciones y consistencia
        evaluation_confidence = min(1.0, evaluations / (self.min_evaluations * 2))
        
        # Confianza basada en consistencia de scores
        scores = [e['overall_score'] for e in candidate_model.get('evaluations', [])]
        if len(scores) > 1:
            consistency = 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
        else:
            consistency = 0.5
        
        overall_confidence = (evaluation_confidence + consistency) / 2
        return max(0, min(1, overall_confidence))
    
    def execute_replacement(self, replacement_decision: Dict) -> bool:
        """Ejecutar el reemplazo del modelo"""
        
        try:
            old_model = self.current_model['name']
            new_model_name = replacement_decision['candidate_model']
            
            # Crear backup del modelo actual
            backup_path = os.path.join(self.results_dir, f"backup_{old_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            self._backup_model(self.current_model['path'], backup_path)
            
            # Cambiar modelo actual
            self.set_current_model(new_model_name)
            
            # Registrar el reemplazo
            replacement_record = {
                'timestamp': datetime.now().isoformat(),
                'old_model': old_model,
                'new_model': new_model_name,
                'improvement': replacement_decision['improvement'],
                'confidence': replacement_decision['confidence'],
                'backup_path': backup_path
            }
            
            self._save_replacement_record(replacement_record)
            
            print(f"✅ Reemplazo ejecutado: {old_model} → {new_model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error ejecutando reemplazo: {e}")
            return False
    
    def _backup_model(self, model_path: str, backup_path: str):
        """Crear backup de un modelo"""
        import shutil
        shutil.copy2(model_path, backup_path)
    
    def _save_candidate_models(self):
        """Guardar información de modelos candidatos"""
        models_path = os.path.join(self.results_dir, "candidate_models.json")
        with open(models_path, 'w') as f:
            json.dump(self.candidate_models, f, indent=2)
    
    def _load_candidate_models(self):
        """Cargar información de modelos candidatos"""
        models_path = os.path.join(self.results_dir, "candidate_models.json")
        if os.path.exists(models_path):
            with open(models_path, 'r') as f:
                self.candidate_models = json.load(f)
            
            # Encontrar modelo actual
            self.current_model = next((m for m in self.candidate_models if m['status'] == 'production'), None)
    
    def _save_evaluation_history(self):
        """Guardar historial de evaluaciones"""
        history_path = os.path.join(self.results_dir, "evaluation_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
    
    def _load_evaluation_history(self):
        """Cargar historial de evaluaciones"""
        history_path = os.path.join(self.results_dir, "evaluation_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.evaluation_history = json.load(f)
    
    def _save_replacement_record(self, record: Dict):
        """Guardar registro de reemplazo"""
        replacements_path = os.path.join(self.results_dir, "replacements.json")
        
        replacements = []
        if os.path.exists(replacements_path):
            with open(replacements_path, 'r') as f:
                replacements = json.load(f)
        
        replacements.append(record)
        
        with open(replacements_path, 'w') as f:
            json.dump(replacements, f, indent=2)
    
    def get_model_status(self) -> Dict:
        """Obtener estado actual de todos los modelos"""
        
        # No recargar desde archivo, usar el estado actual en memoria
        return {
            'current_model': self.current_model,
            'candidate_models': self.candidate_models,
            'total_evaluations': len(self.evaluation_history),
            'last_evaluation': self.evaluation_history[-1] if self.evaluation_history else None
        }
    
    def get_replacement_history(self) -> List[Dict]:
        """Obtener historial de reemplazos"""
        
        replacements_path = os.path.join(self.results_dir, "replacements.json")
        if os.path.exists(replacements_path):
            with open(replacements_path, 'r') as f:
                return json.load(f)
        return []

def test_auto_replacement():
    """Probar el sistema de auto-reemplazo"""
    
    print("🧪 PROBANDO SISTEMA DE AUTO-REEMPLAZO DE MODELOS")
    print("=" * 60)
    
    # Crear sistema
    replacement_system = AutoModelReplacement()
    
    # Simular datos de prueba
    test_texts = [
        "fuck you", "hello world", "you are stupid", "amazing work",
        "hate speech", "brilliant idea", "you are a jerk", "wonderful job",
        "this is great", "that's terrible", "excellent work", "poor quality"
    ] * 5  # 60 textos
    
    true_labels = ["offensive", "neither", "offensive", "neither"] * 15
    predictions_model_a = ["offensive", "neither", "offensive", "neither"] * 15
    predictions_model_b = ["offensive", "neither", "hate_speech", "neither"] * 15  # Mejor modelo
    
    # Registrar modelos
    print("\n📝 Registrando modelos...")
    replacement_system.register_model("Model_A", "backend/models/saved/balanced_model.pkl", "hybrid")
    replacement_system.register_model("Model_B", "backend/models/saved/improved_model.pkl", "hybrid")
    
    # Establecer modelo actual
    replacement_system.set_current_model("Model_A")
    
    # Evaluar modelos
    print("\n🔍 Evaluando modelos...")
    
    # Evaluar modelo actual (peor rendimiento)
    for i in range(12):  # 12 evaluaciones
        replacement_system.evaluate_model_performance(
            "Model_A", test_texts, true_labels, predictions_model_a
        )
    
    # Evaluar modelo candidato (mejor rendimiento)
    for i in range(12):  # 12 evaluaciones
        replacement_system.evaluate_model_performance(
            "Model_B", test_texts, true_labels, predictions_model_b
        )
    
    # Verificar reemplazo
    print("\n🔄 Verificando reemplazo...")
    replacement_decision = replacement_system.check_for_replacement()
    
    if replacement_decision and replacement_decision['should_replace']:
        print(f"✅ Reemplazo recomendado:")
        print(f"   Modelo actual: {replacement_decision['current_model']}")
        print(f"   Modelo candidato: {replacement_decision['candidate_model']}")
        print(f"   Mejora: {replacement_decision['improvement']:.3f}")
        print(f"   Confianza: {replacement_decision['confidence']:.3f}")
        
        # Ejecutar reemplazo
        if replacement_system.execute_replacement(replacement_decision):
            print("✅ Reemplazo ejecutado exitosamente")
        else:
            print("❌ Error ejecutando reemplazo")
    else:
        print("ℹ️ No se recomienda reemplazo")
    
    # Mostrar estado final
    print("\n📊 Estado final de modelos:")
    status = replacement_system.get_model_status()
    
    for model in status['candidate_models']:
        metrics = model.get('performance_metrics', {})
        print(f"   {model['name']} ({model['status']}): Score = {metrics.get('avg_overall_score', 0):.3f}")


if __name__ == "__main__":
    test_auto_replacement()
