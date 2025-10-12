#!/usr/bin/env python3
"""
Monitoreo de Data Drift
Nivel Experto - MLOps
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import kl_divergence  # No disponible en esta versión
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataDriftMonitor:
    """Sistema de monitoreo de Data Drift para modelos de ML"""
    
    def __init__(self, results_dir="backend/mlops/drift_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Configuración de monitoreo
        self.drift_threshold = 0.1  # Umbral para detectar drift
        self.alert_threshold = 0.2  # Umbral para alertas críticas
        self.min_samples = 50       # Mínimo de muestras para análisis
        
        # Referencia del dataset de entrenamiento
        self.reference_data = None
        self.reference_vectorizer = None
        self.reference_stats = None
        
    def set_reference_data(self, texts: List[str], vectorizer=None):
        """Establecer datos de referencia (dataset de entrenamiento)"""
        
        print("🔄 Configurando datos de referencia...")
        
        # Guardar textos de referencia
        self.reference_data = texts
        
        # Crear o usar vectorizador
        if vectorizer is None:
            self.reference_vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.95
            )
            reference_vectors = self.reference_vectorizer.fit_transform(texts)
        else:
            self.reference_vectorizer = vectorizer
            reference_vectors = vectorizer.transform(texts)
        
        # Calcular estadísticas de referencia
        self.reference_stats = self._calculate_reference_stats(reference_vectors, texts)
        
        # Guardar referencia
        self._save_reference_data()
        
        print(f"✅ Datos de referencia configurados: {len(texts)} textos")
        
    def _calculate_reference_stats(self, vectors, texts):
        """Calcular estadísticas de referencia"""
        
        # Estadísticas de vectores
        vector_stats = {
            'mean': np.mean(vectors.toarray(), axis=0),
            'std': np.std(vectors.toarray(), axis=0),
            'shape': vectors.shape,
            'sparsity': 1 - (vectors.nnz / (vectors.shape[0] * vectors.shape[1]))
        }
        
        # Estadísticas de texto
        text_stats = {
            'length_mean': np.mean([len(text) for text in texts]),
            'length_std': np.std([len(text) for text in texts]),
            'word_count_mean': np.mean([len(text.split()) for text in texts]),
            'word_count_std': np.std([len(text.split()) for text in texts]),
            'unique_words': len(set(' '.join(texts).split())),
            'total_words': len(' '.join(texts).split())
        }
        
        return {
            'vector_stats': vector_stats,
            'text_stats': text_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_reference_data(self):
        """Guardar datos de referencia"""
        
        reference_path = os.path.join(self.results_dir, "reference_data.pkl")
        
        reference_info = {
            'data': self.reference_data,
            'vectorizer': self.reference_vectorizer,
            'stats': self.reference_stats
        }
        
        with open(reference_path, 'wb') as f:
            pickle.dump(reference_info, f)
        
        print(f"✅ Datos de referencia guardados en: {reference_path}")
    
    def load_reference_data(self):
        """Cargar datos de referencia"""
        
        reference_path = os.path.join(self.results_dir, "reference_data.pkl")
        
        if not os.path.exists(reference_path):
            print("❌ No se encontraron datos de referencia")
            return False
        
        with open(reference_path, 'rb') as f:
            reference_info = pickle.load(f)
        
        self.reference_data = reference_info['data']
        self.reference_vectorizer = reference_info['vectorizer']
        self.reference_stats = reference_info['stats']
        
        print(f"✅ Datos de referencia cargados: {len(self.reference_data)} textos")
        return True
    
    def detect_drift(self, new_texts: List[str], window_name: str = None) -> Dict:
        """Detectar drift en nuevos datos"""
        
        if self.reference_stats is None:
            if not self.load_reference_data():
                return {"error": "No hay datos de referencia disponibles"}
        
        print(f"🔍 Analizando drift en {len(new_texts)} textos nuevos...")
        
        # Vectorizar nuevos datos
        new_vectors = self.reference_vectorizer.transform(new_texts)
        
        # Calcular estadísticas de nuevos datos
        new_stats = self._calculate_reference_stats(new_vectors, new_texts)
        
        # Comparar con referencia
        drift_results = self._compare_with_reference(new_stats)
        
        # Generar reporte
        drift_report = {
            'window_name': window_name or f"window_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'new_samples': len(new_texts),
            'reference_samples': len(self.reference_data),
            'drift_detected': drift_results['drift_detected'],
            'drift_severity': drift_results['drift_severity'],
            'drift_score': drift_results['drift_score'],
            'metrics': drift_results['metrics'],
            'alerts': drift_results['alerts']
        }
        
        # Guardar reporte
        self._save_drift_report(drift_report)
        
        return drift_report
    
    def _compare_with_reference(self, new_stats: Dict) -> Dict:
        """Comparar estadísticas nuevas con referencia"""
        
        ref_vector = self.reference_stats['vector_stats']
        new_vector = new_stats['vector_stats']
        ref_text = self.reference_stats['text_stats']
        new_text = new_stats['text_stats']
        
        metrics = {}
        alerts = []
        drift_scores = []
        
        # 1. Comparar estadísticas de vectores
        if ref_vector['shape'][1] == new_vector['shape'][1]:
            # KL Divergence en distribuciones de características
            kl_div = self._calculate_kl_divergence(
                ref_vector['mean'], new_vector['mean'],
                ref_vector['std'], new_vector['std']
            )
            metrics['kl_divergence'] = kl_div
            drift_scores.append(kl_div)
            
            if kl_div > self.alert_threshold:
                alerts.append(f"KL Divergence crítica: {kl_div:.3f}")
            elif kl_div > self.drift_threshold:
                alerts.append(f"KL Divergence alta: {kl_div:.3f}")
        
        # 2. Comparar estadísticas de texto
        length_diff = abs(ref_text['length_mean'] - new_text['length_mean']) / ref_text['length_mean']
        metrics['length_drift'] = length_diff
        drift_scores.append(length_diff)
        
        if length_diff > self.alert_threshold:
            alerts.append(f"Drift en longitud de texto: {length_diff:.3f}")
        elif length_diff > self.drift_threshold:
            alerts.append(f"Drift moderado en longitud: {length_diff:.3f}")
        
        word_count_diff = abs(ref_text['word_count_mean'] - new_text['word_count_mean']) / ref_text['word_count_mean']
        metrics['word_count_drift'] = word_count_diff
        drift_scores.append(word_count_diff)
        
        if word_count_diff > self.alert_threshold:
            alerts.append(f"Drift en conteo de palabras: {word_count_diff:.3f}")
        elif word_count_diff > self.drift_threshold:
            alerts.append(f"Drift moderado en palabras: {word_count_diff:.3f}")
        
        # 3. Comparar sparsity
        sparsity_diff = abs(ref_vector['sparsity'] - new_vector['sparsity'])
        metrics['sparsity_drift'] = sparsity_diff
        drift_scores.append(sparsity_diff)
        
        if sparsity_diff > self.alert_threshold:
            alerts.append(f"Drift en sparsity: {sparsity_diff:.3f}")
        elif sparsity_diff > self.drift_threshold:
            alerts.append(f"Drift moderado en sparsity: {sparsity_diff:.3f}")
        
        # 4. Test de Kolmogorov-Smirnov para distribuciones
        ks_stat, ks_pvalue = self._ks_test(ref_vector['mean'], new_vector['mean'])
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pvalue
        drift_scores.append(ks_stat)
        
        if ks_pvalue < 0.01:  # 99% confianza
            alerts.append(f"Distribución significativamente diferente (KS p-value: {ks_pvalue:.3f})")
        
        # Calcular score general de drift
        overall_drift_score = np.mean(drift_scores)
        
        # Determinar severidad
        if overall_drift_score > self.alert_threshold:
            drift_severity = "critical"
        elif overall_drift_score > self.drift_threshold:
            drift_severity = "moderate"
        else:
            drift_severity = "low"
        
        return {
            'drift_detected': overall_drift_score > self.drift_threshold,
            'drift_severity': drift_severity,
            'drift_score': overall_drift_score,
            'metrics': metrics,
            'alerts': alerts
        }
    
    def _calculate_kl_divergence(self, mean1, mean2, std1, std2):
        """Calcular KL Divergence entre dos distribuciones"""
        
        # Evitar división por cero
        std1 = np.maximum(std1, 1e-10)
        std2 = np.maximum(std2, 1e-10)
        
        # KL Divergence simplificada
        kl_div = 0.5 * np.sum(
            (std2**2 / std1**2) + 
            ((mean1 - mean2)**2 / std1**2) - 
            1 + 
            2 * np.log(std1 / std2)
        )
        
        return max(0, kl_div)  # KL Divergence no puede ser negativa
    
    def _ks_test(self, dist1, dist2):
        """Test de Kolmogorov-Smirnov entre dos distribuciones"""
        
        try:
            statistic, pvalue = stats.ks_2samp(dist1, dist2)
            return statistic, pvalue
        except:
            return 0, 1  # Si falla, asumir no hay diferencia
    
    def _save_drift_report(self, report: Dict):
        """Guardar reporte de drift"""
        
        # Convertir tipos numpy a Python nativos para JSON
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convertir el reporte
        report_serializable = convert_numpy_types(report)
        
        report_path = os.path.join(
            self.results_dir, 
            f"drift_report_{report['window_name']}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"✅ Reporte de drift guardado: {report_path}")
    
    def get_drift_history(self) -> List[Dict]:
        """Obtener historial de drift"""
        
        reports = []
        
        for filename in os.listdir(self.results_dir):
            if filename.startswith('drift_report_') and filename.endswith('.json'):
                with open(os.path.join(self.results_dir, filename), 'r') as f:
                    report = json.load(f)
                    reports.append(report)
        
        # Ordenar por timestamp
        reports.sort(key=lambda x: x['timestamp'])
        
        return reports
    
    def get_drift_summary(self) -> Dict:
        """Obtener resumen de drift"""
        
        history = self.get_drift_history()
        
        if not history:
            return {"error": "No hay reportes de drift disponibles"}
        
        # Calcular estadísticas
        drift_scores = [report['drift_score'] for report in history]
        critical_count = len([r for r in history if r['drift_severity'] == 'critical'])
        moderate_count = len([r for r in history if r['drift_severity'] == 'moderate'])
        low_count = len([r for r in history if r['drift_severity'] == 'low'])
        
        return {
            'total_reports': len(history),
            'average_drift_score': np.mean(drift_scores),
            'max_drift_score': np.max(drift_scores),
            'min_drift_score': np.min(drift_scores),
            'critical_alerts': critical_count,
            'moderate_alerts': moderate_count,
            'low_alerts': low_count,
            'last_report': history[-1] if history else None
        }

def test_drift_monitor():
    """Probar el sistema de monitoreo de drift"""
    
    print("🧪 PROBANDO SISTEMA DE MONITOREO DE DATA DRIFT")
    print("=" * 60)
    
    # Crear monitor
    monitor = DataDriftMonitor()
    
    # Simular datos de referencia (entrenamiento)
    reference_texts = [
        "fuck you", "hello world", "you are stupid", "amazing work",
        "hate speech", "brilliant idea", "you are a jerk", "wonderful job",
        "this is great", "that's terrible", "excellent work", "poor quality",
        "outstanding performance", "disappointing results", "fantastic job",
        "awful experience", "incredible talent", "mediocre work", "superb quality"
    ] * 5  # 100 textos de referencia
    
    print(f"📊 Configurando {len(reference_texts)} textos de referencia...")
    monitor.set_reference_data(reference_texts)
    
    # Simular datos nuevos (producción) - con drift
    new_texts_1 = [
        "fuck you", "hello world", "you are stupid", "amazing work",
        "hate speech", "brilliant idea", "you are a jerk", "wonderful job"
    ] * 3  # 24 textos similares (bajo drift)
    
    new_texts_2 = [
        "this is absolutely fantastic and wonderful",  # Textos más largos
        "that is completely terrible and disappointing",
        "the performance was outstanding and remarkable",
        "the quality is superb and excellent",
        "this work is incredible and amazing",
        "the results are fantastic and wonderful"
    ] * 4  # 24 textos con drift (longitud diferente)
    
    # Test 1: Bajo drift
    print("\n🔍 Test 1: Datos con bajo drift...")
    report1 = monitor.detect_drift(new_texts_1, "test_low_drift")
    
    print(f"   Drift detectado: {report1['drift_detected']}")
    print(f"   Severidad: {report1['drift_severity']}")
    print(f"   Score: {report1['drift_score']:.3f}")
    print(f"   Alertas: {len(report1['alerts'])}")
    
    # Test 2: Alto drift
    print("\n🔍 Test 2: Datos con alto drift...")
    report2 = monitor.detect_drift(new_texts_2, "test_high_drift")
    
    print(f"   Drift detectado: {report2['drift_detected']}")
    print(f"   Severidad: {report2['drift_severity']}")
    print(f"   Score: {report2['drift_score']:.3f}")
    print(f"   Alertas: {len(report2['alerts'])}")
    
    # Resumen
    print("\n📈 Resumen de drift:")
    summary = monitor.get_drift_summary()
    
    print(f"   Total reportes: {summary['total_reports']}")
    print(f"   Score promedio: {summary['average_drift_score']:.3f}")
    print(f"   Score máximo: {summary['max_drift_score']:.3f}")
    print(f"   Alertas críticas: {summary['critical_alerts']}")
    print(f"   Alertas moderadas: {summary['moderate_alerts']}")

if __name__ == "__main__":
    test_drift_monitor()
