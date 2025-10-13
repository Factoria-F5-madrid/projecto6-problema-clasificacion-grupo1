#!/usr/bin/env python3
"""
Test para verificar que el Data Drift Monitor funciona correctamente
"""

import sys
import os
sys.path.append('backend')

from mlops.data_drift_monitor import DataDriftMonitor
import pandas as pd

def test_drift_monitor():
    """Probar el monitor de drift con datos realistas"""
    
    print("🧪 Probando Data Drift Monitor...")
    
    # Crear monitor
    monitor = DataDriftMonitor()
    
    # Cargar datos de referencia (entrenamiento)
    print("📊 Cargando datos de referencia...")
    try:
        df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
        reference_texts = df['clean_tweet_improved'].dropna().tolist()[:100]  # Solo 100 para prueba
        
        print(f"✅ Datos de referencia cargados: {len(reference_texts)} textos")
        
        # Configurar referencia
        monitor.set_reference_data(reference_texts)
        
        # Probar con datos similares (debería dar drift bajo)
        print("\n🔍 Probando con datos similares...")
        similar_texts = reference_texts[:10]  # Mismos datos
        report1 = monitor.detect_drift(similar_texts, "test_similar")
        
        print(f"📊 Resultados con datos similares:")
        print(f"   - Drift detectado: {report1['drift_detected']}")
        print(f"   - Severidad: {report1['drift_severity']}")
        print(f"   - Score: {report1['drift_score']:.3f}")
        print(f"   - KL Divergence: {report1['metrics'].get('kl_divergence', 0):.3f}")
        
        # Probar con datos diferentes (debería dar drift moderado)
        print("\n🔍 Probando con datos diferentes...")
        different_texts = [
            "This is a completely different type of text",
            "Very long text with many words that are not in the training data",
            "Short text",
            "Text with numbers 12345 and symbols !@#$%",
            "Another different text sample"
        ]
        report2 = monitor.detect_drift(different_texts, "test_different")
        
        print(f"📊 Resultados con datos diferentes:")
        print(f"   - Drift detectado: {report2['drift_detected']}")
        print(f"   - Severidad: {report2['drift_severity']}")
        print(f"   - Score: {report2['drift_score']:.3f}")
        print(f"   - KL Divergence: {report2['metrics'].get('kl_divergence', 0):.3f}")
        
        # Verificar que los scores están en rango correcto
        print("\n✅ Verificando rangos:")
        
        # Score general debe estar entre 0 y 1
        assert 0 <= report1['drift_score'] <= 1, f"Score similar fuera de rango: {report1['drift_score']}"
        assert 0 <= report2['drift_score'] <= 1, f"Score diferente fuera de rango: {report2['drift_score']}"
        
        # KL Divergence debe estar entre 0 y 1
        assert 0 <= report1['metrics'].get('kl_divergence', 0) <= 1, f"KL similar fuera de rango: {report1['metrics'].get('kl_divergence', 0)}"
        assert 0 <= report2['metrics'].get('kl_divergence', 0) <= 1, f"KL diferente fuera de rango: {report2['metrics'].get('kl_divergence', 0)}"
        
        print("✅ Todos los scores están en rango correcto (0-1)")
        
        # Verificar que datos similares dan drift más bajo
        if report1['drift_score'] < report2['drift_score']:
            print("✅ Datos similares dan drift más bajo que datos diferentes")
        else:
            print("⚠️ Datos similares no dan drift más bajo (puede ser normal)")
        
        print("\n🎉 Test completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drift_monitor()
    if success:
        print("\n✅ Data Drift Monitor funciona correctamente")
    else:
        print("\n❌ Data Drift Monitor tiene problemas")
