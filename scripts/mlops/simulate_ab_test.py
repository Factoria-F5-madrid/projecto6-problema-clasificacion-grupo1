#!/usr/bin/env python3
"""
Simulador de A/B Test realista
"""

import sys
sys.path.append('backend')

from mlops.ab_testing import ABTestingSystem
from models.ultimate_hybrid_system import UltimateHybridSystem
from models.final_smart_selector import FinalSmartSelector
import random
import time

def simulate_realistic_ab_test():
    """Simular un A/B test realista con datos reales"""
    
    print("🚀 SIMULANDO A/B TEST REALISTA")
    print("=" * 50)
    
    # Inicializar sistemas
    ab_system = ABTestingSystem()
    ultimate_system = UltimateHybridSystem()
    final_system = FinalSmartSelector()
    
    # Iniciar A/B test
    test_id = ab_system.start_ab_test(
        "UltimateHybrid", "FinalSmartSelector",
        ultimate_system, final_system, test_duration_days=1
    )
    
    print(f"✅ Test iniciado: {test_id}")
    
    # Casos de prueba realistas
    test_cases = [
        # Casos ofensivos
        ("fuck you", "Offensive Language"),
        ("you are stupid", "Offensive Language"),
        ("you are a jerk", "Offensive Language"),
        ("that's ridiculous", "Offensive Language"),
        ("you're pathetic", "Offensive Language"),
        ("what a loser", "Offensive Language"),
        ("dumbass", "Offensive Language"),
        ("idiot", "Offensive Language"),
        ("moron", "Offensive Language"),
        ("bastard", "Offensive Language"),
        
        # Casos de hate speech
        ("Women are inferior to men", "Hate Speech"),
        ("All men are stupid", "Hate Speech"),
        ("I hate all women", "Hate Speech"),
        ("Black people are criminals", "Hate Speech"),
        ("Muslims are terrorists", "Hate Speech"),
        
        # Casos limpios
        ("Hello, how are you?", "Neither"),
        ("That's brilliant work", "Neither"),
        ("You are amazing", "Neither"),
        ("Great job!", "Neither"),
        ("Thank you very much", "Neither"),
        ("Have a nice day", "Neither"),
        ("Good morning", "Neither"),
        ("You're welcome", "Neither"),
        ("That's interesting", "Neither"),
        ("I appreciate it", "Neither"),
        
        # Casos ambiguos
        ("This is fucking amazing", "Neither"),  # Positivo con palabra ofensiva
        ("You're fucking brilliant", "Neither"), # Positivo con palabra ofensiva
        ("That's fucking incredible", "Neither"), # Positivo con palabra ofensiva
    ]
    
    print(f"\n📊 Simulando {len(test_cases)} predicciones...")
    
    # Simular predicciones
    for i, (text, actual_label) in enumerate(test_cases):
        # Asignar tráfico (50/50)
        variant = ab_system.assign_traffic(test_id, f"user_{i}")
        
        # Predecir con el modelo asignado
        start_time = time.time()
        
        if variant == 'A':
            result = ultimate_system.predict(text)
            prediction = result['prediction']
            confidence = result['confidence']
        else:
            result = final_system.predict(text)
            prediction = result['prediction']
            confidence = result['confidence']
        
        response_time = time.time() - start_time
        
        # Log predicción
        ab_system.log_prediction(
            test_id, variant, text, prediction, confidence,
            actual_label, response_time
        )
        
        # Mostrar progreso
        if (i + 1) % 5 == 0:
            print(f"   Procesadas {i + 1}/{len(test_cases)} predicciones...")
    
    print("✅ Simulación completada")
    
    # Obtener resultados
    print("\n📈 RESULTADOS DEL A/B TEST:")
    print("-" * 50)
    
    results = ab_system.get_test_results(test_id)
    
    # Métricas generales
    print(f"Test ID: {results['test_id']}")
    print(f"Total predicciones: {results['total_predictions']}")
    
    # Métricas del Modelo A
    model_a = results['model_a']
    print(f"\n🔵 Modelo A (UltimateHybrid):")
    print(f"   Predicciones: {model_a['total_predictions']}")
    print(f"   Con etiquetas: {model_a['labeled_predictions']}")
    print(f"   Accuracy: {model_a.get('accuracy', 0):.3f}")
    print(f"   Confianza promedio: {model_a.get('avg_confidence', 0):.3f}")
    print(f"   Tiempo promedio: {model_a.get('avg_response_time', 0):.3f}s")
    
    # Métricas del Modelo B
    model_b = results['model_b']
    print(f"\n🔴 Modelo B (FinalSmartSelector):")
    print(f"   Predicciones: {model_b['total_predictions']}")
    print(f"   Con etiquetas: {model_b['labeled_predictions']}")
    print(f"   Accuracy: {model_b.get('accuracy', 0):.3f}")
    print(f"   Confianza promedio: {model_b.get('avg_confidence', 0):.3f}")
    print(f"   Tiempo promedio: {model_b.get('avg_response_time', 0):.3f}s")
    
    # Significancia estadística
    significance = results['statistical_significance']
    print(f"\n📊 Significancia Estadística:")
    print(f"   Estado: {significance['status']}")
    
    if significance['status'] == 'sufficient_data':
        print(f"   Accuracy A: {significance['model_a_accuracy']:.3f}")
        print(f"   Accuracy B: {significance['model_b_accuracy']:.3f}")
        print(f"   Diferencia: {significance['difference']:.3f}")
        print(f"   Z-Score: {significance['z_score']:.3f}")
        print(f"   Significativo: {'✅ Sí' if significance['is_significant'] else '❌ No'}")
        print(f"   Ganador: {significance['winner']}")
    else:
        print(f"   {significance['message']}")
    
    # Recomendación
    recommendation = ab_system.get_recommendation(test_id)
    print(f"\n🎯 Recomendación:")
    print(f"   {recommendation['recommendation']}")
    print(f"   {recommendation['message']}")
    
    return test_id, results

if __name__ == "__main__":
    test_id, results = simulate_realistic_ab_test()
    
    print(f"\n✅ Test completado: {test_id}")
    print("📱 Puedes ver los resultados en Streamlit: http://localhost:8518")
