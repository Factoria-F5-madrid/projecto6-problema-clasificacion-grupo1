#!/usr/bin/env python3
"""
Prueba rápida del A/B Testing en vivo
"""

import sys
sys.path.append('backend')

from mlops.ab_testing import ABTestingSystem
from models.ultimate_hybrid_system import UltimateHybridSystem
from models.final_smart_selector import FinalSmartSelector
import time

def test_ab_live():
    """Probar A/B Testing en vivo"""
    
    print("🧪 PROBANDO A/B TESTING EN VIVO")
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
    
    # Casos de prueba
    test_cases = [
        "fuck you",
        "hello world", 
        "you are stupid",
        "amazing work",
        "hate speech"
    ]
    
    print(f"\n📊 Simulando {len(test_cases)} predicciones...")
    
    for i, text in enumerate(test_cases):
        # Asignar tráfico
        variant = ab_system.assign_traffic(test_id, f"user_{i}")
        
        # Obtener modelo
        if variant == 'A':
            model = ultimate_system
            model_name = "UltimateHybrid"
        else:
            model = final_system
            model_name = "FinalSmartSelector"
        
        # Hacer predicción
        start_time = time.time()
        result = model.predict(text)
        response_time = time.time() - start_time
        
        # Log predicción
        ab_system.log_prediction(
            test_id, variant, text, 
            result['prediction'], result['confidence'],
            None, response_time
        )
        
        print(f"   {i+1}. '{text}' → {result['prediction']} ({variant})")
    
    print("✅ Predicciones registradas")
    
    # Obtener resultados
    print("\n📈 RESULTADOS:")
    print("-" * 30)
    
    results = ab_system.get_test_results(test_id)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"Test ID: {results['test_id']}")
        print(f"Total predicciones: {results['total_predictions']}")
        
        model_a = results['model_a']
        model_b = results['model_b']
        
        print(f"\nModelo A: {model_a.get('total_predictions', 0)} predicciones")
        print(f"Modelo B: {model_b.get('total_predictions', 0)} predicciones")
        
        # Verificar archivos
        import os
        log_path = f"backend/mlops/ab_results/{test_id}_logs.jsonl"
        
        if os.path.exists(log_path):
            print(f"✅ Logs guardados en: {log_path}")
            
            # Leer logs
            with open(log_path, 'r') as f:
                logs = [line.strip() for line in f]
            
            print(f"📄 Líneas en el archivo: {len(logs)}")
            
            # Mostrar primera línea como ejemplo
            if logs:
                import json
                first_log = json.loads(logs[0])
                print(f"📝 Ejemplo de log: {first_log}")
        else:
            print(f"❌ No se encontró el archivo de logs: {log_path}")
    
    print(f"\n🎯 Test ID para Streamlit: {test_id}")
    print("📱 Ve a http://localhost:8518 → A/B Testing → Ver Resultados")

if __name__ == "__main__":
    test_ab_live()
