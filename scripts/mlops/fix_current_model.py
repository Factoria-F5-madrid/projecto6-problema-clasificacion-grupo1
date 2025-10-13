#!/usr/bin/env python3
"""
Script para arreglar el modelo actual con métricas correctas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from mlops.auto_model_replacement import AutoModelReplacement

def fix_current_model():
    """Arreglar el modelo actual con métricas correctas"""
    print("🔧 Arreglando modelo actual...")
    
    replacement_system = AutoModelReplacement()
    
    # Ver estado actual
    print(f"📊 Estado actual:")
    print(f"   - Modelos candidatos: {len(replacement_system.candidate_models)}")
    print(f"   - Modelo actual: {replacement_system.current_model['name'] if replacement_system.current_model else 'None'}")
    
    # Buscar Model_A con métricas
    model_a = None
    for model in replacement_system.candidate_models:
        if model['name'] == 'Model_A' and model.get('performance_metrics', {}).get('avg_overall_score', 0) > 0:
            model_a = model
            break
    
    if model_a:
        print(f"✅ Encontrado Model_A con métricas: {model_a['performance_metrics']['avg_overall_score']:.3f}")
        
        # Establecer como modelo actual
        replacement_system.current_model = model_a
        replacement_system._save_state()
        
        print(f"✅ Model_A establecido como modelo actual")
        print(f"   - Score: {model_a['performance_metrics']['avg_overall_score']:.3f}")
        print(f"   - Precision: {model_a['performance_metrics']['avg_precision']:.3f}")
        print(f"   - Recall: {model_a['performance_metrics']['avg_recall']:.3f}")
    else:
        print("❌ No se encontró Model_A con métricas válidas")
    
    # Verificar Model_B
    model_b = None
    for model in replacement_system.candidate_models:
        if model['name'] == 'Model_B' and model.get('performance_metrics', {}).get('avg_overall_score', 0) > 0:
            model_b = model
            break
    
    if model_b:
        print(f"✅ Encontrado Model_B con métricas: {model_b['performance_metrics']['avg_overall_score']:.3f}")
    else:
        print("❌ No se encontró Model_B con métricas válidas")
    
    # Probar verificación de reemplazo
    print(f"\n🔄 Probando verificación de reemplazo...")
    recommendation = replacement_system.check_for_replacement()
    
    print(f"📊 Resultado:")
    print(f"   - Debería reemplazar: {recommendation['should_replace']}")
    
    if 'reason' in recommendation:
        print(f"   - Razón: {recommendation['reason']}")
    elif 'candidate_model' in recommendation:
        print(f"   - Modelo candidato: {recommendation['candidate_model']}")
        print(f"   - Mejora: {recommendation['improvement']:.3f}")
    
    if 'improvement' in recommendation:
        print(f"   - Mejora: {recommendation['improvement']:.3f}")
    
    print(f"\n✅ Proceso completado")

if __name__ == "__main__":
    fix_current_model()
