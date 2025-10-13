#!/usr/bin/env python3
"""
Script de diagnóstico para el sistema de Auto-reemplazo de Modelos
FIX: Métricas de modelos en 0.0
"""

import os
import sys
import json
from datetime import datetime

# Añadir el path del backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from mlops.auto_model_replacement import AutoModelReplacement

def diagnose_models():
    """Diagnosticar el estado de los modelos"""
    print("🔍 DIAGNÓSTICO DEL SISTEMA DE AUTO-REEMPLAZO")
    print("=" * 60)
    
    replacement_system = AutoModelReplacement()
    
    # Ver estado actual
    print("\n📊 Estado cargado:")
    print(f"   - Modelos candidatos: {len(replacement_system.candidate_models)}")
    print(f"   - Evaluaciones: {len(replacement_system.evaluation_history)}")
    print(f"   - Modelo actual: {replacement_system.current_model['name'] if replacement_system.current_model else 'None'}")
    
    # Verificar cada modelo
    print("\n🔍 ANÁLISIS DE MODELOS:")
    for i, model in enumerate(replacement_system.candidate_models):
        print(f"\n📋 Modelo {i+1}: {model['name']}")
        print(f"   - Estado: {model['status']}")
        print(f"   - Tipo: {model['type']}")
        print(f"   - Path: {model['model_path']}")
        print(f"   - Existe archivo: {os.path.exists(model['model_path'])}")
        
        metrics = model.get('performance_metrics', {})
        print(f"   - Métricas:")
        print(f"     • Score: {metrics.get('avg_overall_score', 0):.3f}")
        print(f"     • Accuracy: {metrics.get('avg_accuracy', 0):.3f}")
        print(f"     • Precision: {metrics.get('avg_precision', 0):.3f}")
        print(f"     • Recall: {metrics.get('avg_recall', 0):.3f}")
        print(f"     • F1: {metrics.get('avg_f1_score', 0):.3f}")
        print(f"     • Evaluaciones: {metrics.get('total_evaluations', 0)}")
        
        evals = model.get('evaluations', [])
        print(f"   - Historial de evaluaciones: {len(evals)}")
        if evals:
            print(f"     • Última evaluación: {evals[-1].get('timestamp', 'N/A')}")
            print(f"     • Último score: {evals[-1].get('overall_score', 0):.3f}")
    
    # Verificar archivos de estado
    print("\n📁 ARCHIVOS DE ESTADO:")
    state_path = "backend/mlops/replacement_results/system_state.json"
    if os.path.exists(state_path):
        print(f"   ✅ system_state.json existe")
        with open(state_path, 'r') as f:
            state = json.load(f)
        print(f"   - Modelos en archivo: {len(state.get('candidate_models', []))}")
        print(f"   - Evaluaciones en archivo: {len(state.get('evaluation_history', []))}")
    else:
        print(f"   ❌ system_state.json NO existe")
    
    # Verificar modelos demo
    print("\n🤖 MODELOS DEMO:")
    demo_models = [
        "backend/models/saved/demo_model_a.pkl",
        "backend/models/saved/demo_model_b.pkl",
        "backend/models/saved/demo_vectorizer.pkl"
    ]
    for model_path in demo_models:
        exists = os.path.exists(model_path)
        print(f"   {'✅' if exists else '❌'} {model_path}")
    
    print("\n" + "=" * 60)
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    
    # Verificar si hay modelos con score 0
    zero_score_models = [m for m in replacement_system.candidate_models 
                        if m.get('performance_metrics', {}).get('avg_overall_score', 0) == 0]
    
    if zero_score_models:
        print("   ⚠️  Hay modelos con score 0.0:")
        for model in zero_score_models:
            print(f"      - {model['name']}: {model.get('performance_metrics', {}).get('avg_overall_score', 0):.3f}")
        print("   🔧 Solución: Ejecutar evaluaciones para actualizar métricas")
    
    # Verificar si hay duplicados
    model_names = [m['name'] for m in replacement_system.candidate_models]
    duplicates = set([name for name in model_names if model_names.count(name) > 1])
    if duplicates:
        print(f"   ⚠️  Hay modelos duplicados: {duplicates}")
        print("   🔧 Solución: El sistema debería limpiar automáticamente")
    
    # Verificar si el modelo actual tiene score 0
    if replacement_system.current_model:
        current_score = replacement_system.current_model.get('performance_metrics', {}).get('avg_overall_score', 0)
        if current_score == 0:
            print(f"   ⚠️  Modelo actual tiene score 0.0")
            print("   🔧 Solución: Establecer un modelo con score > 0")
    
    print("\n✅ Diagnóstico completado")

if __name__ == "__main__":
    diagnose_models()
