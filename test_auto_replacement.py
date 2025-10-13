#!/usr/bin/env python3
"""
Test para verificar que el Auto-reemplazo de Modelos funciona correctamente
"""

import sys
import os
sys.path.append('backend')

from mlops.auto_model_replacement import AutoModelReplacement

def test_auto_replacement():
    """Probar el sistema de auto-reemplazo con modelos reales"""
    
    print("🧪 Probando Auto-reemplazo de Modelos...")
    
    # Crear sistema de auto-reemplazo
    replacement_system = AutoModelReplacement()
    
    # Verificar que existen modelos guardados
    model_files = [
        'backend/models/saved/improved_model.pkl',
        'backend/models/saved/balanced_model.pkl'
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            available_models.append(model_file)
            print(f"✅ Modelo encontrado: {model_file}")
        else:
            print(f"❌ Modelo no encontrado: {model_file}")
    
    if not available_models:
        print("❌ No hay modelos disponibles para probar")
        return False
    
    # Registrar modelos
    print("\n📝 Registrando modelos...")
    for i, model_file in enumerate(available_models):
        model_name = f"Model_{chr(65+i)}"  # Model_A, Model_B, etc.
        success = replacement_system.register_model(
            model_name, 
            model_file, 
            "hybrid"
        )
        if success:
            print(f"✅ {model_name} registrado exitosamente")
        else:
            print(f"❌ Error registrando {model_name}")
    
    # Crear datos de prueba
    test_data = [
        "I hate you so much",
        "You are amazing",
        "This is terrible",
        "I love this product",
        "You are stupid",
        "Great job!",
        "This sucks",
        "Wonderful work",
        "You are an idiot",
        "Excellent performance"
    ]
    
    true_labels = [
        "offensive", "neither", "offensive", "neither", "offensive",
        "neither", "offensive", "neither", "offensive", "neither"
    ]
    
    print(f"\n🔍 Evaluando modelos con {len(test_data)} textos de prueba...")
    
    # Evaluar cada modelo
    for model_name in [f"Model_{chr(65+i)}" for i in range(len(available_models))]:
        print(f"\n📊 Evaluando {model_name}...")
        
        try:
            # Buscar el modelo en candidatos
            model_info = None
            for candidate in replacement_system.candidate_models:
                if candidate['name'] == model_name:
                    model_info = candidate
                    break
            
            if model_info:
                # Cargar modelo real
                model = replacement_system._load_model(model_info['path'])
                if model:
                    # Hacer predicciones reales
                    predictions = replacement_system._make_predictions(model, test_data)
                    
                    # Evaluar modelo
                    evaluation = replacement_system.evaluate_model_performance(
                        model_name, test_data, true_labels, predictions
                    )
                    
                    if evaluation:
                        print(f"✅ {model_name}:")
                        print(f"   - Accuracy: {evaluation['accuracy']:.3f}")
                        print(f"   - F1-Score: {evaluation['f1_score']:.3f}")
                        print(f"   - Overall Score: {evaluation['overall_score']:.3f}")
                        print(f"   - Predicciones: {predictions[:3]}...")
                    else:
                        print(f"❌ Error evaluando {model_name}")
                else:
                    print(f"❌ No se pudo cargar {model_name}")
            else:
                print(f"❌ {model_name} no encontrado en candidatos")
                
        except Exception as e:
            print(f"❌ Error evaluando {model_name}: {e}")
    
    # Verificar reemplazo
    print("\n🔄 Verificando reemplazo...")
    replacement_result = replacement_system.check_for_replacement()
    
    if replacement_result:
        print("✅ Recomendación de reemplazo:")
        print(f"   - Candidato: {replacement_result['candidate_model']}")
        print(f"   - Actual: {replacement_result['current_model']}")
        print(f"   - Mejora: {replacement_result['improvement_percentage']:.1f}%")
        print(f"   - Confianza: {replacement_result['confidence']:.1f}%")
    else:
        print("ℹ️ No se recomienda reemplazo en este momento")
    
    # Mostrar estado final
    print("\n📊 Estado final de modelos:")
    status = replacement_system.get_model_status()
    
    for model in status['candidate_models']:
        metrics = model.get('performance_metrics', {})
        score = metrics.get('avg_overall_score', 0)
        evaluations = metrics.get('total_evaluations', 0)
        print(f"   {model['name']} ({model['status']}): Score = {score:.3f}, Evaluaciones = {evaluations}")
    
    # Verificar que los scores son realistas
    print("\n✅ Verificando scores:")
    all_scores_realistic = True
    
    for model in status['candidate_models']:
        metrics = model.get('performance_metrics', {})
        score = metrics.get('avg_overall_score', 0)
        
        if score == 0.0:
            print(f"❌ {model['name']} tiene score 0.000 (problema)")
            all_scores_realistic = False
        elif 0.0 <= score <= 1.0:
            print(f"✅ {model['name']} tiene score realista: {score:.3f}")
        else:
            print(f"❌ {model['name']} tiene score fuera de rango: {score:.3f}")
            all_scores_realistic = False
    
    if all_scores_realistic:
        print("\n🎉 Auto-reemplazo funciona correctamente!")
        return True
    else:
        print("\n❌ Auto-reemplazo tiene problemas con los scores")
        return False

if __name__ == "__main__":
    success = test_auto_replacement()
    if success:
        print("\n✅ Sistema de Auto-reemplazo funciona correctamente")
    else:
        print("\n❌ Sistema de Auto-reemplazo tiene problemas")
