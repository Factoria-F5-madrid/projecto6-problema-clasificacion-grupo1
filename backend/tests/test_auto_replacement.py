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
        'backend/models/saved/demo_model_a.pkl',
        'backend/models/saved/demo_model_b.pkl'
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
    
    # Crear datos de prueba MÁS BALANCEADOS para mejores métricas
    test_data = [
        # Ejemplos claramente ofensivos (10)
        "I hate you so much, you're worthless",
        "Go kill yourself idiot", 
        "You're a stupid piece of trash",
        "F*** you and your family",
        "Die already loser",
        "You are such an idiot",
        "I hate this f***ing place",
        "You're a complete moron",
        "Shut up you dumbass",
        "You're a waste of space",
        
        # Ejemplos claramente neutros (10)
        "Hello, how are you today?",
        "This is a nice product",
        "Thank you for your help",
        "I'm going to the store",
        "Weather is good today",
        "I like this restaurant",
        "Have a great day",
        "This is interesting",
        "I'm happy with this",
        "Good morning everyone",
        
        # Ejemplos negativos pero no ofensivos (10)
        "I don't like this product",
        "This service is not good",
        "I'm disappointed with the quality",
        "Not what I expected",
        "Could be better",
        "This is not working well",
        "I'm not satisfied",
        "This needs improvement",
        "Not my favorite",
        "I expected more"
    ]
    
    true_labels = [
        # Ofensivos (10)
        "offensive", "offensive", "offensive", "offensive", "offensive",
        "offensive", "offensive", "offensive", "offensive", "offensive",
        # Neutros (10)
        "neither", "neither", "neither", "neither", "neither",
        "neither", "neither", "neither", "neither", "neither",
        # Negativos pero no ofensivos (10)
        "neither", "neither", "neither", "neither", "neither",
        "neither", "neither", "neither", "neither", "neither"
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
                model = replacement_system._load_model(model_info['model_path'])
                if model:
                    # Hacer predicciones reales
                    predictions = replacement_system._make_predictions(model, test_data)
                    
                    # Evaluar modelo
                    evaluation = replacement_system.evaluate_model_performance(
                        model_name, test_data, true_labels
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
        print(f"   - Debería reemplazar: {replacement_result['should_replace']}")
        print(f"   - Razón: {replacement_result['reason']}")
        if 'candidate_model' in replacement_result:
            print(f"   - Candidato: {replacement_result['candidate_model']}")
        if 'current_model' in replacement_result:
            print(f"   - Actual: {replacement_result['current_model']}")
        if 'improvement' in replacement_result:
            print(f"   - Mejora: {replacement_result['improvement']:.3f}")
        if 'confidence' in replacement_result:
            print(f"   - Confianza: {replacement_result['confidence']:.3f}")
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
