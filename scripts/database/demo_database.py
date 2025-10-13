import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.database import DatabaseManager
import pandas as pd
from datetime import datetime, timedelta
import random

def demo_database():
    """Demostración completa de la funcionalidad de base de datos."""
    
    print("🗄️ DEMOSTRACIÓN DE BASE DE DATOS - HATE SPEECH DETECTOR")
    print("=" * 60)
    
    # Inicializar base de datos
    db = DatabaseManager("demo_hate_speech.db")
    print("✅ Base de datos inicializada")
    
    # 1. Simular predicciones
    print("\n📊 1. SIMULANDO PREDICCIONES...")
    test_texts = [
        "Hello, how are you?",
        "You are an idiot",
        "This is fucking stupid",
        "Have a great day!",
        "You're a moron",
        "Thanks for your help",
        "Go to hell",
        "Nice work!",
        "You're worthless",
        "Good morning everyone"
    ]
    
    classifications = ["clean", "offensive_language", "hate_speech"]
    models = ["UltimateHybrid", "MLRulesHybrid", "TransformerModel"]
    
    for i, text in enumerate(test_texts):
        classification = random.choice(classifications)
        confidence = random.uniform(0.6, 0.95)
        model = random.choice(models)
        
        db.log_prediction(
            text=text,
            classification=classification,
            confidence=confidence,
            model_used=model,
            preprocessing_info={"normalized": True, "language": "en"},
            user_ip=f"192.168.1.{random.randint(1, 100)}",
            session_id=f"session_{i+1}"
        )
    
    print(f"✅ {len(test_texts)} predicciones registradas")
    
    # 2. Simular métricas de modelos
    print("\n📈 2. SIMULANDO MÉTRICAS DE MODELOS...")
    for model in models:
        metrics = {
            'accuracy': random.uniform(0.85, 0.95),
            'precision': random.uniform(0.80, 0.90),
            'recall': random.uniform(0.75, 0.85),
            'f1_score': random.uniform(0.77, 0.87),
            'confidence_avg': random.uniform(0.70, 0.85),
            'response_time_ms': random.uniform(50, 200)
        }
        
        db.log_model_metrics(
            model_name=model,
            metrics=metrics,
            evaluation_type="evaluation",
            dataset_size=1000
        )
    
    print(f"✅ Métricas registradas para {len(models)} modelos")
    
    # 3. Simular reemplazo de modelo
    print("\n🔄 3. SIMULANDO REEMPLAZO DE MODELO...")
    db.log_model_replacement(
        old_model="OldModel_v1.0",
        new_model="UltimateHybrid_v2.0",
        old_score=0.82,
        new_score=0.89,
        improvement=0.07,
        reason="Mejor rendimiento en métricas",
        triggered_by="auto_replacement_system"
    )
    print("✅ Reemplazo de modelo registrado")
    
    # 4. Simular detección de drift
    print("\n📊 4. SIMULANDO DETECCIÓN DE DRIFT...")
    for i in range(5):
        db.log_drift_detection(
            drift_score=random.uniform(0.1, 0.8),
            kl_divergence=random.uniform(0.05, 0.6),
            ks_statistic=random.uniform(0.1, 0.7),
            p_value=random.uniform(0.001, 0.1),
            alert_level=random.choice(["normal", "warning", "critical"]),
            features_analyzed=50,
            sample_size=100
        )
    print("✅ 5 detecciones de drift registradas")
    
    # 5. Simular A/B testing
    print("\n🧪 5. SIMULANDO A/B TESTING...")
    db.log_ab_test(
        test_id="ab_test_001",
        model_a="Model_A",
        model_b="Model_B",
        traffic_split=0.5,
        results={
            'model_a_predictions': 150,
            'model_b_predictions': 145,
            'model_a_accuracy': 0.87,
            'model_b_accuracy': 0.91,
            'statistical_significance': 0.95,
            'recommendation': "Model_B es significativamente mejor"
        }
    )
    print("✅ A/B test registrado")
    
    # 6. Mostrar estadísticas
    print("\n📊 6. ESTADÍSTICAS DE LA BASE DE DATOS...")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 7. Mostrar resumen de predicciones
    print("\n📈 7. RESUMEN DE PREDICCIONES (últimos 7 días)...")
    predictions_summary = db.get_predictions_summary(days=7)
    print(predictions_summary.to_string(index=False))
    
    # 8. Mostrar historial de métricas
    print("\n📊 8. HISTORIAL DE MÉTRICAS DE MODELOS...")
    metrics_history = db.get_model_performance_history()
    print(metrics_history[['model_name', 'accuracy', 'f1_score', 'timestamp']].to_string(index=False))
    
    # 9. Mostrar historial de drift
    print("\n📊 9. HISTORIAL DE DETECCIÓN DE DRIFT...")
    drift_history = db.get_drift_history(days=30)
    print(drift_history[['drift_score', 'alert_level', 'timestamp']].to_string(index=False))
    
    print("\n✅ DEMOSTRACIÓN COMPLETADA")
    print("🗄️ Base de datos: demo_hate_speech.db")
    print("📊 Total de registros por tabla:")
    for key, value in stats.items():
        if key.endswith('_count'):
            table_name = key.replace('_count', '')
            print(f"   - {table_name}: {value}")

if __name__ == "__main__":
    demo_database()
