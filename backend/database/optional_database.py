# ===========================================
# BASE DE DATOS OPCIONAL - HATE SPEECH DETECTOR
# ===========================================

import os
import sqlite3
from typing import Optional, Dict, Any
from .database_manager import DatabaseManager

class OptionalDatabase:
    """
    Wrapper que hace la base de datos opcional.
    Si no se puede crear la base de datos, funciona sin ella.
    """
    
    def __init__(self, db_path: str = "hate_speech.db"):
        self.db_path = db_path
        self.db_manager = None
        self.enabled = False
        self._try_init_database()
    
    def _try_init_database(self):
        """Intenta inicializar la base de datos. Si falla, continúa sin ella."""
        try:
            self.db_manager = DatabaseManager(self.db_path)
            self.enabled = True
            print("✅ Base de datos inicializada correctamente")
        except Exception as e:
            print(f"⚠️ Base de datos no disponible: {e}")
            print("📊 Continuando en modo básico (sin logs persistentes)")
            self.enabled = False
    
    def log_prediction(self, text: str, classification: str, confidence: float, 
                      model_used: str, **kwargs):
        """Registra predicción si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                self.db_manager.log_prediction(
                    text=text, classification=classification, 
                    confidence=confidence, model_used=model_used, **kwargs
                )
            except Exception as e:
                print(f"⚠️ Error registrando predicción: {e}")
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], **kwargs):
        """Registra métricas si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                self.db_manager.log_model_metrics(
                    model_name=model_name, metrics=metrics, **kwargs
                )
            except Exception as e:
                print(f"⚠️ Error registrando métricas: {e}")
    
    def log_model_replacement(self, old_model: str, new_model: str, **kwargs):
        """Registra reemplazo si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                self.db_manager.log_model_replacement(
                    old_model=old_model, new_model=new_model, **kwargs
                )
            except Exception as e:
                print(f"⚠️ Error registrando reemplazo: {e}")
    
    def log_drift_detection(self, drift_score: float, **kwargs):
        """Registra drift si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                self.db_manager.log_drift_detection(
                    drift_score=drift_score, **kwargs
                )
            except Exception as e:
                print(f"⚠️ Error registrando drift: {e}")
    
    def log_ab_test(self, test_id: str, model_a: str, model_b: str, **kwargs):
        """Registra A/B test si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                self.db_manager.log_ab_test(
                    test_id=test_id, model_a=model_a, model_b=model_b, **kwargs
                )
            except Exception as e:
                print(f"⚠️ Error registrando A/B test: {e}")
    
    def get_predictions_summary(self, days: int = 7):
        """Obtiene resumen si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                return self.db_manager.get_predictions_summary(days=days)
            except Exception as e:
                print(f"⚠️ Error obteniendo resumen: {e}")
        return None
    
    def get_database_stats(self):
        """Obtiene estadísticas si la base de datos está disponible."""
        if self.enabled and self.db_manager:
            try:
                return self.db_manager.get_database_stats()
            except Exception as e:
                print(f"⚠️ Error obteniendo estadísticas: {e}")
        return {"status": "disabled", "reason": "Database not available"}
    
    def is_enabled(self):
        """Retorna True si la base de datos está disponible."""
        return self.enabled
