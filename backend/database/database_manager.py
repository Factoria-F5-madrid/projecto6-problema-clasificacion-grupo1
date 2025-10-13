import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any
import os

class DatabaseManager:
    """
    Gestor de base de datos SQLite para el sistema de detección de hate speech.
    Maneja logs de predicciones, métricas de modelos y historial de reemplazos.
    """
    
    def __init__(self, db_path: str = "hate_speech.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos con las tablas necesarias."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de predicciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                text TEXT NOT NULL,
                classification TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL,
                preprocessing_info TEXT,
                user_ip TEXT,
                session_id TEXT
            )
        ''')
        
        # Tabla de métricas de modelos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                confidence_avg REAL,
                response_time_ms REAL,
                evaluation_type TEXT,
                dataset_size INTEGER
            )
        ''')
        
        # Tabla de reemplazos de modelos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_replacements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                old_model TEXT NOT NULL,
                new_model TEXT NOT NULL,
                old_score REAL,
                new_score REAL,
                improvement REAL,
                reason TEXT,
                triggered_by TEXT
            )
        ''')
        
        # Tabla de drift detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                drift_score REAL NOT NULL,
                kl_divergence REAL,
                ks_statistic REAL,
                p_value REAL,
                alert_level TEXT,
                features_analyzed INTEGER,
                sample_size INTEGER
            )
        ''')
        
        # Tabla de A/B testing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_testing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                test_id TEXT NOT NULL,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                traffic_split REAL,
                model_a_predictions INTEGER,
                model_b_predictions INTEGER,
                model_a_accuracy REAL,
                model_b_accuracy REAL,
                statistical_significance REAL,
                recommendation TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, text: str, classification: str, confidence: float, 
                      model_used: str, preprocessing_info: Dict = None, 
                      user_ip: str = None, session_id: str = None):
        """Registra una predicción en la base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (text, classification, confidence, model_used, preprocessing_info, user_ip, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            text, classification, confidence, model_used,
            json.dumps(preprocessing_info) if preprocessing_info else None,
            user_ip, session_id
        ))
        
        conn.commit()
        conn.close()
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], 
                         evaluation_type: str = "evaluation", dataset_size: int = None):
        """Registra métricas de un modelo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics 
            (model_name, accuracy, precision_score, recall_score, f1_score, 
             confidence_avg, response_time_ms, evaluation_type, dataset_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            metrics.get('accuracy'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('f1_score'),
            metrics.get('confidence_avg'),
            metrics.get('response_time_ms'),
            evaluation_type,
            dataset_size
        ))
        
        conn.commit()
        conn.close()
    
    def log_model_replacement(self, old_model: str, new_model: str, 
                            old_score: float, new_score: float, 
                            improvement: float, reason: str, triggered_by: str):
        """Registra un reemplazo de modelo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_replacements 
            (old_model, new_model, old_score, new_score, improvement, reason, triggered_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (old_model, new_model, old_score, new_score, improvement, reason, triggered_by))
        
        conn.commit()
        conn.close()
    
    def log_drift_detection(self, drift_score: float, kl_divergence: float = None,
                          ks_statistic: float = None, p_value: float = None,
                          alert_level: str = "normal", features_analyzed: int = None,
                          sample_size: int = None):
        """Registra detección de drift."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_detection 
            (drift_score, kl_divergence, ks_statistic, p_value, alert_level, 
             features_analyzed, sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (drift_score, kl_divergence, ks_statistic, p_value, alert_level, 
              features_analyzed, sample_size))
        
        conn.commit()
        conn.close()
    
    def log_ab_test(self, test_id: str, model_a: str, model_b: str, 
                   traffic_split: float, results: Dict[str, Any]):
        """Registra resultados de A/B testing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_testing 
            (test_id, model_a, model_b, traffic_split, model_a_predictions, 
             model_b_predictions, model_a_accuracy, model_b_accuracy, 
             statistical_significance, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_id, model_a, model_b, traffic_split,
            results.get('model_a_predictions', 0),
            results.get('model_b_predictions', 0),
            results.get('model_a_accuracy'),
            results.get('model_b_accuracy'),
            results.get('statistical_significance'),
            results.get('recommendation')
        ))
        
        conn.commit()
        conn.close()
    
    def get_predictions_summary(self, days: int = 7) -> pd.DataFrame:
        """Obtiene resumen de predicciones de los últimos N días."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                DATE(timestamp) as date,
                classification,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM predictions 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp), classification
            ORDER BY date DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_model_performance_history(self, model_name: str = None) -> pd.DataFrame:
        """Obtiene historial de rendimiento de modelos."""
        conn = sqlite3.connect(self.db_path)
        
        if model_name:
            query = '''
                SELECT * FROM model_metrics 
                WHERE model_name = ?
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(model_name,))
        else:
            query = '''
                SELECT * FROM model_metrics 
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_drift_history(self, days: int = 30) -> pd.DataFrame:
        """Obtiene historial de detección de drift."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM drift_detection 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales de la base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Contar registros por tabla
        tables = ['predictions', 'model_metrics', 'model_replacements', 'drift_detection', 'ab_testing']
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[f'{table}_count'] = cursor.fetchone()[0]
        
        # Última predicción
        cursor.execute('SELECT MAX(timestamp) FROM predictions')
        stats['last_prediction'] = cursor.fetchone()[0]
        
        # Modelos únicos
        cursor.execute('SELECT COUNT(DISTINCT model_name) FROM model_metrics')
        stats['unique_models'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
