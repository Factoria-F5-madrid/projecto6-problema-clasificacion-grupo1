#!/usr/bin/env python3
# ===========================================
# INICIALIZACIÓN AUTOMÁTICA DE BASE DE DATOS
# ===========================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.database import DatabaseManager

def init_database():
    """Inicializa la base de datos si no existe."""
    try:
        db = DatabaseManager("hate_speech.db")
        print("✅ Base de datos inicializada correctamente")
        print("📊 Tablas creadas:")
        print("   - predictions")
        print("   - model_metrics") 
        print("   - model_replacements")
        print("   - drift_detection")
        print("   - ab_testing")
        return True
    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        return False

if __name__ == "__main__":
    init_database()
