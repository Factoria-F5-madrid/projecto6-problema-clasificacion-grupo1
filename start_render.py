#!/usr/bin/env python3
# ===========================================
# SCRIPT DE INICIO OPTIMIZADO PARA RENDER
# ===========================================

import os
import sys
import subprocess

def main():
    """Inicia la aplicación Streamlit optimizada para Render."""
    
    # Obtener puerto de Render
    port = os.environ.get("PORT", "8501")
    
    print(f"🚀 Iniciando Hate Speech Detector en puerto {port}")
    
    # Comando de Streamlit optimizado para Render
    cmd = [
        "streamlit", "run", "frontend/apps/app_organized.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"📋 Comando: {' '.join(cmd)}")
    
    try:
        # Ejecutar Streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("🛑 Aplicación detenida por el usuario")
        sys.exit(0)

if __name__ == "__main__":
    main()
