#!/usr/bin/env python3
"""
Script para iniciar la app de Hate Speech Detection
Permite elegir entre diferentes versiones
"""

import subprocess
import sys
import os

def main():
    """Función principal para elegir la app"""
    
    print("🚨 Detector de Hate Speech - Selector de Versión")
    print("=" * 50)
    print()
    print("Elige la versión que quieres usar:")
    print()
    print("1. 🏠 App Simple (Recomendada)")
    print("   - Interfaz limpia y fácil de usar")
    print("   - Solo funciones esenciales")
    print("   - Perfecta para usuarios finales")
    print()
    print("2. 📊 App Organizada (Para desarrolladores)")
    print("   - Menú con diferentes funciones")
    print("   - Comparación de modelos")
    print("   - Métricas y configuración avanzada")
    print()
    print("3. 🔧 App Original (Completa)")
    print("   - Todas las funciones disponibles")
    print("   - Interfaz completa pero compleja")
    print()
    
    while True:
        try:
            choice = input("Ingresa tu opción (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 Iniciando App Simple...")
                print("📱 Abre tu navegador en: http://localhost:8515")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app_simple.py", "--server.port", "8515"])
                break
                
            elif choice == "2":
                print("\n🚀 Iniciando App Organizada...")
                print("📱 Abre tu navegador en: http://localhost:8516")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app_organized.py", "--server.port", "8516"])
                break
                
            elif choice == "3":
                print("\n🚀 Iniciando App Original...")
                print("📱 Abre tu navegador en: http://localhost:8501")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
                break
                
            else:
                print("❌ Opción inválida. Por favor, elige 1, 2 o 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
