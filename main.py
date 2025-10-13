
"""
Sistema de Clasificación de Hate Speech - Punto de entrada principal
Proyecto: Clasificación de Hate Speech con MLOps
"""

import sys
import os

# Añadir el directorio del proyecto al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def main():
    """Función principal para ejecutar la aplicación"""
    print("🚀 Sistema de Clasificación de Hate Speech")
    print("=" * 50)
    print("1. Aplicación Organizada (Recomendada)")
    print("2. Aplicación Simple")
    print("3. Aplicación Básica")
    print("=" * 50)
    
    choice = input("Selecciona una opción (1-3): ").strip()
    
    if choice == "1":
        print("🚀 Iniciando aplicación organizada...")
        os.system("streamlit run frontend/apps/app_organized.py --server.port 8501")
    elif choice == "2":
        print("🚀 Iniciando aplicación simple...")
        os.system("streamlit run frontend/apps/app_simple.py --server.port 8502")
    elif choice == "3":
        print("🚀 Iniciando aplicación básica...")
        os.system("streamlit run frontend/apps/app.py --server.port 8503")
    else:
        print("❌ Opción inválida. Ejecutando aplicación organizada por defecto...")
        os.system("streamlit run frontend/apps/app_organized.py --server.port 8501")

if __name__ == "__main__":
    main()
