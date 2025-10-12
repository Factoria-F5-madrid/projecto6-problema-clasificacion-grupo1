"""
Test script for API Verve integration
"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv
from utils.api_verve import verve_detector

def test_verve_api():
    """Test API Verve with sample texts"""
    
    # Cargar variables de entorno
    load_dotenv()
    
    print("🧪 TESTING API VERVE")
    print("=" * 50)
    
    # Verificar si la API está disponible
    if not verve_detector.is_available():
        print("❌ API Verve no está configurada")
        print("   Asegúrate de tener VERVE_API_KEY en tu archivo .env")
        return
    
    print("✅ API Verve configurada correctamente")
    print()
    
    # Textos de prueba
    test_texts = [
        "Hello, how are you?",  # Texto normal
        "You are an asshole",   # Lenguaje ofensivo
        "I hate you so much",   # Posible hate speech
        "Fuck you bitch",       # Lenguaje muy ofensivo
        "Good morning everyone" # Texto positivo
    ]
    
    print("📝 Probando textos:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Texto: '{text}'")
        
        # Llamar a la API
        result = verve_detector.detect_hate_speech(text)
        
        # Mostrar resultados
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Clasificación: {result['classification']}")
            print(f"   📊 Confianza: {result['confidence']:.2f}")
            print(f"   🔍 Fuente: {result['source']}")
    
    print("\n" + "=" * 50)
    print("✅ Prueba completada")

if __name__ == "__main__":
    test_verve_api()
