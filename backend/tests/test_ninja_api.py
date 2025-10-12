"""
Test script for API Ninja integration
"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv
from utils.api_ninja import ninja_detector

def test_ninja_api():
    """Test API Ninja with sample texts"""
    
    # Cargar variables de entorno
    load_dotenv()
    
    print("🧪 TESTING API NINJA")
    print("=" * 50)
    
    # Verificar si la API está disponible
    if not ninja_detector.is_available():
        print("❌ API Ninja no está configurada")
        print("   Asegúrate de tener NINJA_API_KEY en tu archivo .env")
        return
    
    print("✅ API Ninja configurada correctamente")
    print()
    
    # Textos de prueba
    test_texts = [
        "Hello, how are you?",  # Texto normal
        "You are an asshole",   # Lenguaje ofensivo
        "Fuck you bitch",       # Lenguaje muy ofensivo
        "This is fucking stupid", # Lenguaje ofensivo
        "Good morning everyone", # Texto positivo
        "You are a fucking idiot", # Múltiples palabras ofensivas
        "I hate you so much",   # Posible hate speech
        "What a beautiful day"  # Texto positivo
    ]
    
    print("📝 Probando textos:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Texto: '{text}'")
        
        # Llamar a la API
        result = ninja_detector.detect_profanity(text)
        
        # Mostrar resultados
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Clasificación: {result['classification']}")
            print(f"   📊 Confianza: {result['confidence']:.2f}")
            print(f"   🔍 Fuente: {result['source']}")
            print(f"   🚫 Palabras ofensivas: {result.get('profanity_words', [])}")
            print(f"   📈 Total palabras ofensivas: {result.get('profanity_count', 0)}")
    
    print("\n" + "=" * 50)
    print("✅ Prueba completada")

if __name__ == "__main__":
    test_ninja_api()
