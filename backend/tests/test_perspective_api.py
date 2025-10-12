"""
Test script for Google Perspective API integration
"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv
from utils.api_perspective import perspective_detector

def test_perspective_api():
    """Test Google Perspective API with sample texts"""
    
    # Cargar variables de entorno
    load_dotenv()
    
    print("🧪 TESTING GOOGLE PERSPECTIVE API")
    print("=" * 50)
    
    # Verificar si la API está disponible
    if not perspective_detector.is_available():
        print("❌ Google Perspective API no está configurada")
        print("   Asegúrate de tener PERSPECTIVE_API_KEY en tu archivo .env")
        return
    
    print("✅ Google Perspective API configurada correctamente")
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
        "What a beautiful day",  # Texto positivo
        "You are a faggot",     # Hate speech
        "I will kill you"       # Threat
    ]
    
    print("📝 Probando textos:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Texto: '{text}'")
        
        # Llamar a la API
        result = perspective_detector.detect_toxicity(text)
        
        # Mostrar resultados
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Clasificación: {result['classification']}")
            print(f"   📊 Confianza: {result['confidence']:.2f}")
            print(f"   🔍 Fuente: {result['source']}")
            print(f"   🚫 Tipo principal: {result.get('main_toxicity_type', 'N/A')}")
            print(f"   📈 Score principal: {result.get('main_toxicity_score', 0):.2f}")
            print(f"   🎯 Toxicidad: {result.get('toxicity_score', 0):.2f}")
            print(f"   ⚠️  Insulto: {result.get('insult', 0):.2f}")
            print(f"   💀 Amenaza: {result.get('threat', 0):.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Prueba completada")

if __name__ == "__main__":
    test_perspective_api()
