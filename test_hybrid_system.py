"""
Test script for the complete hybrid hate speech detection system
"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv

def test_hybrid_system():
    """Test the complete hybrid system"""
    
    # Cargar variables de entorno PRIMERO
    load_dotenv()
    
    # Importar detectores DESPUÉS de cargar .env
    from utils.api_verve import verve_detector
    from utils.api_neutrino import neutrino_detector
    from utils.api_ninja import ninja_detector
    from utils.api_perspective import perspective_detector
    
    print("🧪 TESTING COMPLETE HYBRID SYSTEM")
    print("=" * 60)
    
    # Verificar estado de cada API
    print("📊 API STATUS:")
    print("-" * 30)
    print(f"API Verve: {'✅ Available' if verve_detector.is_available() else '❌ Not configured'}")
    print(f"Neutrino API: {'✅ Available' if neutrino_detector.is_available() else '❌ Not configured'}")
    print(f"API Ninja: {'✅ Available' if ninja_detector.is_available() else '❌ Not configured'}")
    print(f"Google Perspective: {'✅ Available' if perspective_detector.is_available() else '❌ Not configured'}")
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
    
    print("📝 TESTING EACH API INDIVIDUALLY:")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Texto: '{text}'")
        print("-" * 40)
        
        # Probar cada API individualmente
        apis = [
            ("API Verve", verve_detector, "detect_hate_speech"),
            ("Neutrino API", neutrino_detector, "detect_profanity"),
            ("API Ninja", ninja_detector, "detect_profanity"),
            ("Google Perspective", perspective_detector, "detect_toxicity")
        ]
        
        for api_name, detector, method_name in apis:
            if detector.is_available():
                try:
                    method = getattr(detector, method_name)
                    result = method(text)
                    
                    if "error" in result:
                        print(f"   {api_name}: ❌ {result['error']}")
                    else:
                        print(f"   {api_name}: ✅ {result['classification']} (conf: {result['confidence']:.2f})")
                except Exception as e:
                    print(f"   {api_name}: ❌ Exception: {e}")
            else:
                print(f"   {api_name}: ⏭️  Not configured")
    
    print("\n" + "=" * 60)
    print("🎯 TESTING HYBRID SYSTEM (as used in app.py):")
    print("=" * 60)
    
    # Simular la lógica del sistema híbrido
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Texto: '{text}'")
        print("-" * 40)
        
        # Nivel 1: API Verve
        if verve_detector.is_available():
            verve_result = verve_detector.detect_hate_speech(text)
            if "error" not in verve_result and verve_result['confidence'] > 0.7:
                print(f"   🎯 RESULTADO: {verve_result['classification']} (conf: {verve_result['confidence']:.2f}) - Fuente: API Verve")
                continue
        
        # Nivel 2: Neutrino API
        if neutrino_detector.is_available():
            neutrino_result = neutrino_detector.detect_profanity(text)
            if "error" not in neutrino_result and neutrino_result['confidence'] > 0.6:
                print(f"   🎯 RESULTADO: {neutrino_result['classification']} (conf: {neutrino_result['confidence']:.2f}) - Fuente: Neutrino API")
                continue
        
        # Nivel 3: API Ninja
        if ninja_detector.is_available():
            ninja_result = ninja_detector.detect_profanity(text)
            if "error" not in ninja_result and ninja_result['confidence'] > 0.6:
                print(f"   🎯 RESULTADO: {ninja_result['classification']} (conf: {ninja_result['confidence']:.2f}) - Fuente: API Ninja")
                continue
        
        # Nivel 4: Google Perspective
        if perspective_detector.is_available():
            perspective_result = perspective_detector.detect_toxicity(text)
            if "error" not in perspective_result and perspective_result['confidence'] > 0.7:
                print(f"   🎯 RESULTADO: {perspective_result['classification']} (conf: {perspective_result['confidence']:.2f}) - Fuente: Google Perspective")
                continue
        
        # Nivel 5: Reglas básicas (simuladas)
        offensive_words = ['asshole', 'fuck', 'bitch', 'stupid', 'idiot', 'hate']
        text_lower = text.lower()
        found_offensive = [word for word in offensive_words if word in text_lower]
        
        if found_offensive:
            print(f"   🎯 RESULTADO: Offensive Language (conf: 0.8) - Fuente: Reglas básicas - Palabras: {found_offensive}")
        else:
            print(f"   🎯 RESULTADO: Neither (conf: 0.9) - Fuente: Reglas básicas")
    
    print("\n" + "=" * 60)
    print("✅ Prueba del sistema híbrido completada")

if __name__ == "__main__":
    test_hybrid_system()
