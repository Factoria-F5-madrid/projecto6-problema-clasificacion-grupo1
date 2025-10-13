#!/usr/bin/env python3
"""
Test de integración del detector inteligente de evasiones
"""

import sys
import os
sys.path.append('backend')

from backend.utils.robust_preprocessor import RobustPreprocessor

def test_evasion_integration():
    """Probar la integración del detector de evasiones"""
    
    print("🧪 PROBANDO INTEGRACIÓN DE DETECTOR DE EVASIONES")
    print("=" * 60)
    
    # Crear preprocesador
    preprocessor = RobustPreprocessor()
    
    # Casos de prueba
    test_cases = [
        "You are an @sshole!",
        "f*ck you",
        "1d10t",
        "sh1t",
        "f*cking stupid",
        "h3ll0 world",  # No debería detectar como ofensivo
        "Hello, how are you?",  # No debería detectar nada
        "You are a f*cking @sshole!",
        "st*pid m0r0n",
        "b1tch please"
    ]
    
    for text in test_cases:
        print(f"\n📝 Texto original: '{text}'")
        
        # Normalizar evasiones
        normalized = preprocessor.normalize_evasions(text)
        print(f"   Normalizado: '{normalized}'")
        
        # Detectar evasiones si el detector está disponible
        if hasattr(preprocessor, 'smart_evasion_detector') and preprocessor.smart_evasion_detector:
            result = preprocessor.smart_evasion_detector.detect_evasions(text)
            confidence = preprocessor.smart_evasion_detector.get_evasion_confidence(text)
            print(f"   Evasiones detectadas: {result['evasion_count']}")
            print(f"   Confianza: {confidence:.2f}")
            
            if result['evasions']:
                print("   Detalles:")
                for evasion in result['evasions']:
                    print(f"     '{evasion['original']}' → '{evasion['normalized']}'")

if __name__ == "__main__":
    test_evasion_integration()
