#!/usr/bin/env python3
"""
Test del UltimateHybridSystem con detector de evasiones
"""

import sys
import os
sys.path.append('backend')

from backend.models.ultimate_hybrid_system import UltimateHybridSystem

def test_ultimate_with_evasions():
    """Probar el sistema definitivo con evasiones"""
    
    print("🧪 PROBANDO ULTIMATE HYBRID SYSTEM CON DETECTOR DE EVASIONES")
    print("=" * 70)
    
    # Crear sistema
    system = UltimateHybridSystem()
    
    # Casos de prueba con evasiones
    test_cases = [
        "You are an @sshole!",
        "f*ck you",
        "1d10t",
        "sh1t",
        "f*cking stupid",
        "h3ll0 world",  # No debería detectar como ofensivo
        "Hello, how are you?",  # No debería detectar nada
    ]
    
    for text in test_cases:
        print(f"\n📝 Texto: '{text}'")
        
        try:
            result = system.predict(text)
            print(f"   Predicción: {result['prediction']}")
            print(f"   Confianza: {result['confidence']:.2f}")
            print(f"   Método: {result['method']}")
            print(f"   Explicación: {result['explanation']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_ultimate_with_evasions()
