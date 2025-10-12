#!/usr/bin/env python3
"""
Debug para "eres muy tonto"
"""

from backend.models.final_smart_selector import FinalSmartSelector

def debug_tonto():
    """Debug específico para 'eres muy tonto'"""
    print("🔍 DEBUG: 'eres muy tonto'")
    print("=" * 50)
    
    # Inicializar selector
    selector = FinalSmartSelector()
    
    # Probar el caso problemático
    text = "eres muy tonto"
    result = selector.predict(text)
    
    print(f"📝 Texto: '{text}'")
    print(f"🎯 Predicción: {result['prediction']}")
    print(f"📊 Confianza: {result['confidence']:.1%}")
    print(f"🔧 Método: {result['method']}")
    print(f"💡 Explicación: {result['explanation']}")
    print()
    
    # Verificar si "tonto" está en las reglas
    print("🔍 Verificando reglas:")
    print(f"   - 'tonto' en offensive_words: {'tonto' in selector.offensive_words}")
    print(f"   - 'tonto' en hate_patterns: {any('tonto' in pattern for pattern in selector.hate_patterns)}")
    print(f"   - 'tonto' en evasion_patterns: {any('tonto' in pattern for pattern in selector.evasion_patterns)}")
    print()
    
    # Probar variaciones
    test_cases = [
        "eres muy tonto",
        "tonto",
        "que tonto eres",
        "sois tontos",
        "son tontos"
    ]
    
    print("🧪 Probando variaciones:")
    for test_text in test_cases:
        test_result = selector.predict(test_text)
        print(f"   '{test_text}' → {test_result['prediction']} ({test_result['confidence']:.1%})")

if __name__ == "__main__":
    debug_tonto()
