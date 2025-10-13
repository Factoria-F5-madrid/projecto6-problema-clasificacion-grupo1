#!/usr/bin/env python3
"""
Prueba con palabras NO reconocidas
"""

from backend.models.final_smart_selector import FinalSmartSelector

def test_unknown_words():
    """Probar con palabras que NO están en las reglas"""
    print("🧪 PRUEBA CON PALABRAS NO RECONOCIDAS")
    print("=" * 60)
    
    # Inicializar selector
    selector = FinalSmartSelector()
    
    # Casos con palabras NO hardcodeadas
    test_cases = [
        # Palabras ofensivas NO reconocidas
        ("eres un zoquete", "¿Offensive?"),           # "zoquete" no está en la lista
        ("que bobo eres", "¿Offensive?"),             # "bobo" no está en la lista
        ("sois unos pardillos", "¿Offensive?"),       # "pardillos" no está en la lista
        ("eres un memo", "¿Offensive?"),              # "memo" no está en la lista
        ("que palurdo", "¿Offensive?"),               # "palurdo" no está en la lista
        
        # Palabras inventadas
        ("xyz123 abc456", "¿Neither?"),               # Palabras sin sentido
        ("qwerty asdfgh", "¿Neither?"),               # Palabras sin sentido
        
        # Palabras neutras
        ("Hello, how are you?", "Neither"),           # Debería ser Neither
        ("That's brilliant work", "Neither"),         # Debería ser Neither
    ]
    
    print("📊 Resultados:")
    print("-" * 60)
    
    for text, expected in test_cases:
        result = selector.predict(text)
        prediction = result['prediction']
        confidence = result['confidence']
        method = result['method']
        
        # Determinar si es razonable
        is_reasonable = True
        if "zoquete" in text or "bobo" in text or "pardillos" in text or "memo" in text or "palurdo" in text:
            # Estas palabras son ofensivas pero no están en nuestras reglas
            # El ML debería detectarlas o al menos dar Neither
            is_reasonable = prediction in ['Offensive Language', 'Neither']
        elif "xyz" in text or "qwerty" in text:
            # Palabras sin sentido deberían ser Neither
            is_reasonable = prediction == 'Neither'
        elif "brilliant" in text or "Hello" in text:
            # Palabras positivas deberían ser Neither
            is_reasonable = prediction == 'Neither'
        
        status_icon = "✅" if is_reasonable else "❌"
        print(f"{status_icon} '{text}'")
        print(f"   → {prediction} ({confidence:.1%}) [{method}]")
        print(f"   💡 {result['explanation']}")
        print()
    
    print("=" * 60)
    print("🔍 ANÁLISIS:")
    print("• Palabras NO en reglas → ML decide")
    print("• ML no seguro → Neither (conservador)")
    print("• Palabras inventadas → Neither")
    print("• Palabras positivas → Neither")

if __name__ == "__main__":
    test_unknown_words()
