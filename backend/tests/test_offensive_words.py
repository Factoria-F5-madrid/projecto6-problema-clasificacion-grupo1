#!/usr/bin/env python3
"""
Prueba de palabras ofensivas mejoradas
"""

from backend.models.final_smart_selector import FinalSmartSelector

def test_offensive_words():
    """Probar palabras ofensivas en español e inglés"""
    print("🧪 PRUEBA DE PALABRAS OFENSIVAS MEJORADAS")
    print("=" * 60)
    
    # Inicializar selector
    selector = FinalSmartSelector()
    
    # Casos de prueba
    test_cases = [
        # Español
        ("eres muy tonto", "Offensive Language"),
        ("que tonta eres", "Offensive Language"),
        ("sois tontos", "Offensive Language"),
        ("son tontas", "Offensive Language"),
        ("eres un imbécil", "Offensive Language"),
        ("que gilipollas", "Offensive Language"),
        ("eres un capullo", "Offensive Language"),
        ("hijo de puta", "Offensive Language"),
        ("joder, qué mierda", "Offensive Language"),
        
        # Inglés
        ("you are a jerk", "Offensive Language"),
        ("what a loser", "Offensive Language"),
        ("that's pathetic", "Offensive Language"),
        ("you're ridiculous", "Offensive Language"),
        ("dumbass", "Offensive Language"),
        
        # Casos limpios (deben seguir siendo Neither)
        ("Hello, how are you?", "Neither"),
        ("That's brilliant work", "Neither"),
        ("You are amazing", "Neither"),
        ("Great job!", "Neither"),
    ]
    
    print("📊 Resultados:")
    print("-" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        result = selector.predict(text)
        prediction = result['prediction']
        confidence = result['confidence']
        method = result['method']
        
        # Verificar si es correcto
        is_correct = prediction == expected
        if is_correct:
            correct += 1
        
        status_icon = "✅" if is_correct else "❌"
        print(f"{status_icon} '{text}'")
        print(f"   → {prediction} ({confidence:.1%}) [{method}]")
        if not is_correct:
            print(f"   ⚠️  Esperado: {expected}")
        print()
    
    # Resumen
    accuracy = (correct / total) * 100
    print("=" * 60)
    print(f"📈 Precisión: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("🎉 ¡Excelente! El sistema funciona muy bien")
    elif accuracy >= 60:
        print("👍 Bueno, pero se puede mejorar")
    else:
        print("⚠️ Necesita mejoras")

if __name__ == "__main__":
    test_offensive_words()
