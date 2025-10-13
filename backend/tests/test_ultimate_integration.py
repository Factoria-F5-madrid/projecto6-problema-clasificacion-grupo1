#!/usr/bin/env python3
"""
Prueba de integración del sistema definitivo
"""

from backend.models.ultimate_hybrid_system import UltimateHybridSystem

def test_ultimate_integration():
    """Probar la integración del sistema definitivo"""
    
    print("🧪 PRUEBA DE INTEGRACIÓN DEL SISTEMA DEFINITIVO")
    print("=" * 60)
    
    # Inicializar sistema
    system = UltimateHybridSystem()
    
    # Casos de prueba críticos
    test_cases = [
        # Casos que antes fallaban
        ("eres muy tonto", "Offensive Language"),
        ("eres un zoquete", "Offensive Language"),
        ("que bobo eres", "Offensive Language"),
        ("sois unos pardillos", "Offensive Language"),
        
        # Casos que siempre funcionaron
        ("fuck you", "Offensive Language"),
        ("Women are inferior to men", "Hate Speech"),
        ("Hello, how are you?", "Neither"),
        ("That's brilliant work", "Neither"),
        
        # Casos de evasión
        ("f*ck you", "Offensive Language"),
        ("@sshole", "Offensive Language"),
    ]
    
    print("📊 Resultados:")
    print("-" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        result = system.predict(text)
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
    
    if accuracy >= 90:
        print("🎉 ¡Excelente! Sistema definitivo funcionando perfectamente")
        return True
    elif accuracy >= 80:
        print("👍 Muy bueno, sistema híbrido efectivo")
        return True
    else:
        print("⚠️ Necesita ajustes adicionales")
        return False

if __name__ == "__main__":
    success = test_ultimate_integration()
    if success:
        print("\n✅ ¡Integración exitosa! Listo para commit.")
    else:
        print("\n❌ Integración fallida. Revisar errores.")
