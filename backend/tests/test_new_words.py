#!/usr/bin/env python3
"""
Prueba con palabras NO hardcodeadas
Demuestra que el sistema funciona con palabras nuevas
"""

from backend.models.final_smart_selector import FinalSmartSelector

def test_new_words():
    """Probar con palabras no hardcodeadas"""
    print("🧪 PROBANDO CON PALABRAS NO HARDCODEADAS")
    print("=" * 50)
    
    # Inicializar selector
    selector = FinalSmartSelector()
    
    # Casos con palabras NO hardcodeadas
    test_cases = [
        # Palabras nuevas ofensivas
        "you are a jerk",           # "jerk" no está hardcodeado
        "that's ridiculous",        # "ridiculous" no está hardcodeado
        "you're pathetic",          # "pathetic" no está hardcodeado
        "what a loser",             # "loser" no está hardcodeado
        
        # Palabras nuevas positivas
        "you are brilliant",        # "brilliant" no está hardcodeado
        "that's magnificent",       # "magnificent" no está hardcodeado
        "you're outstanding",       # "outstanding" no está hardcodeado
        "what a genius",            # "genius" no está hardcodeado
        
        # Palabras completamente nuevas
        "xyz123 abc456",            # Palabras inventadas
        "qwerty asdfgh",            # Palabras sin sentido
        
        # Mezclas de palabras conocidas y nuevas
        "you fucking genius",       # "fucking" conocido + "genius" nuevo
        "that's fucking ridiculous", # "fucking" conocido + "ridiculous" nuevo
        "you are fucking brilliant", # "fucking" conocido + "brilliant" nuevo
    ]
    
    print("📊 Resultados con palabras NO hardcodeadas:")
    print("-" * 50)
    
    for text in test_cases:
        result = selector.predict(text)
        
        # Determinar si es correcto (simplificado)
        is_likely_correct = True
        if "jerk" in text.lower() or "ridiculous" in text.lower() or "pathetic" in text.lower() or "loser" in text.lower():
            is_likely_correct = result['prediction'] in ['Offensive Language', 'Hate Speech']
        elif "brilliant" in text.lower() or "magnificent" in text.lower() or "outstanding" in text.lower() or "genius" in text.lower():
            is_likely_correct = result['prediction'] == 'Neither'
        elif "xyz" in text.lower() or "qwerty" in text.lower():
            is_likely_correct = result['prediction'] == 'Neither'
        
        status_icon = "✅" if is_likely_correct else "❌"
        print(f"{status_icon} '{text}'")
        print(f"   → {result['prediction']} ({result['confidence']:.1%}) [{result['method']}]")
        print(f"   💡 {result['explanation']}")
        print()

if __name__ == "__main__":
    test_new_words()
