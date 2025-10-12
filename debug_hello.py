#!/usr/bin/env python3
"""
Debug para "hello, how are you?"
"""

from backend.models.ultimate_hybrid_system import UltimateHybridSystem

def debug_hello():
    """Debug específico para 'hello, how are you?'"""
    print("🔍 DEBUG: 'hello, how are you?'")
    print("=" * 50)
    
    # Inicializar sistema
    system = UltimateHybridSystem()
    
    # Probar el caso problemático
    text = "hello, how are you?"
    result = system.predict(text)
    
    print(f"📝 Texto: '{text}'")
    print(f"🎯 Predicción: {result['prediction']}")
    print(f"📊 Confianza: {result['confidence']:.1%}")
    print(f"🔧 Método: {result['method']}")
    print(f"💡 Explicación: {result['explanation']}")
    print()
    
    # Verificar qué palabras ofensivas está detectando
    text_lower = text.lower()
    print("🔍 Verificando palabras ofensivas:")
    
    offensive_found = []
    for word in system.offensive_words:
        if word in text_lower:
            offensive_found.append(word)
    
    if offensive_found:
        print(f"   ❌ Palabras ofensivas encontradas: {offensive_found}")
    else:
        print("   ✅ No se encontraron palabras ofensivas directas")
    
    print()
    
    # Verificar patrones de hate speech
    print("🔍 Verificando patrones de hate speech:")
    hate_found = []
    for pattern in system.hate_patterns:
        if re.search(pattern, text_lower):
            hate_found.append(pattern)
    
    if hate_found:
        print(f"   ❌ Patrones de hate speech encontrados: {hate_found}")
    else:
        print("   ✅ No se encontraron patrones de hate speech")
    
    print()
    
    # Verificar patrones de evasión
    print("🔍 Verificando patrones de evasión:")
    evasion_found = []
    for pattern in system.evasion_patterns:
        if re.search(pattern, text_lower):
            evasion_found.append(pattern)
    
    if evasion_found:
        print(f"   ❌ Patrones de evasión encontrados: {evasion_found}")
    else:
        print("   ✅ No se encontraron patrones de evasión")
    
    print()
    
    # Verificar frases positivas
    print("🔍 Verificando frases positivas:")
    positive_found = []
    for phrase in system.positive_phrases:
        if phrase in text_lower:
            positive_found.append(phrase)
    
    if positive_found:
        print(f"   ✅ Frases positivas encontradas: {positive_found}")
    else:
        print("   ❌ No se encontraron frases positivas")
    
    print()
    
    # Analizar contexto
    context = system._analyze_context(text)
    print("🔍 Análisis de contexto:")
    print(f"   - Indicadores positivos: {context['positive_indicators']}")
    print(f"   - Indicadores ofensivos: {context['offensive_indicators']}")
    print(f"   - Indicadores de hate: {context['hate_indicators']}")
    print(f"   - Indicadores de evasión: {context['evasion_indicators']}")

if __name__ == "__main__":
    import re
    debug_hello()
