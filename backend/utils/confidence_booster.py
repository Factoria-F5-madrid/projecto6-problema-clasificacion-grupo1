"""
Booster de confianza para mejorar la certeza del modelo
Aumenta la confianza cuando detecta patrones claros
"""

import numpy as np
from typing import Dict, List, Tuple

class ConfidenceBooster:
    """Booster de confianza para mejorar la certeza del modelo"""
    
    def __init__(self):
        # Patrones que indican alta confianza
        self.hate_speech_patterns = [
            'inferior', 'superior', 'hate', 'kill', 'die', 'destroy',
            'women are', 'men are', 'all women', 'all men', 'immigrants',
            'jews', 'muslims', 'blacks', 'whites', 'gays', 'lesbians'
        ]
        
        self.offensive_patterns = [
            'fuck', 'shit', 'damn', 'hell', 'bitch', 'asshole',
            'stupid', 'idiot', 'moron', 'loser', 'pathetic'
        ]
        
        self.clean_patterns = [
            'hello', 'thank you', 'please', 'good morning', 'good afternoon',
            'how are you', 'nice to meet you', 'have a good day'
        ]
    
    def boost_confidence(self, text: str, probabilities: np.ndarray, 
                        classes: List[str]) -> Tuple[np.ndarray, str]:
        """
        Aumentar confianza basado en patrones detectados
        
        Args:
            text: Texto analizado
            probabilities: Probabilidades originales
            classes: Nombres de las clases
            
        Returns:
            Probabilidades ajustadas y explicación
        """
        text_lower = text.lower()
        boosted_probs = probabilities.copy()
        explanation = []
        
        # Detectar patrones de hate speech
        hate_score = sum(1 for pattern in self.hate_speech_patterns 
                        if pattern in text_lower)
        
        # Detectar patrones ofensivos
        offensive_score = sum(1 for pattern in self.offensive_patterns 
                             if pattern in text_lower)
        
        # Detectar patrones limpios
        clean_score = sum(1 for pattern in self.clean_patterns 
                         if pattern in text_lower)
        
        # Ajustar probabilidades basado en patrones
        if hate_score > 0:
            # Boost hate speech
            hate_idx = classes.index('Hate Speech') if 'Hate Speech' in classes else 0
            boost_factor = min(0.3, hate_score * 0.1)  # Máximo 30% de boost
            boosted_probs[hate_idx] = min(0.95, boosted_probs[hate_idx] + boost_factor)
            explanation.append(f"🚨 Patrones de hate speech detectados (+{boost_factor:.1%})")
        
        if offensive_score > 0:
            # Boost offensive language
            offensive_idx = classes.index('Offensive Language') if 'Offensive Language' in classes else 1
            boost_factor = min(0.2, offensive_score * 0.05)  # Máximo 20% de boost
            boosted_probs[offensive_idx] = min(0.95, boosted_probs[offensive_idx] + boost_factor)
            explanation.append(f"⚠️ Patrones ofensivos detectados (+{boost_factor:.1%})")
        
        if clean_score > 0:
            # Boost neither
            neither_idx = classes.index('Neither') if 'Neither' in classes else 2
            boost_factor = min(0.2, clean_score * 0.1)  # Máximo 20% de boost
            boosted_probs[neither_idx] = min(0.95, boosted_probs[neither_idx] + boost_factor)
            explanation.append(f"✅ Patrones limpios detectados (+{boost_factor:.1%})")
        
        # Normalizar probabilidades
        boosted_probs = boosted_probs / np.sum(boosted_probs)
        
        # Detectar evasiones (boost adicional)
        evasion_patterns = ['*', '@', '!', '1', '0', '3', '4', '5', '7']
        evasion_score = sum(1 for char in text if char in evasion_patterns)
        
        if evasion_score > 0:
            # Si hay evasiones, es más probable que sea ofensivo
            max_idx = np.argmax(boosted_probs)
            if max_idx != 2:  # No es Neither
                boost_factor = min(0.15, evasion_score * 0.05)
                boosted_probs[max_idx] = min(0.95, boosted_probs[max_idx] + boost_factor)
                explanation.append(f"🔍 Evasiones detectadas (+{boost_factor:.1%})")
        
        # Detectar longitud (textos muy cortos son más ambiguos)
        # PERO solo si no hay patrones claros detectados
        if len(text.split()) < 3 and not (hate_score > 0 or offensive_score > 0):
            # Reducir confianza para textos muy cortos solo si no hay patrones
            max_idx = np.argmax(boosted_probs)
            reduction = 0.05  # Reducir penalty
            boosted_probs[max_idx] = max(0.1, boosted_probs[max_idx] - reduction)
            explanation.append(f"📝 Texto corto sin patrones, confianza reducida (-{reduction:.1%})")
        
        # Normalizar nuevamente
        boosted_probs = boosted_probs / np.sum(boosted_probs)
        
        return boosted_probs, '; '.join(explanation) if explanation else "Sin ajustes aplicados"
    
    def get_confidence_level(self, probabilities: np.ndarray) -> str:
        """Determinar el nivel de confianza"""
        max_prob = np.max(probabilities)
        second_max = np.partition(probabilities, -2)[-2]
        difference = max_prob - second_max
        
        if max_prob > 0.8 and difference > 0.4:
            return "🟢 Muy Alta"
        elif max_prob > 0.6 and difference > 0.2:
            return "🟡 Alta"
        elif max_prob > 0.4 and difference > 0.1:
            return "🟠 Media"
        else:
            return "🔴 Baja"

def test_confidence_booster():
    """Test del booster de confianza"""
    print("🧪 TEST DEL BOOSTER DE CONFIANZA")
    print("=" * 50)
    
    booster = ConfidenceBooster()
    
    # Casos de prueba
    test_cases = [
        ("@sshole", [0.3, 0.4, 0.3]),
        ("Women are inferior to men", [0.4, 0.3, 0.3]),
        ("This is fucking stupid", [0.3, 0.5, 0.2]),
        ("Hello, how are you?", [0.1, 0.2, 0.7]),
        ("I hate all immigrants", [0.5, 0.3, 0.2])
    ]
    
    classes = ['Hate Speech', 'Offensive Language', 'Neither']
    
    for text, original_probs in test_cases:
        print(f"\n📝 Texto: '{text}'")
        print(f"   Probabilidades originales: {[f'{p:.1%}' for p in original_probs]}")
        
        # Aplicar booster
        boosted_probs, explanation = booster.boost_confidence(text, np.array(original_probs), classes)
        confidence_level = booster.get_confidence_level(boosted_probs)
        
        print(f"   Probabilidades mejoradas: {[f'{p:.1%}' for p in boosted_probs]}")
        print(f"   Nivel de confianza: {confidence_level}")
        print(f"   Explicación: {explanation}")
        
        # Mostrar mejora
        max_original = np.max(original_probs)
        max_boosted = np.max(boosted_probs)
        improvement = max_boosted - max_original
        print(f"   Mejora: +{improvement:.1%}")

if __name__ == "__main__":
    test_confidence_booster()
