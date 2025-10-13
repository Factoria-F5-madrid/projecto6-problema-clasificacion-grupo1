"""
Detector Inteligente de Evasiones
Detecta y normaliza lenguaje ofensivo disfrazado
"""

import re
from typing import List, Dict, Tuple

class SmartEvasionDetector:
    """Detector inteligente de evasiones en texto ofensivo"""
    
    def __init__(self):
        # Diccionario de reemplazos comunes
        self.replacements = {
            '@': 'a', '4': 'a', '1': 'i', '!': 'i', '3': 'e', 
            '0': 'o', '$': 's', '*': '', '#': 'h', '7': 't',
            '5': 's', '8': 'b', '2': 'z', '6': 'g', '9': 'g'
        }
        
        # Palabras ofensivas base para comparación
        self.offensive_words = [
            'fuck', 'shit', 'idiot', 'asshole', 'bitch', 'bastard',
            'damn', 'crap', 'stupid', 'moron', 'dumb',
            'hate', 'kill', 'die', 'ugly', 'fat', 'loser',
            'jerk', 'freak', 'weird', 'gay', 'retard', 'nigger',
            'faggot', 'whore', 'slut', 'cunt', 'pussy',
            'dick', 'cock', 'penis', 'vagina', 'sex', 'porn'
        ]
        
        # Palabras NO ofensivas que podrían confundirse
        self.safe_words = [
            'hello', 'hi', 'hey', 'help', 'hell', 'well', 'tell', 'sell',
            'bell', 'fell', 'yell', 'shell', 'smell', 'spell', 'cell'
        ]
        
        # Patrones de evasión comunes
        self.evasion_patterns = [
            r'[a-z]*[!@#$%^&*()_+=\[\]{}|;:,.<>?/~`][a-z]*',  # Caracteres especiales
            r'[a-z]*[0-9][a-z]*',  # Números mezclados
            r'[a-z]*[!@#$%^&*()_+=\[\]{}|;:,.<>?/~`][0-9][a-z]*',  # Especiales + números
        ]
    
    def normalize_word(self, word: str) -> str:
        """Normaliza una palabra aplicando reemplazos de caracteres"""
        normalized = word.lower()
        
        # Aplicar reemplazos de caracteres
        for char, replacement in self.replacements.items():
            normalized = normalized.replace(char, replacement)
        
        return normalized
    
    def is_offensive_evasion(self, word: str) -> Tuple[bool, str]:
        """
        Detecta si una palabra es una evasión de una palabra ofensiva
        
        Returns:
            (is_offensive, normalized_word)
        """
        # Normalizar la palabra
        normalized = self.normalize_word(word)
        
        # Verificar si es una palabra segura (evitar falsos positivos)
        if normalized in self.safe_words:
            return False, word
        
        # Verificar si coincide exactamente con una palabra ofensiva
        if normalized in self.offensive_words:
            return True, normalized
        
        # Verificar si contiene una palabra ofensiva (para casos como "fucking")
        for offensive in self.offensive_words:
            if offensive in normalized and len(normalized) >= len(offensive) - 1:
                return True, offensive
        
        # Verificar patrones específicos de evasión
        if self._has_evasion_pattern(word):
            # Intentar diferentes variaciones de normalización
            variations = self._generate_variations(word)
            for variation in variations:
                if variation in self.offensive_words:
                    return True, variation
            
            # Verificar patrones específicos como f*ck, sh*t, etc.
            if self._matches_offensive_pattern(word):
                return True, self._extract_offensive_word(word)
        
        return False, word
    
    def _has_evasion_pattern(self, word: str) -> bool:
        """Verifica si una palabra tiene patrones de evasión"""
        # Contar caracteres especiales y números
        special_chars = sum(1 for c in word if c in '!@#$%^&*()_+={}[]|;:,.<>?/~`')
        numbers = sum(1 for c in word if c.isdigit())
        
        # Si tiene caracteres especiales o números, probablemente es evasión
        return special_chars > 0 or numbers > 0
    
    def _generate_variations(self, word: str) -> List[str]:
        """Genera variaciones de normalización para una palabra"""
        variations = []
        
        # Aplicar diferentes combinaciones de reemplazos
        current = word.lower()
        variations.append(current)
        
        # Reemplazos básicos
        for char, replacement in self.replacements.items():
            if char in current:
                variations.append(current.replace(char, replacement))
        
        # Combinaciones múltiples
        temp = word.lower()
        for char, replacement in self.replacements.items():
            temp = temp.replace(char, replacement)
        variations.append(temp)
        
        return variations
    
    def _matches_offensive_pattern(self, word: str) -> bool:
        """Verifica si una palabra coincide con patrones ofensivos específicos"""
        word_lower = word.lower()
        
        # Patrones específicos de evasión
        patterns = [
            r'f[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]ck',  # f*ck, f!ck, etc.
            r'sh[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]t',   # sh*t, sh!t, etc.
            r'b[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]tch',  # b*tch, b!tch, etc.
            r'd[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]mn',   # d*mn, d!mn, etc.
            r'c[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]nt',   # c*nt, c!nt, etc.
        ]
        
        for pattern in patterns:
            if re.search(pattern, word_lower):
                return True
        
        return False
    
    def _extract_offensive_word(self, word: str) -> str:
        """Extrae la palabra ofensiva de un patrón de evasión"""
        word_lower = word.lower()
        
        # Mapeo de patrones a palabras ofensivas
        pattern_mappings = {
            r'f[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]ck': 'fuck',
            r'sh[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]t': 'shit',
            r'b[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]tch': 'bitch',
            r'd[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]mn': 'damn',
            r'c[*!@#$%^&*()_+=\[\]{}|;:,.<>?/~`]nt': 'cunt',
        }
        
        for pattern, offensive_word in pattern_mappings.items():
            if re.search(pattern, word_lower):
                return offensive_word
        
        return word
    
    def detect_evasions(self, text: str) -> Dict[str, any]:
        """
        Detecta evasiones en el texto completo
        
        Returns:
            Dict con información sobre evasiones detectadas
        """
        words = text.split()
        evasions_found = []
        normalized_words = []
        
        for word in words:
            # Limpiar palabra de puntuación
            clean_word = re.sub(r'[^\w@#$%^&*()_+=\[\]{}|;:,.<>?/~`0-9]', '', word)
            
            if len(clean_word) > 2:  # Solo palabras de más de 2 caracteres
                is_offensive, normalized = self.is_offensive_evasion(clean_word)
                
                if is_offensive:
                    evasions_found.append({
                        'original': word,
                        'clean': clean_word,
                        'normalized': normalized,
                        'position': text.find(word)
                    })
                    normalized_words.append(normalized)
                else:
                    normalized_words.append(word)
            else:
                normalized_words.append(word)
        
        return {
            'has_evasions': len(evasions_found) > 0,
            'evasions': evasions_found,
            'normalized_text': ' '.join(normalized_words),
            'evasion_count': len(evasions_found)
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliza el texto completo, reemplazando evasiones por palabras ofensivas normales
        
        Returns:
            Texto normalizado
        """
        result = self.detect_evasions(text)
        return result['normalized_text']
    
    def get_evasion_confidence(self, text: str) -> float:
        """
        Calcula la confianza de que el texto contiene evasiones ofensivas
        
        Returns:
            Float entre 0.0 y 1.0
        """
        result = self.detect_evasions(text)
        
        if not result['has_evasions']:
            return 0.0
        
        # Calcular confianza basada en:
        # 1. Número de evasiones detectadas
        # 2. Longitud del texto
        # 3. Proporción de evasiones vs palabras totales
        
        total_words = len(text.split())
        evasion_ratio = result['evasion_count'] / max(total_words, 1)
        
        # Confianza base por evasión detectada
        base_confidence = min(0.8, result['evasion_count'] * 0.3)
        
        # Bonus por alta proporción de evasiones
        ratio_bonus = min(0.2, evasion_ratio * 0.5)
        
        return min(1.0, base_confidence + ratio_bonus)

def test_evasion_detector():
    """Probar el detector de evasiones"""
    
    print("🧪 PROBANDO DETECTOR INTELIGENTE DE EVASIONES")
    print("=" * 60)
    
    detector = SmartEvasionDetector()
    
    # Casos de prueba
    test_cases = [
        "You are an @sshole!",
        "f*ck you",
        "1d10t",
        "sh1t",
        "f*cking stupid",
        "h3ll0 world", 
        "Hello, how are you?",  
        "You are a f*cking @sshole!",
        "st*pid m0r0n",
        "b1tch please"
    ]
    
    for text in test_cases:
        print(f"\n📝 Texto: '{text}'")
        
        result = detector.detect_evasions(text)
        normalized = detector.normalize_text(text)
        confidence = detector.get_evasion_confidence(text)
        
        print(f"   Evasiones: {result['evasion_count']}")
        print(f"   Normalizado: '{normalized}'")
        print(f"   Confianza: {confidence:.2f}")
        
        if result['evasions']:
            print("   Detalles:")
            for evasion in result['evasions']:
                print(f"     '{evasion['original']}' → '{evasion['normalized']}'")

if __name__ == "__main__":
    test_evasion_detector()
