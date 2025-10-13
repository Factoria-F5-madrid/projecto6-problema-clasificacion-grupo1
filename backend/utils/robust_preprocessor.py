"""
Preprocesador robusto para detección de hate speech
Normaliza evasiones, detecta idioma y extrae características
"""

import re
import string
import unicodedata
from typing import Dict, List, Any, Optional
import pandas as pd

try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Importar el detector inteligente de evasiones
try:
    from .smart_evasion_detector import SmartEvasionDetector
    EVASION_DETECTOR_AVAILABLE = True
except ImportError:
    EVASION_DETECTOR_AVAILABLE = False

class RobustPreprocessor:
    """Preprocesador robusto para normalización de evasiones y limpieza de texto"""
    
    def __init__(self):
        """Inicializar preprocesador con patrones de evasión"""
        self.evasion_patterns = self._create_evasion_patterns()
        self.compiled_patterns = self._compile_patterns()
        
        # Inicializar detector inteligente de evasiones
        if EVASION_DETECTOR_AVAILABLE:
            self.smart_evasion_detector = SmartEvasionDetector()
        else:
            self.smart_evasion_detector = None
        
    def _create_evasion_patterns(self) -> Dict[str, str]:
        """Crear patrones de evasión comunes"""
        return {
            # Fuck variations
            'f*ck': 'fuck',
            'f_ck': 'fuck', 
            'fck': 'fuck',
            'f**k': 'fuck',
            'f!ck': 'fuck',
            'f@ck': 'fuck',
            'fuck': 'fuck',  # Keep original
            
            # Shit variations
            'sh*t': 'shit',
            'sht': 'shit',
            's**t': 'shit',
            's!t': 'shit',
            'shit': 'shit',  # Keep original
            
            # Asshole variations
            '@sshole': 'asshole',
            'a$$hole': 'asshole',
            'assh0le': 'asshole',
            'assh*le': 'asshole',
            'asshole': 'asshole',  # Keep original
            
            # Bitch variations
            'b*tch': 'bitch',
            'btch': 'bitch',
            'b**ch': 'bitch',
            'b!tch': 'bitch',
            'bitch': 'bitch',  # Keep original
            
            # Damn variations
            'd*mn': 'damn',
            'dmn': 'damn',
            'd**n': 'damn',
            'd!mn': 'damn',
            'damn': 'damn',  # Keep original
            
            # Hell variations
            'h*ll': 'hell',
            'hll': 'hell',
            'h**l': 'hell',
            'h!ll': 'hell',
            'hell': 'hell',  # Keep original
            
            # Stupid variations
            'st*pid': 'stupid',
            'stpid': 'stupid',
            'st**id': 'stupid',
            'st!pid': 'stupid',
            'stupid': 'stupid',  # Keep original
            
            # Idiot variations
            '1d10t': 'idiot',
            'id10t': 'idiot',
            '1d!0t': 'idiot',
            'id!ot': 'idiot',
            'idiot': 'idiot',  # Keep original
            
            # Spanish variations
            'j*d*r': 'joder',
            'jdr': 'joder',
            'j**er': 'joder',
            'j!der': 'joder',
            'joder': 'joder',  # Keep original
            
            'p*ta': 'puta',
            'pta': 'puta',
            'p**a': 'puta',
            'p!ta': 'puta',
            'puta': 'puta',  # Keep original
            
            'm*erda': 'mierda',
            'mrda': 'mierda',
            'm**da': 'mierda',
            'm!erda': 'mierda',
            'mierda': 'mierda',  # Keep original
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compilar patrones regex para eficiencia"""
        compiled = {}
        
        for pattern, replacement in self.evasion_patterns.items():
            # Escapar caracteres especiales excepto * y !
            escaped_pattern = re.escape(pattern).replace(r'\*', r'[\W_]*').replace(r'\!', r'[\W_]*')
            compiled[pattern] = re.compile(escaped_pattern, re.IGNORECASE)
        
        return compiled
    
    def normalize_evasions(self, text: str) -> str:
        """
        Normalizar evasiones en el texto usando detector inteligente
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto con evasiones normalizadas
        """
        if not isinstance(text, str):
            return text
        
        # Usar detector inteligente si está disponible
        if self.smart_evasion_detector is not None:
            try:
                # Usar el detector inteligente para normalización
                normalized_text = self.smart_evasion_detector.normalize_text(text)
                return normalized_text
            except Exception as e:
                print(f"Error en detector inteligente, usando método básico: {e}")
        
        # Fallback al método básico si el detector inteligente falla
        normalized_text = text.lower()
        
        # Aplicar patrones de evasión básicos
        for pattern, replacement in self.evasion_patterns.items():
            if pattern in normalized_text:
                # Usar regex para reemplazo más preciso
                normalized_text = re.sub(
                    pattern.replace('*', r'[\W_]*').replace('!', r'[\W_]*'),
                    replacement,
                    normalized_text,
                    flags=re.IGNORECASE
                )
        
        return normalized_text
    
    def clean_text(self, text: str) -> str:
        """
        Limpiar texto básico
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Eliminar menciones (@usuario)
        text = re.sub(r'@\w+', '', text)
        
        # Eliminar hashtags (#hashtag)
        text = re.sub(r'#\w+', '', text)
        
        # Eliminar caracteres especiales excepto espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar espacios al inicio y final
        text = text.strip()
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalizar caracteres Unicode
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto con Unicode normalizado
        """
        if not isinstance(text, str):
            return text
        
        # Normalizar Unicode (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Detectar idioma del texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Código de idioma detectado
        """
        if not LANGDETECT_AVAILABLE or not isinstance(text, str) or len(text.strip()) < 3:
            return 'unknown'
        
        try:
            # Usar texto original para detección de idioma
            language = detect(text)
            return language
        except Exception as e:
            print(f"Error detectando idioma: {e}")
            return 'unknown'
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extraer características del texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con características extraídas
        """
        if not isinstance(text, str):
            return {}
        
        features = {}
        
        # Características básicas
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Características de mayúsculas
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = features['uppercase_count'] / max(len(text), 1)
        
        # Características de puntuación
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['special_char_count'] = sum(1 for c in text if c in string.punctuation)
        
        # Características de repetición
        words = text.split()
        if words:
            features['repeated_words'] = len(words) - len(set(words))
            features['repetition_ratio'] = features['repeated_words'] / len(words)
        else:
            features['repeated_words'] = 0
            features['repetition_ratio'] = 0
        
        # Detectar patrones específicos
        features['has_evasion'] = any(pattern in text.lower() for pattern in self.evasion_patterns.keys())
        features['has_mention'] = '@' in text
        features['has_hashtag'] = '#' in text
        features['has_url'] = 'http' in text.lower()
        
        return features
    
    def preprocess_text(self, text: str, 
                       normalize_evasions: bool = True,
                       clean_text: bool = True,
                       normalize_unicode: bool = True,
                       extract_features: bool = False) -> Dict[str, Any]:
        """
        Preprocesar texto completo
        
        Args:
            text: Texto a preprocesar
            normalize_evasions: Si normalizar evasiones
            clean_text: Si limpiar texto básico
            normalize_unicode: Si normalizar Unicode
            extract_features: Si extraer características
            
        Returns:
            Diccionario con texto procesado y metadatos
        """
        result = {
            'original_text': text,
            'processed_text': text,
            'language': 'unknown',
            'features': {},
            'evasions_found': []
        }
        
        if not isinstance(text, str):
            return result
        
        processed_text = text
        
        # Paso 1: Normalizar Unicode
        if normalize_unicode:
            processed_text = self.normalize_unicode(processed_text)
        
        # Paso 2: Detectar evasiones antes de normalizar
        if normalize_evasions:
            evasions_found = []
            for pattern in self.evasion_patterns.keys():
                if pattern in processed_text.lower():
                    evasions_found.append(pattern)
            result['evasions_found'] = evasions_found
            
            # Normalizar evasiones
            processed_text = self.normalize_evasions(processed_text)
        
        # Paso 3: Limpiar texto
        if clean_text:
            processed_text = self.clean_text(processed_text)
        
        # Paso 4: Detectar idioma (usar texto original)
        result['language'] = self.detect_language(text)
        
        # Paso 5: Extraer características
        if extract_features:
            result['features'] = self.extract_features(processed_text)
        
        result['processed_text'] = processed_text
        
        return result

def test_robust_preprocessor():
    """Test del preprocesador robusto"""
    print("🧪 TEST DEL PREPROCESADOR ROBUSTO")
    print("=" * 50)
    
    preprocessor = RobustPreprocessor()
    
    # Casos de prueba
    test_cases = [
        "Hello, how are you?",
        "You are an idiot",
        "F*ck you, @sshole",
        "Women are inferior to men",
        "This is fucking stupid",
        "I hate all immigrants",
        "No mames, qué pedo",
        "H*ll no, st*pid",
        "J*d*r, qué p*ta m*erda",
        "1d10t4 completo"
    ]
    
    print(f"🔍 PROBANDO {len(test_cases)} CASOS:")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Texto original: '{text}'")
        
        # Preprocesar
        result = preprocessor.preprocess_text(
            text, 
            normalize_evasions=True,
            clean_text=True,
            normalize_unicode=True,
            extract_features=True
        )
        
        print(f"   Procesado: '{result['processed_text']}'")
        print(f"   Idioma: {result['language']}")
        print(f"   Evasiones: {result['evasions_found']}")
        
        # Mostrar características importantes
        features = result['features']
        print(f"   Características:")
        print(f"     - Longitud: {features.get('length', 0)}")
        print(f"     - Palabras: {features.get('word_count', 0)}")
        print(f"     - Mayúsculas: {features.get('uppercase_ratio', 0):.1%}")
        print(f"     - Evasión: {features.get('has_evasion', False)}")
        print(f"     - Repetición: {features.get('repetition_ratio', 0):.1%}")
    
    print(f"\n✅ Test completado exitosamente!")

if __name__ == "__main__":
    test_robust_preprocessor()
