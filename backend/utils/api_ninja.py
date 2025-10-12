"""
API Ninja integration for profanity detection
"""

import requests
import os
from typing import Dict, Any, Optional

class APINinjaDetector:
    """API Ninja detector for profanity and content analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('NINJA_API_KEY')
        self.base_url = "https://api.api-ninjas.com/v1"
        self.timeout = 10  # 10 segundos timeout
    
    def detect_profanity(self, text: str) -> Dict[str, Any]:
        """
        Detect profanity using API Ninja
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not self.api_key:
            return {
                "error": "API key not configured",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "ninja"
            }
        
        try:
            # Preparar headers
            headers = {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Preparar payload
            payload = {
                "text": text
            }
            
            # Llamar a la API de Profanity Check
            response = requests.post(
                f"{self.base_url}/profanitycheck",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                data = response.json()
                return self._process_ninja_response(data, text)
            else:
                return {
                    "error": f"API error: {response.status_code} - {response.text}",
                    "classification": "Neither",
                    "confidence": 0.0,
                    "source": "ninja"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "API timeout",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "ninja"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Request error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "ninja"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "ninja"
            }
    
    def _process_ninja_response(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        Process API Ninja response
        
        Args:
            data: Raw API response
            original_text: Original text that was analyzed
            
        Returns:
            Processed detection result
        """
        try:
            # API Ninja devuelve algo como:
            # {
            #     "is_profane": true/false,
            #     "profanity_count": 2,
            #     "profanity_words": ["word1", "word2"]
            # }
            
            is_profane = data.get('is_profane', False)
            profanity_count = data.get('profanity_count', 0)
            profanity_words = data.get('profanity_words', [])
            
            # Calcular confianza basada en el número de palabras ofensivas
            confidence = min(0.9, 0.5 + (profanity_count * 0.2)) if is_profane else 0.1
            
            # Clasificar basado en el resultado
            if is_profane and profanity_count > 0:
                # Si hay palabras ofensivas, clasificar como Offensive Language
                classification = "Offensive Language"
            else:
                classification = "Neither"
            
            return {
                "classification": classification,
                "confidence": confidence,
                "source": "ninja",
                "profanity_words": profanity_words,
                "profanity_count": profanity_count,
                "raw_response": data
            }
            
        except Exception as e:
            return {
                "error": f"Error processing response: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "ninja"
            }
    
    def is_available(self) -> bool:
        """
        Check if API Ninja is available
        
        Returns:
            True if API key is configured
        """
        return self.api_key is not None and self.api_key != "tu_api_key_de_ninja_aqui"
    
    def get_remaining_requests(self) -> Optional[int]:
        """
        Get remaining API requests (if available in response)
        
        Returns:
            Number of remaining requests or None
        """
        # API Ninja puede incluir info sobre requests restantes en headers
        return None

# Instancia global para usar en la app
ninja_detector = APINinjaDetector()
