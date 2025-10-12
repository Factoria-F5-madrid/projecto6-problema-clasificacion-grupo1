"""
Neutrino API integration for profanity detection
"""

import requests
import os
from typing import Dict, Any, Optional

class NeutrinoAPIDetector:
    """Neutrino API detector for profanity and content analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('NEUTRINO_API_KEY')
        self.base_url = "https://neutrinoapi.net"
        self.timeout = 10  # 10 segundos timeout
    
    def detect_profanity(self, text: str) -> Dict[str, Any]:
        """
        Detect profanity using Neutrino API
        
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
                "source": "neutrino"
            }
        
        try:
            # Preparar headers
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "HateSpeechDetector/1.0"
            }
            
            # Preparar payload según documentación de Neutrino
            payload = {
                "user-id": "barb",  # Tu user ID
                "api-key": self.api_key,
                "content": text,  # Cambiado de 'text' a 'content'
                "censor-character": "*"
            }
            
            # Llamar a la API de Bad Word Filter (nombre correcto)
            response = requests.post(
                f"{self.base_url}/bad-word-filter",
                data=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                data = response.json()
                
                # Procesar respuesta de Neutrino
                return self._process_neutrino_response(data, text)
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "classification": "Neither",
                    "confidence": 0.0,
                    "source": "neutrino"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "API timeout",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "neutrino"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Request error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "neutrino"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "neutrino"
            }
    
    def _process_neutrino_response(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        Process Neutrino API response
        
        Args:
            data: Raw API response
            original_text: Original text that was analyzed
            
        Returns:
            Processed detection result
        """
        try:
            # Neutrino devuelve algo como:
            # {
            #     "is-bad": true/false,
            #     "censored-text": "texto con ***",
            #     "bad-words-list": ["palabra1", "palabra2"],
            #     "bad-words-total": 2
            # }
            
            is_bad = data.get('is-bad', False)
            bad_words_list = data.get('bad-words-list', [])
            bad_words_total = data.get('bad-words-total', 0)
            censored_text = data.get('censored-text', original_text)
            
            # Calcular confianza basada en el número de palabras ofensivas
            confidence = min(0.9, 0.5 + (bad_words_total * 0.2)) if is_bad else 0.1
            
            # Clasificar basado en el resultado
            if is_bad and bad_words_total > 0:
                # Si hay palabras ofensivas, clasificar como Offensive Language
                classification = "Offensive Language"
            else:
                classification = "Neither"
            
            return {
                "classification": classification,
                "confidence": confidence,
                "source": "neutrino",
                "bad_words": bad_words_list,
                "bad_words_count": bad_words_total,
                "censored_text": censored_text,
                "raw_response": data
            }
            
        except Exception as e:
            return {
                "error": f"Error processing response: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "neutrino"
            }
    
    def is_available(self) -> bool:
        """
        Check if Neutrino API is available
        
        Returns:
            True if API key is configured
        """
        return self.api_key is not None and self.api_key != "tu_api_key_de_neutrino_aqui"
    
    def get_remaining_requests(self) -> Optional[int]:
        """
        Get remaining API requests (if available in response)
        
        Returns:
            Number of remaining requests or None
        """
        # Neutrino puede incluir info sobre requests restantes
        # Esto depende de la implementación específica de la API
        return None

# Instancia global para usar en la app
neutrino_detector = NeutrinoAPIDetector()
