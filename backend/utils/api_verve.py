"""
API Verve integration for hate speech detection
"""

import requests
import os
from typing import Dict, Any, Optional

class VerveAPIDetector:
    """API Verve detector for hate speech"""
    
    def __init__(self):
        self.api_key = os.getenv('VERVE_API_KEY')
        self.base_url = "https://api.apiverve.com/v1/profanityfilter"  # URL correcta
        self.timeout = 10  # 10 segundos timeout
    
    def detect_hate_speech(self, text: str) -> Dict[str, Any]:
        """
        Detect hate speech using API Verve
        
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
                "source": "verve"
            }
        
        try:
            # Preparar headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Preparar payload según documentación de API Verve
            payload = {
                "text": text,
                "mask": "*"  # Caracter para enmascarar palabras ofensivas
            }
            
            # Llamar a la API (URL ya incluye el endpoint)
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                data = response.json()
                
                # Procesar respuesta de Verve
                return self._process_verve_response(data)
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "classification": "Neither",
                    "confidence": 0.0,
                    "source": "verve"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "API timeout",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "verve"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Request error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "verve"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "verve"
            }
    
    def _process_verve_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Verve API response
        
        Args:
            data: Raw API response
            
        Returns:
            Processed detection result
        """
        try:
            # Formato real de API Verve según la imagen:
            # {
            #     "status": "ok",
            #     "error": null,
            #     "data": {
            #         "isProfane": true,
            #         "filteredText": "Today is so **** hot! Why the **** would anyone go outside?",
            #         "mask": "*",
            #         "trimmed": false,
            #         "profaneWords": 2
            #     }
            # }
            
            if data.get('status') != 'ok':
                return {
                    "error": f"API error: {data.get('error', 'Unknown error')}",
                    "classification": "Neither",
                    "confidence": 0.0,
                    "source": "verve"
                }
            
            # Extraer datos de la respuesta
            api_data = data.get('data', {})
            is_profane = api_data.get('isProfane', False)
            profane_words_count = api_data.get('profaneWords', 0)
            filtered_text = api_data.get('filteredText', '')
            
            # Calcular confianza basada en el número de palabras ofensivas
            confidence = min(0.9, 0.5 + (profane_words_count * 0.2)) if is_profane else 0.1
            
            # Clasificar basado en el resultado
            if is_profane and profane_words_count > 0:
                classification = "Offensive Language"
            else:
                classification = "Neither"
            
            return {
                "classification": classification,
                "confidence": confidence,
                "source": "verve",
                "method": "API Verve",
                "profane_words_count": profane_words_count,
                "filtered_text": filtered_text,
                "raw_response": data
            }
            
        except Exception as e:
            return {
                "error": f"Error processing response: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "verve"
            }
    
    def is_available(self) -> bool:
        """
        Check if API Verve is available
        
        Returns:
            True if API key is configured
        """
        return self.api_key is not None and self.api_key != "tu_api_key_de_verve_aqui" and len(self.api_key) > 10

# Instancia global para usar en la app
verve_detector = VerveAPIDetector()
