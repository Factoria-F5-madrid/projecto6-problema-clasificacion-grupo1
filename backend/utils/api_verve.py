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
        self.base_url = "https://api.verve.com/v1"  # URL base de Verve
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
            
            # Preparar payload
            payload = {
                "text": text,
                "language": "auto",  # Detección automática de idioma
                "include_confidence": True
            }
            
            # Llamar a la API
            response = requests.post(
                f"{self.base_url}/hate-speech",
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
            # Mapear respuesta de Verve a nuestro formato
            # (Esto puede variar según la estructura real de la API)
            
            # Asumir que Verve devuelve algo como:
            # {
            #     "is_hate_speech": true/false,
            #     "confidence": 0.85,
            #     "category": "hate_speech" o "offensive" o "neither"
            # }
            
            is_hate_speech = data.get('is_hate_speech', False)
            confidence = data.get('confidence', 0.0)
            category = data.get('category', 'neither')
            
            # Mapear categorías
            if is_hate_speech or category in ['hate_speech', 'hate']:
                classification = "Hate Speech"
            elif category in ['offensive', 'profanity', 'inappropriate']:
                classification = "Offensive Language"
            else:
                classification = "Neither"
            
            return {
                "classification": classification,
                "confidence": confidence,
                "source": "verve",
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
        return self.api_key is not None and self.api_key != "tu_api_key_de_verve_aqui"

# Instancia global para usar en la app
verve_detector = VerveAPIDetector()
