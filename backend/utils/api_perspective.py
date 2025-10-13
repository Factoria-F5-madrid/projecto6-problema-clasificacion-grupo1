"""
Google Perspective API integration for toxicity detection
"""

import requests
import os
from typing import Dict, Any, Optional

class PerspectiveAPIDetector:
    """Google Perspective API detector for toxicity and hate speech"""
    
    def __init__(self):
        self.api_key = os.getenv('PERSPECTIVE_API_KEY')
        self.base_url = "https://commentanalyzer.googleapis.com/v1alpha1"
        self.timeout = 10  # 10 segundos timeout
    
    def detect_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Detect toxicity using Google Perspective API
        
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
                "source": "perspective"
            }
        
        try:
            # Preparar URL con API key
            url = f"{self.base_url}/comments:analyze?key={self.api_key}"
            
            # Preparar payload
            payload = {
                "comment": {
                    "text": text
                },
                "requestedAttributes": {
                    "TOXICITY": {},
                    "SEVERE_TOXICITY": {},
                    "IDENTITY_ATTACK": {},
                    "INSULT": {},
                    "PROFANITY": {},
                    "THREAT": {}
                },
                "languages": ["en", "es"]  # English and Spanish
            }
            
            # Llamar a la API
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                data = response.json()
                return self._process_perspective_response(data, text)
            else:
                return {
                    "error": f"API error: {response.status_code} - {response.text}",
                    "classification": "Neither",
                    "confidence": 0.0,
                    "source": "perspective"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "API timeout",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "perspective"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Request error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "perspective"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "perspective"
            }
    
    def _process_perspective_response(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        Process Google Perspective API response
        
        Args:
            data: Raw API response
            original_text: Original text that was analyzed
            
        Returns:
            Processed detection result
        """
        try:
            # Extraer atributos de la respuesta
            attributes = data.get('attributeScores', {})
            
            # Obtener scores de toxicidad
            toxicity_score = attributes.get('TOXICITY', {}).get('summaryScore', {}).get('value', 0.0)
            severe_toxicity = attributes.get('SEVERE_TOXICITY', {}).get('summaryScore', {}).get('value', 0.0)
            identity_attack = attributes.get('IDENTITY_ATTACK', {}).get('summaryScore', {}).get('value', 0.0)
            insult = attributes.get('INSULT', {}).get('summaryScore', {}).get('value', 0.0)
            profanity = attributes.get('PROFANITY', {}).get('summaryScore', {}).get('value', 0.0)
            threat = attributes.get('THREAT', {}).get('summaryScore', {}).get('value', 0.0)
            
            # Calcular score máximo
            max_score = max(toxicity_score, severe_toxicity, identity_attack, insult, profanity, threat)
            
            # Clasificar basado en los scores
            if max_score >= 0.8:
                classification = "Hate Speech"
                confidence = max_score
            elif max_score >= 0.5:
                classification = "Offensive Language"
                confidence = max_score
            else:
                classification = "Neither"
                confidence = 1.0 - max_score
            
            # Detectar el tipo principal de toxicidad
            toxicity_types = {
                "TOXICITY": toxicity_score,
                "SEVERE_TOXICITY": severe_toxicity,
                "IDENTITY_ATTACK": identity_attack,
                "INSULT": insult,
                "PROFANITY": profanity,
                "THREAT": threat
            }
            
            main_toxicity = max(toxicity_types.items(), key=lambda x: x[1])
            
            return {
                "classification": classification,
                "confidence": confidence,
                "source": "perspective",
                "toxicity_score": toxicity_score,
                "severe_toxicity": severe_toxicity,
                "identity_attack": identity_attack,
                "insult": insult,
                "profanity": profanity,
                "threat": threat,
                "main_toxicity_type": main_toxicity[0],
                "main_toxicity_score": main_toxicity[1],
                "raw_response": data
            }
            
        except Exception as e:
            return {
                "error": f"Error processing response: {str(e)}",
                "classification": "Neither",
                "confidence": 0.0,
                "source": "perspective"
            }
    
    def is_available(self) -> bool:
        """
        Check if Google Perspective API is available
        
        Returns:
            True if API key is configured
        """
        return self.api_key is not None and self.api_key != "tu_google_api_key_aqui"
    
    def get_remaining_requests(self) -> Optional[int]:
        """
        Get remaining API requests (if available in response)
        
        Returns:
            Number of remaining requests or None
        """
        # Google Perspective puede incluir info sobre requests restantes en headers
        return None

# Instancia global para usar en la app
perspective_detector = PerspectiveAPIDetector()
