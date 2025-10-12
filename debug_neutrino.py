"""
Debug script for Neutrino API
"""

import requests
import os

# Configurar API key
api_key = 'ZBx00tE5qDKO3FRbPqogTnoYzidLL4Tcj7FWX9zgYJ1oDDoP'
user_id = 'barb'

print("🔍 DEBUGGING NEUTRINO API")
print("=" * 50)
print(f"API Key: {api_key[:10]}...")
print(f"User ID: {user_id}")
print()

# Probar diferentes endpoints y formatos
endpoints_to_try = [
    "https://neutrinoapi.net/bad-word-filter",
    "https://neutrinoapi.net/profanity-filter",
    "https://neutrinoapi.net/text-filter"
]

test_text = "Hello world"

for endpoint in endpoints_to_try:
    print(f"🧪 Probando endpoint: {endpoint}")
    
    # Formato 1: application/x-www-form-urlencoded
    try:
        response = requests.post(
            endpoint,
            data={
                "user-id": user_id,
                "api-key": api_key,
                "content": test_text
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Formato 2: JSON
    try:
        response = requests.post(
            endpoint,
            json={
                "user-id": user_id,
                "api-key": api_key,
                "content": test_text
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("-" * 50)
