"""
Debug script for Neutrino API - testing different user-id formats
"""

import requests

api_key = 'ZBx00tE5qDKO3FRbPqogTnoYzidLL4Tcj7FWX9zgYJ1oDDoP'

print("🔍 DEBUGGING NEUTRINO API - USER ID FORMATS")
print("=" * 60)

# Diferentes formatos de user-id a probar
user_ids_to_try = [
    "barb",
    "Barb",
    "BARB", 
    "barb@example.com",
    "barb123",
    "user-barb",
    "barb_user"
]

test_text = "Hello world"
endpoint = "https://neutrinoapi.net/bad-word-filter"

for user_id in user_ids_to_try:
    print(f"🧪 Probando User ID: '{user_id}'")
    
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
        if response.status_code == 200:
            print(f"   ✅ SUCCESS! Response: {response.text}")
            break
        else:
            print(f"   ❌ Error: {response.text[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print("-" * 40)

print("\n🔍 También probando sin user-id...")
try:
    response = requests.post(
        endpoint,
        data={
            "api-key": api_key,
            "content": test_text
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:100]}...")
except Exception as e:
    print(f"   ❌ Exception: {e}")
