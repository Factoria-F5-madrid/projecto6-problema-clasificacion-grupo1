#!/usr/bin/env python3
"""
Start script for Render deployment
"""

import os
import subprocess
import sys

def main():
    # Get port from environment variable
    port = os.environ.get('PORT', '8501')
    
    # Start Streamlit
    cmd = [
        'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true'
    ]
    
    print(f"Starting Streamlit on port {port}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
