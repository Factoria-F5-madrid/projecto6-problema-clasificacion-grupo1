#!/usr/bin/env python3
"""
Optimized start script for Render deployment
"""

import os
import subprocess
import sys

def main():
    # Get port from environment variable
    port = os.environ.get('PORT', '8501')
    
    print(f"Starting Streamlit on port {port}...")
    print(f"Environment: {os.environ.get('RENDER', 'local')}")
    
    # Start Streamlit with optimized settings for Render
    cmd = [
        'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--browser.gatherUsageStats', 'false',
        '--server.runOnSave', 'false',
        '--server.fileWatcherType', 'none'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
