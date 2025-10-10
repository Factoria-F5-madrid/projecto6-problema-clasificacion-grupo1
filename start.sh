#!/bin/bash
# Start script for Render deployment

# Get port from environment variable or use default
PORT=${PORT:-8501}

# Start Streamlit with the correct port
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
