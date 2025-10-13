# ===========================================
# APLICACIÓN PRINCIPAL PARA STREAMLIT CLOUD
# ===========================================

import streamlit as st
import sys
import os

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar la aplicación organizada
from frontend.apps.app_organized import main

if __name__ == "__main__":
    # Configurar la página
    st.set_page_config(
        page_title="Hate Speech Detector",
        page_icon="🚫",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ejecutar la aplicación principal
    main()
