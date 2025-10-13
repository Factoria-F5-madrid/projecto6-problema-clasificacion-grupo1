
"""
App Simplificada - Detector de Hate Speech
Versión limpia y fácil de usar
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from PIL import Image

# Add backend to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'backend'))

# Import our ultimate system
from models.ultimate_hybrid_system import UltimateHybridSystem

# Page config
st.set_page_config(
    page_title='Detector de Hate Speech',
    page_icon='🚨',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Load the best system
@st.cache_resource
def load_best_system():
    """Carga el mejor sistema disponible"""
    try:
        system = UltimateHybridSystem()
        return system
    except Exception as e:
        st.error(f"Error cargando el sistema: {e}")
        return None

def main():
    """Función principal de la app"""
    
    # Header
    st.title("🚨 Detector de Hate Speech")
    st.markdown("**Sistema híbrido definitivo con 93.8% de precisión**")
    
    # Load system
    with st.spinner("🔄 Cargando sistema..."):
        system = load_best_system()
    
    if system is None:
        st.error("❌ No se pudo cargar el sistema")
        return
    
    st.success("✅ Sistema cargado correctamente")
    
    # Main detector
    st.markdown("---")
    st.header("🔍 Analizar Texto")
    
    # Input
    text_input = st.text_area(
        "Ingresa el texto a analizar:",
        placeholder="Escribe aquí el texto que quieres analizar...",
        height=100
    )
    
    # Analyze button
    if st.button("🚀 Analizar", type="primary", use_container_width=True):
        if text_input.strip():
            analyze_text(system, text_input)
        else:
            st.warning("⚠️ Por favor, ingresa algún texto para analizar")
    
    # Quick test cases
    st.markdown("---")
    st.header("🧪 Casos de Prueba Rápidos")
    
    test_cases = [
        ("fuck you", "Offensive Language"),
        ("Hello, how are you?", "Neither"),
        ("Women are inferior to men", "Hate Speech"),
        ("This is fucking amazing", "Neither"),
        ("You are a jerk", "Offensive Language"),
        ("That's brilliant work", "Neither"),
        ("eres un zoquete", "Offensive Language"),
        ("que bobo eres", "Offensive Language"),
        ("sois unos pardillos", "Offensive Language")
    ]
    
    cols = st.columns(3)
    for i, (text, expected) in enumerate(test_cases):
        with cols[i % 3]:
            if st.button(f"📝 {text[:20]}...", key=f"test_{i}"):
                analyze_text(system, text)

def analyze_text(system, text):
    """Analiza el texto y muestra los resultados"""
    
    with st.spinner("🔄 Analizando..."):
        result = system.predict(text)
    
    # Results
    st.markdown("---")
    st.header("📊 Resultados del Análisis")
    
    # Classification
    classification = result['prediction']
    confidence = result['confidence']
    method = result['method']
    
    # Color coding
    if classification == "Hate Speech":
        color = "🔴"
        bg_color = "#ffebee"
    elif classification == "Offensive Language":
        color = "🟡"
        bg_color = "#fff3e0"
    else:
        color = "🟢"
        bg_color = "#e8f5e8"
    
    # Display result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Clasificación", f"{color} {classification}")
    
    with col2:
        st.metric("Confianza", f"{confidence:.1%}")
    
    with col3:
        st.metric("Método", method.replace('_', ' ').title())
    
    # Explanation
    st.markdown("**💡 Explicación:**")
    st.info(result['explanation'])
    
    # Detailed breakdown (collapsible)
    with st.expander("🔍 Análisis Detallado"):
        st.json({
            "Texto Original": text,
            "Predicción": classification,
            "Confianza": f"{confidence:.1%}",
            "Método": method,
            "Explicación": result['explanation']
        })

if __name__ == "__main__":
    main()
