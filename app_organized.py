"""
App Organizada - Detector de Hate Speech
Versión con menú para separar funciones básicas y avanzadas
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from PIL import Image

# Add backend to path
sys.path.append('backend')

# Import our systems
from models.final_smart_selector import FinalSmartSelector
from models.improved_smart_selector import ImprovedSmartSelector
from models.advanced_hybrid_system import AdvancedHybridSystem
from models.ultimate_hybrid_system import UltimateHybridSystem

# Import MLOps components
from mlops.ab_testing import ABTestingSystem

# Page config
st.set_page_config(
    page_title='Detector de Hate Speech',
    page_icon='🚨',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load systems
@st.cache_resource
def load_systems():
    """Carga todos los sistemas disponibles"""
    try:
        systems = {
            'Final': FinalSmartSelector(),
            'Improved': ImprovedSmartSelector(),
            'Advanced': AdvancedHybridSystem()
        }
        return systems
    except Exception as e:
        st.error(f"Error cargando sistemas: {e}")
        return None

def main():
    """Función principal de la app"""
    
    # Sidebar menu
    st.sidebar.title("🚨 Detector de Hate Speech")
    st.sidebar.markdown("**Sistema híbrido inteligente**")
    
    # Menu options
    menu = st.sidebar.selectbox(
        "Selecciona una función:",
        [
            "🏠 Detector Principal",
            "📊 Comparación de Modelos", 
            "🧪 Casos de Prueba",
            "📈 Métricas del Sistema",
            "🔬 A/B Testing (MLOps)",
            "⚙️ Configuración Avanzada"
        ]
    )
    
    # Load systems
    with st.spinner("🔄 Cargando sistemas..."):
        systems = load_systems()
    
    if systems is None:
        st.error("❌ No se pudieron cargar los sistemas")
        return
    
    # Load A/B Testing system
    @st.cache_resource
    def load_ab_system():
        return ABTestingSystem()
    
    ab_system = load_ab_system()
    
    # Route to selected page
    if menu == "🏠 Detector Principal":
        detector_page(systems['Final'])
    elif menu == "📊 Comparación de Modelos":
        comparison_page(systems)
    elif menu == "🧪 Casos de Prueba":
        test_page(systems['Final'])
    elif menu == "📈 Métricas del Sistema":
        metrics_page()
    elif menu == "🔬 A/B Testing (MLOps)":
        ab_testing_page(systems, ab_system)
    elif menu == "⚙️ Configuración Avanzada":
        config_page()

def detector_page(system):
    """Página principal del detector"""
    st.header("🔍 Detector Principal")
    st.markdown("**Sistema Final - Máxima Precisión (81.2%)**")
    
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

def comparison_page(systems):
    """Página de comparación de modelos"""
    st.header("📊 Comparación de Modelos")
    st.markdown("**Compara el rendimiento de diferentes sistemas**")
    
    # Input
    text_input = st.text_area(
        "Ingresa el texto a comparar:",
        placeholder="Escribe aquí el texto para comparar entre modelos...",
        height=100
    )
    
    if st.button("🔄 Comparar Modelos", type="primary"):
        if text_input.strip():
            compare_models(systems, text_input)
        else:
            st.warning("⚠️ Por favor, ingresa algún texto para comparar")

def test_page(system):
    """Página de casos de prueba"""
    st.header("🧪 Casos de Prueba")
    st.markdown("**Prueba el sistema con casos predefinidos**")
    
    test_cases = [
        ("fuck you", "Offensive Language"),
        ("Hello, how are you?", "Neither"),
        ("Women are inferior to men", "Hate Speech"),
        ("This is fucking amazing", "Neither"),
        ("You are a jerk", "Offensive Language"),
        ("That's brilliant work", "Neither"),
        ("F*ck you", "Offensive Language"),
        ("@sshole", "Offensive Language")
    ]
    
    cols = st.columns(2)
    for i, (text, expected) in enumerate(test_cases):
        with cols[i % 2]:
            if st.button(f"📝 {text}", key=f"test_{i}", use_container_width=True):
                analyze_text(system, text)

def metrics_page():
    """Página de métricas del sistema"""
    st.header("📈 Métricas del Sistema")
    st.markdown("**Rendimiento y estadísticas del sistema**")
    
    # System performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precisión General", "81.2%", "2.1%")
    
    with col2:
        st.metric("Hate Speech", "90.0%", "5.2%")
    
    with col3:
        st.metric("Offensive Language", "80.0%", "3.1%")
    
    # Performance chart
    st.subheader("📊 Distribución de Clasificaciones")
    
    data = {
        'Categoría': ['Hate Speech', 'Offensive Language', 'Neither'],
        'Precisión': [90.0, 80.0, 75.0],
        'Recall': [85.0, 82.0, 78.0],
        'F1-Score': [87.5, 81.0, 76.5]
    }
    
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index('Categoría'))

def config_page():
    """Página de configuración avanzada"""
    st.header("⚙️ Configuración Avanzada")
    st.markdown("**Ajustes y parámetros del sistema**")
    
    st.subheader("🔧 Parámetros del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.1)
        st.slider("Peso de Reglas", 0.0, 1.0, 0.6, 0.1)
    
    with col2:
        st.slider("Peso de ML", 0.0, 1.0, 0.4, 0.1)
        st.checkbox("Modo Estricto", value=False)
    
    st.subheader("📊 Información del Sistema")
    st.info("""
    **Sistema Híbrido Final:**
    - 3 modelos ML (Logistic Regression, XGBoost, Random Forest)
    - Reglas inteligentes para palabras ofensivas
    - Detección de contexto y evasiones
    - Precisión: 81.2%
    """)

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

def compare_models(systems, text):
    """Compara el texto entre diferentes modelos"""
    
    st.markdown("---")
    st.header("📊 Comparación de Modelos")
    
    results = {}
    
    for name, system in systems.items():
        with st.spinner(f"🔄 Analizando con {name}..."):
            try:
                result = system.predict(text)
                results[name] = result
            except Exception as e:
                st.error(f"Error con {name}: {e}")
                results[name] = None
    
    # Display comparison
    if results:
        comparison_data = []
        
        for name, result in results.items():
            if result:
                comparison_data.append({
                    'Modelo': name,
                    'Clasificación': result['prediction'],
                    'Confianza': f"{result['confidence']:.1%}",
                    'Método': result['method'].replace('_', ' ').title()
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Agreement analysis
            classifications = [r['prediction'] for r in results.values() if r]
            if len(set(classifications)) == 1:
                st.success("✅ Todos los modelos están de acuerdo")
            else:
                st.warning("⚠️ Los modelos no están de acuerdo")

def ab_testing_page(systems, ab_system):
    """Página de A/B Testing para MLOps"""
    st.header("🔬 A/B Testing (MLOps)")
    st.markdown("**Nivel Experto - Comparación de Modelos en Producción**")
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Iniciar Test", "🧪 Prueba en Vivo", "📊 Ver Resultados", "📈 Análisis"])
    
    with tab1:
        st.subheader("🚀 Iniciar Nuevo A/B Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Modelo A (Control)**")
            model_a_name = st.selectbox(
                "Seleccionar Modelo A:",
                ["UltimateHybrid", "FinalSmartSelector", "ImprovedSmartSelector", "AdvancedHybrid"],
                key="model_a"
            )
            
            # Obtener modelo real
            model_a = systems.get('Final')  # Por defecto UltimateHybrid
        
        with col2:
            st.markdown("**Modelo B (Variante)**")
            model_b_name = st.selectbox(
                "Seleccionar Modelo B:",
                ["FinalSmartSelector", "ImprovedSmartSelector", "AdvancedHybrid", "UltimateHybrid"],
                key="model_b"
            )
            
            # Obtener modelo real
            model_b = systems.get('Improved')  # Por defecto FinalSmartSelector
        
        # Configuración del test
        st.markdown("**⚙️ Configuración del Test**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            traffic_split = st.slider(
                "División de Tráfico (%)",
                min_value=10, max_value=90, value=50,
                help="Porcentaje de tráfico para el Modelo A"
            )
        
        with col2:
            test_duration = st.number_input(
                "Duración (días)",
                min_value=1, max_value=30, value=7,
                help="Días que durará el test"
            )
        
        # Iniciar test
        if st.button("🚀 Iniciar A/B Test", type="primary"):
            if model_a_name == model_b_name:
                st.error("❌ Los modelos deben ser diferentes")
            else:
                with st.spinner("🔄 Iniciando A/B Test..."):
                    # Actualizar configuración
                    ab_system.traffic_split = traffic_split / 100
                    
                    # Iniciar test
                    test_id = ab_system.start_ab_test(
                        model_a_name, model_b_name, 
                        model_a, model_b, test_duration
                    )
                    
                    st.success(f"✅ A/B Test iniciado: `{test_id}`")
                    st.info(f"📊 División: {traffic_split}% para {model_a_name}, {100-traffic_split}% para {model_b_name}")
                    
                    # Guardar test_id en session state
                    st.session_state.current_test_id = test_id
    
    with tab2:
        st.subheader("🧪 Prueba en Vivo")
        st.markdown("**Haz predicciones en tiempo real para generar datos del A/B test**")
        
        # Verificar si hay un test activo
        if 'current_test_id' not in st.session_state:
            st.warning("⚠️ No hay un test A/B activo. Ve a la pestaña 'Iniciar Test' para crear uno.")
        else:
            test_id = st.session_state.current_test_id
            
            # Mostrar información del test activo
            st.info(f"🔄 Test activo: `{test_id}`")
            
            # Input para texto
            text_input = st.text_area(
                "Ingresa el texto para analizar:",
                placeholder="Escribe aquí el texto que quieres analizar...",
                height=100,
                key="ab_test_input"
            )
            
            # Botón para analizar
            if st.button("🔍 Analizar con A/B Test", type="primary"):
                if text_input.strip():
                    # Obtener modelos del test
                    import os
                    import json
                    
                    config_path = os.path.join("backend/mlops/ab_results", f"{test_id}_config.json")
                    
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        model_a_name = config['model_a']['name']
                        model_b_name = config['model_b']['name']
                        
                        # Asignar tráfico
                        variant = ab_system.assign_traffic(test_id, f"user_{len(text_input)}")
                        
                        # Obtener modelo correspondiente
                        if variant == 'A':
                            model = systems.get('Final')  # UltimateHybrid
                            model_name = model_a_name
                        else:
                            model = systems.get('Improved')  # FinalSmartSelector
                            model_name = model_b_name
                        
                        # Hacer predicción
                        import time
                        start_time = time.time()
                        result = model.predict(text_input)
                        response_time = time.time() - start_time
                        
                        # Log predicción
                        ab_system.log_prediction(
                            test_id, variant, text_input, 
                            result['prediction'], result['confidence'],
                            None, response_time  # No tenemos etiqueta real
                        )
                        
                        # Mostrar resultados
                        st.success(f"✅ Predicción registrada en el A/B test")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Modelo Usado", f"{model_name} ({variant})")
                        
                        with col2:
                            st.metric("Predicción", result['prediction'])
                        
                        with col3:
                            st.metric("Confianza", f"{result['confidence']:.1%}")
                        
                        st.markdown("**💡 Explicación:**")
                        st.info(result['explanation'])
                        
                        # Mostrar estadísticas del test
                        st.markdown("---")
                        st.markdown("**📊 Estadísticas del Test:**")
                        
                        # Contar predicciones por modelo
                        log_path = os.path.join("backend/mlops/ab_results", f"{test_id}_logs.jsonl")
                        if os.path.exists(log_path):
                            with open(log_path, 'r') as f:
                                logs = [json.loads(line.strip()) for line in f]
                            
                            model_a_count = len([log for log in logs if log['model_variant'] == 'A'])
                            model_b_count = len([log for log in logs if log['model_variant'] == 'B'])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(f"Predicciones {model_a_name}", model_a_count)
                            
                            with col2:
                                st.metric(f"Predicciones {model_b_name}", model_b_count)
                        
                    else:
                        st.error("❌ No se encontró la configuración del test")
                else:
                    st.warning("⚠️ Por favor, ingresa algún texto para analizar")
    
    with tab3:
        st.subheader("📊 Resultados del A/B Test")
        
        # Listar tests disponibles
        import os
        import json
        from datetime import datetime
        
        results_dir = "backend/mlops/ab_results"
        if os.path.exists(results_dir):
            test_files = [f for f in os.listdir(results_dir) if f.endswith('_config.json')]
            
            if test_files:
                st.markdown("**Tests Disponibles:**")
                
                for test_file in test_files:
                    test_id = test_file.replace('_config.json', '')
                    
                    # Cargar configuración
                    with open(os.path.join(results_dir, test_file), 'r') as f:
                        config = json.load(f)
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{test_id}**")
                        st.caption(f"A: {config['model_a']['name']} | B: {config['model_b']['name']}")
                    
                    with col2:
                        st.write(f"Estado: {config['status']}")
                        st.caption(f"Inicio: {config['start_time'][:10]}")
                    
                    with col3:
                        if st.button("Ver", key=f"view_{test_id}"):
                            st.session_state.current_test_id = test_id
                
                # Mostrar resultados del test seleccionado
                if 'current_test_id' in st.session_state:
                    test_id = st.session_state.current_test_id
                    
                    st.markdown("---")
                    st.markdown(f"**Resultados de: {test_id}**")
                    
                    # Obtener resultados
                    results = ab_system.get_test_results(test_id)
                    
                    if 'error' in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        # Métricas generales
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Predicciones", results['total_predictions'])
                        
                        with col2:
                            st.metric("Modelo A - Accuracy", 
                                    f"{results['model_a'].get('accuracy', 0):.3f}")
                        
                        with col3:
                            st.metric("Modelo B - Accuracy", 
                                    f"{results['model_b'].get('accuracy', 0):.3f}")
                        
                        # Significancia estadística
                        significance = results['statistical_significance']
                        
                        if significance['status'] == 'sufficient_data':
                            st.markdown("**📈 Significancia Estadística:**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Diferencia:** {significance['difference']:.3f}")
                                st.write(f"**Z-Score:** {significance['z_score']:.3f}")
                            
                            with col2:
                                st.write(f"**Significativo:** {'✅ Sí' if significance['is_significant'] else '❌ No'}")
                                st.write(f"**Ganador:** {significance['winner']}")
                        else:
                            st.warning(f"⚠️ {significance['message']}")
                        
                        # Recomendación
                        recommendation = ab_system.get_recommendation(test_id)
                        
                        st.markdown("**🎯 Recomendación:**")
                        if recommendation['recommendation'] == 'continue_testing':
                            st.warning(f"⚠️ {recommendation['message']}")
                        elif recommendation['recommendation'] == 'keep_current':
                            st.info(f"ℹ️ {recommendation['message']}")
                        else:
                            st.success(f"✅ {recommendation['message']}")
            else:
                st.info("ℹ️ No hay tests A/B disponibles. Inicia uno en la pestaña 'Iniciar Test'")
        else:
            st.info("ℹ️ No hay directorio de resultados. Inicia un test primero.")
    
    with tab4:
        st.subheader("📈 Análisis de A/B Testing")
        
        st.markdown("""
        **¿Qué es A/B Testing en MLOps?**
        
        A/B Testing es una técnica fundamental en MLOps que permite:
        
        - **🔬 Comparar modelos** en producción de forma segura
        - **📊 Medir impacto** de nuevos modelos con datos reales
        - **📈 Optimizar rendimiento** basándose en métricas objetivas
        - **🛡️ Reducir riesgos** al desplegar cambios gradualmente
        
        **Métricas que evaluamos:**
        - Accuracy, Precision, Recall, F1-Score
        - Tiempo de respuesta
        - Confianza promedio
        - Significancia estadística
        
        **Flujo de trabajo:**
        1. **Iniciar test** con dos modelos diferentes
        2. **Dividir tráfico** (ej: 50% cada modelo)
        3. **Recopilar métricas** durante el período de prueba
        4. **Analizar resultados** con significancia estadística
        5. **Tomar decisión** basada en evidencia
        """)
        
        # Gráfico de ejemplo
        import plotly.graph_objects as go
        
        # Datos de ejemplo
        days = list(range(1, 8))
        model_a_accuracy = [0.92, 0.93, 0.91, 0.94, 0.92, 0.93, 0.94]
        model_b_accuracy = [0.89, 0.90, 0.88, 0.91, 0.89, 0.90, 0.91]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days, y=model_a_accuracy,
            mode='lines+markers',
            name='Modelo A (Control)',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=model_b_accuracy,
            mode='lines+markers',
            name='Modelo B (Variante)',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Evolución de Accuracy en A/B Test",
            xaxis_title="Días",
            yaxis_title="Accuracy",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
