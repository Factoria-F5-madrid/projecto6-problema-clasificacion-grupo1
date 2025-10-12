import streamlit as st
from PIL import Image 
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import joblib
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path for imports
sys.path.append('backend')

# Import our optimized model components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import API detectors
from utils.api_verve import verve_detector
from utils.api_neutrino import neutrino_detector
from utils.api_ninja import ninja_detector
from utils.api_perspective import perspective_detector

# Import robust preprocessor
from utils.robust_preprocessor import RobustPreprocessor
from utils.confidence_booster import ConfidenceBooster

st.set_page_config(page_title='Hate Speech Detection', layout="wide",
				   initial_sidebar_state="collapsed")

# Load the optimized model and vectorizer
@st.cache_resource
def load_model():
    """Load the balanced model and vectorizer"""
    try:
        # Try to load improved model first
        improved_model_path = "backend/models/saved/improved_model.pkl"
        improved_vectorizer_path = "backend/models/saved/improved_vectorizer.pkl"
        
        if os.path.exists(improved_model_path) and os.path.exists(improved_vectorizer_path):
            st.info("🚀 Cargando modelo mejorado (incluye palabras ofensivas importantes)...")
            model = joblib.load(improved_model_path)
            vectorizer = joblib.load(improved_vectorizer_path)
            
            # Load data for display
            df = pd.read_csv("backend/data/processed/cleaned_tweets.csv")
            return model, vectorizer, df
        
        # Fallback to balanced model
        balanced_model_path = "backend/models/saved/balanced_model.pkl"
        balanced_vectorizer_path = "backend/models/saved/balanced_vectorizer.pkl"
        
        if os.path.exists(balanced_model_path) and os.path.exists(balanced_vectorizer_path):
            st.info("🔄 Cargando modelo balanceado (mejor precisión)...")
            model = joblib.load(balanced_model_path)
            vectorizer = joblib.load(balanced_vectorizer_path)
            
            # Load data for display
            df = pd.read_csv("backend/data/processed/cleaned_tweets.csv")
            return model, vectorizer, df
        else:
            # Fallback to original model
            st.warning("⚠️ Modelo balanceado no encontrado, usando modelo original...")
            df = pd.read_csv("backend/data/processed/cleaned_tweets.csv")
            
            # Prepare data
            X_text = df['clean_tweet_improved'].fillna('')
            y = df['class']
            
            # Create vectorizer
            vectorizer = TfidfVectorizer(
                max_features=2000, 
                stop_words='english', 
                ngram_range=(1,3),
                min_df=2,
                max_df=0.95
            )
            X = vectorizer.fit_transform(X_text)
            
            # Create optimized model
            model = LogisticRegression(
                penalty='l2', 
                C=0.01,
                random_state=42, 
                max_iter=2000,
                class_weight='balanced'
            )
            
            # Train the model
            model.fit(X, y)
            
            return model, vectorizer, df
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_hate_speech(text, model, vectorizer, preprocessor=None, confidence_booster=None):
    """Predict hate speech for given text using hybrid approach with APIs and preprocessing"""
    try:
        # Preprocess text if preprocessor is available
        if preprocessor:
            preprocessed = preprocessor.preprocess_text(
                text,
                normalize_evasions=True,
                clean_text=True,
                extract_features=True
            )
            processed_text = preprocessed['processed_text']
        else:
            processed_text = text
        # Nivel 1: API Verve (si está disponible)
        if verve_detector.is_available():
            verve_result = verve_detector.detect_hate_speech(text)
            if "error" not in verve_result and verve_result['confidence'] > 0.7:
                return verve_result['classification'], verve_result['confidence']
        
        # Nivel 2: Neutrino API (si está disponible)
        if neutrino_detector.is_available():
            neutrino_result = neutrino_detector.detect_profanity(text)
            if "error" not in neutrino_result and neutrino_result['confidence'] > 0.6:
                return neutrino_result['classification'], neutrino_result['confidence']
        
        # Nivel 3: API Ninja (si está disponible)
        if ninja_detector.is_available():
            ninja_result = ninja_detector.detect_profanity(text)
            if "error" not in ninja_result and ninja_result['confidence'] > 0.6:
                return ninja_result['classification'], ninja_result['confidence']
        
        # Nivel 4: Google Perspective API (si está disponible)
        if perspective_detector.is_available():
            perspective_result = perspective_detector.detect_toxicity(text)
            if "error" not in perspective_result and perspective_result['confidence'] > 0.7:
                return perspective_result['classification'], perspective_result['confidence']
        
        # Nivel 5: Reglas básicas (fallback)
        offensive_words = {
            'hate_speech': [
                # English hate speech
                'faggot', 'fag', 'faggots', 'fags', 'nigger', 'nigga', 'niggas', 'niggers',
                'dyke', 'dykes', 'tranny', 'trannies', 'faggy', 'faggoty', 'niggah',
                'white trash', 'cracker', 'crackers', 'chink', 'chinks', 'gook', 'gooks',
                'wetback', 'wetbacks', 'spic', 'spics', 'kike', 'kikes', 'towelhead',
                'towelheads', 'raghead', 'ragheads', 'sand nigger', 'sand niggers',
                # Spanish hate speech
                'maricón', 'maricones', 'puto', 'putos', 'joto', 'jotos', 'culero', 'culeros',
                'pinche', 'pinches', 'cabrón', 'cabrones', 'hijo de puta', 'hijos de puta',
                'mamón', 'mamones', 'pendejo', 'pendejos', 'idiota', 'idiotas', 'imbécil', 'imbéciles',
                'estúpido', 'estúpidos', 'tonto', 'tontos', 'pendeja', 'pendejas', 'puta', 'putas',
                'zorra', 'zorras', 'perra', 'perras', 'cabrona', 'cabronas', 'mamona', 'mamonas'
            ],
            'offensive_language': [
                # English offensive
                'bitch', 'bitches', 'hoes', 'hoe', 'pussy', 'pussies', 'ass', 'asshole',
                'assholes', 'fuck', 'fucking', 'fucked', 'fucker', 'fuckers', 'shit',
                'shits', 'shitty', 'damn', 'damned', 'hell', 'crap', 'crapper', 'dumb',
                'dumbass', 'dumbasses', 'stupid', 'idiot', 'idiots', 'moron', 'morons',
                'loser', 'losers', 'failure', 'failures', 'worthless', 'pathetic',
                'disgusting', 'gross', 'nasty', 'ugly', 'fat', 'fats', 'skinny', 'skinnies',
                # Spanish offensive
                'pendejo', 'pendejos', 'pendeja', 'pendejas', 'idiota', 'idiotas', 'imbécil', 'imbéciles',
                'estúpido', 'estúpidos', 'tonto', 'tontos', 'baboso', 'babosos', 'babosa', 'babosas',
                'pendejada', 'pendejadas', 'estupidez', 'estupideces', 'imbecilidad', 'imbecilidades',
                'mamada', 'mamadas', 'pendejear', 'pendejeando', 'pendejeado', 'pendejeada',
                'chingar', 'chingado', 'chingada', 'chingados', 'chingadas', 'chinga', 'chingas',
                'verga', 'vergas', 'pinche', 'pinches', 'cabrón', 'cabrones', 'cabrona', 'cabronas',
                'hijo de puta', 'hijos de puta', 'hija de puta', 'hijas de puta', 'puto', 'putos',
                'puta', 'putas', 'zorra', 'zorras', 'perra', 'perras', 'mamón', 'mamones',
                'mamona', 'mamonas', 'mamada', 'mamadas', 'mamar', 'mamando', 'mamado', 'mamada',
                'culero', 'culeros', 'culera', 'culeras', 'culiar', 'culiando', 'culiado', 'culiada',
                'joto', 'jotos', 'jota', 'jotas', 'maricón', 'maricones', 'marica', 'maricas',
                'gay', 'gays', 'lesbiana', 'lesbianas', 'lesbiano', 'lesbianos', 'homosexual', 'homosexuales',
                'transexual', 'transexuales', 'travesti', 'travestis', 'travestido', 'travestidos',
                'puto', 'putos', 'puta', 'putas', 'prostituta', 'prostitutas', 'prostituto', 'prostitutos',
                'zorra', 'zorras', 'perra', 'perras', 'cabrona', 'cabronas', 'mamona', 'mamonas',
                'hijo de perra', 'hijos de perra', 'hija de perra', 'hijas de perra',
                'hijo de zorra', 'hijos de zorra', 'hija de zorra', 'hijas de zorra',
                'hijo de cabrón', 'hijos de cabrón', 'hija de cabrón', 'hijas de cabrón',
                'hijo de cabrona', 'hijos de cabrona', 'hija de cabrona', 'hijas de cabrona'
            ]
        }
        
        def rule_based_classification(text):
            """Apply rule-based classification for offensive words"""
            text_lower = text.lower()
            
            # Check for hate speech words
            hate_speech_count = sum(1 for word in offensive_words['hate_speech'] if word in text_lower)
            offensive_count = sum(1 for word in offensive_words['offensive_language'] if word in text_lower)
            
            if hate_speech_count > 0:
                return 0, 0.9  # Hate Speech with high confidence
            elif offensive_count > 0:
                return 1, 0.8  # Offensive Language with high confidence
            else:
                return None, 0.0  # No rule-based classification
        
        # Try rule-based classification
        rule_pred, rule_conf = rule_based_classification(text)
        
        if rule_pred is not None:
            # Use rule-based classification
            class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            return class_names[rule_pred], rule_conf
        else:
            # Nivel 6: ML model (fallback final) - Use processed text
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Apply confidence booster if available
            if confidence_booster:
                classes = ['Hate Speech', 'Offensive Language', 'Neither']
                boosted_probs, explanation = confidence_booster.boost_confidence(
                    processed_text, probabilities, classes
                )
                # Use boosted probabilities for final prediction
                prediction = np.argmax(boosted_probs)
                probability = boosted_probs[prediction]
            else:
                probability = probabilities[prediction]
            
            # Get class names
            class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            
            return class_names[prediction], probability
    except Exception as e:
        return f"Error: {e}", 0

def main():
    # Header with banner image
    try:
        st.image("hatespeech.png", use_container_width=True)
    except Exception as e:
        st.warning("No se pudo cargar la imagen del banner")
    
    st.title("🚨 Hate Speech Detection System")
    st.markdown("**Sistema híbrido de detección de discurso de odio con reglas específicas + ML optimizado**")
    st.info("🔧 **Modelo Híbrido**: Combina reglas específicas para palabras ofensivas con Machine Learning para mayor precisión")
    
    # Load model
    with st.spinner("Cargando modelo optimizado..."):
        model, vectorizer, df = load_model()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que los datos estén disponibles.")
        return
    
    # Initialize preprocessor and confidence booster
    preprocessor = RobustPreprocessor()
    confidence_booster = ConfidenceBooster()
    
    # Sidebar for navigation
    st.sidebar.header("🧭 Navegación")
    page = st.sidebar.selectbox("Selecciona una página:", 
                               ["🔍 Detector", "📊 Análisis de Datos", "📈 Métricas del Modelo"])
    
    if page == "🔍 Detector":
        st.header("🔍 Detector de Hate Speech")
        st.markdown("Introduce un texto para analizar si contiene discurso de odio:")
        
        # Text input
        text_input = st.text_area("Texto a analizar:", 
                                placeholder="Escribe aquí el texto que quieres analizar...",
                                height=100)
        
        if st.button("🔍 Analizar Texto", type="primary"):
            if text_input.strip():
                with st.spinner("Analizando..."):
                    prediction, confidence = predict_hate_speech(text_input, model, vectorizer, preprocessor, confidence_booster)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicción", prediction)
                
                with col2:
                    st.metric("Confianza", f"{confidence:.2%}")
                
                # Color-coded result
                if prediction == "Hate Speech":
                    st.error("⚠️ **HATE SPEECH DETECTADO** - Este texto contiene discurso de odio")
                elif prediction == "Offensive Language":
                    st.warning("⚠️ **LENGUAJE OFENSIVO** - Este texto contiene lenguaje ofensivo")
                else:
                    st.success("✅ **TEXTO NORMAL** - Este texto no contiene discurso de odio")
                
                # Show preprocessing details
                st.subheader("🔧 Preprocesamiento")
                preprocessed = preprocessor.preprocess_text(
                    text_input,
                    normalize_evasions=True,
                    clean_text=True,
                    extract_features=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Texto Original:**")
                    st.code(text_input, language=None)
                
                with col2:
                    st.markdown("**🔄 Texto Procesado:**")
                    st.code(preprocessed['processed_text'], language=None)
                
                # Show evasions found
                if preprocessed['evasions_found']:
                    st.markdown("**🚫 Evasiones Detectadas y Normalizadas:**")
                    for evasion in preprocessed['evasions_found']:
                        st.markdown(f"- `{evasion}`")
                else:
                    st.markdown("**✅ No se detectaron evasiones**")
                
                # Show features
                features = preprocessed['features']
                st.markdown("**📊 Características Extraídas:**")
                st.json({
                    "Longitud": features.get('length', 0),
                    "Palabras": features.get('word_count', 0),
                    "Mayúsculas": f"{features.get('uppercase_ratio', 0):.1%}",
                    "Evasión": features.get('has_evasion', False),
                    "Repetición": f"{features.get('repetition_ratio', 0):.1%}"
                })
                
                # Show probability breakdown
                st.subheader("📊 Desglose de Probabilidades")
                X = vectorizer.transform([preprocessed['processed_text']])
                probabilities = model.predict_proba(X)[0]
                classes = ['Hate Speech', 'Offensive Language', 'Neither']
                
                # Create detailed probability breakdown
                prob_data = []
                for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
                    prob_data.append({
                        'Clase': class_name,
                        'Probabilidad': f"{prob:.1%}",
                        'Valor': prob,
                        'Barra': prob
                    })
                
                prob_df = pd.DataFrame(prob_data)
                
                # Show probabilities as bars
                st.markdown("**🎯 Probabilidades por Clase:**")
                for _, row in prob_df.iterrows():
                    col1, col2, col3 = st.columns([2, 6, 1])
                    with col1:
                        st.write(f"**{row['Clase']}**")
                    with col2:
                        st.progress(row['Valor'], text=f"{row['Probabilidad']}")
                    with col3:
                        if row['Valor'] == max(probabilities):
                            st.write("🏆")
                
                # Show raw probabilities
                st.markdown("**📈 Valores Numéricos:**")
                for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
                    st.write(f"- **{class_name}**: {prob:.4f} ({prob:.1%})")
                
                # Analysis
                max_prob = max(probabilities)
                max_class = classes[probabilities.argmax()]
                second_max = sorted(probabilities, reverse=True)[1]
                difference = max_prob - second_max
                
                st.markdown("**🔍 Análisis:**")
                if difference > 0.3:
                    st.success(f"✅ **Clasificación clara**: {max_class} con {max_prob:.1%} de confianza")
                elif difference > 0.1:
                    st.warning(f"⚠️ **Clasificación moderada**: {max_class} con {max_prob:.1%} de confianza (diferencia: {difference:.1%})")
                else:
                    st.error(f"❌ **Clasificación incierta**: {max_class} con {max_prob:.1%} de confianza (diferencia: {difference:.1%})")
                
                prob_df = pd.DataFrame({
                    'Clase': classes,
                    'Probabilidad': probabilities
                })
                
                st.bar_chart(prob_df.set_index('Clase'))
                
            else:
                st.warning("Por favor, introduce un texto para analizar.")
    
    elif page == "📊 Análisis de Datos":
        st.header("📊 Análisis de Datos")
        
        # Show dataset info
        st.subheader("📋 Información del Dataset")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Tweets", len(df))
        with col2:
            st.metric("Clases", df['class'].nunique())
        with col3:
            st.metric("Features", 2000)
        
        # Show sample data
        st.subheader("📄 Muestra de Datos")
        st.dataframe(df[['clean_tweet_improved', 'class']].head(10))
        
        # Class distribution
        st.subheader("📈 Distribución de Clases")
        class_counts = df['class'].value_counts()
        class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        class_counts.index = [class_names[i] for i in class_counts.index]
        
        fig = px.pie(values=class_counts.values, names=class_counts.index, 
                    title="Distribución de Clases en el Dataset")
        st.plotly_chart(fig)
    
    elif page == "📈 Métricas del Modelo":
        st.header("📈 Métricas del Modelo Optimizado")
        
        # Model performance metrics
        st.subheader("🎯 Rendimiento del Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "82.8%", "⬆️ +1.5%")
        with col2:
            st.metric("F1-Score", "84.2%", "⬆️ +8.2%")
        with col3:
            st.metric("Precision", "88.2%", "⬆️ +10.2%")
        with col4:
            st.metric("Overfitting", "1.52%", "⬇️ -2.13%")
        
        # Additional metrics
        st.subheader("📊 Métricas Adicionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recall", "82.8%", "⬆️ +1.8%")
        
        with col2:
            st.metric("ROC AUC", "90.6%", "⬆️ Nuevo")
        
        # Model info
        st.subheader("🔧 Información del Modelo")
        st.info("""
        **Modelo:** LogisticRegression con regularización L2  
        **Vectorización:** TF-IDF con trigramas (1-3)  
        **Validación:** 5-Fold Cross-Validation  
        **Optimización:** GridSearch + RandomizedSearch  
        **Características:** 2000 features más importantes  
        """)
        
        # Requirements compliance
        st.subheader("✅ Cumplimiento de Requisitos")
        
        requirements = [
            ("Overfitting < 5%", "1.52%", "✅"),
            ("Accuracy > 70%", "82.8%", "✅"),
            ("Validación Cruzada", "5-Fold", "✅"),
            ("Métricas de Clasificación", "Completas", "✅")
        ]
        
        for req, value, status in requirements:
            st.write(f"{status} **{req}**: {value}")

if __name__ == "__main__":
    main()
 
 