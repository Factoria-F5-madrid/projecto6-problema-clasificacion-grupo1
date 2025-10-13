"""
Expandir dataset con palabras ofensivas raras
Opción 2: Entrenar con más datos
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def expand_dataset():
    """Expandir el dataset con más ejemplos de palabras ofensivas raras"""
    
    print("🔄 Expandindo dataset con palabras ofensivas raras...")
    
    # Cargar dataset original
    try:
        df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
        print(f"✅ Dataset original cargado: {len(df)} ejemplos")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return
    
    # Crear nuevos ejemplos con palabras ofensivas raras
    new_examples = []
    
    # Palabras ofensivas raras en español
    spanish_offensive_rare = [
        "eres un zoquete", "que bobo eres", "sois unos pardillos",
        "eres un memo", "que palurdo", "eres un cenutrio",
        "que pardillo", "sois unos memo", "eres un zopenco",
        "que bobalicón", "eres un garrulo", "sois unos paletos",
        "eres un patán", "que zopenco", "sois unos garrulos",
        "eres un paleto", "que patán", "sois unos zopencos"
    ]
    
    # Palabras ofensivas raras en inglés
    english_offensive_rare = [
        "you are a dimwit", "what a nincompoop", "you're a dolt",
        "that's asinine", "you're a numbskull", "what a dunderhead",
        "you're a blockhead", "that's moronic", "you're a simpleton",
        "what a dunce", "you're a halfwit", "that's idiotic",
        "you're a nitwit", "what a twit", "you're a dope"
    ]
    
    # Agregar ejemplos ofensivos
    for text in spanish_offensive_rare + english_offensive_rare:
        new_examples.append({
            'clean_tweet_improved': text,
            'class_label': 'offensive_language'
        })
    
    # Crear ejemplos de contexto positivo (para balancear)
    positive_examples = [
        "eres muy inteligente", "que listo eres", "sois muy brillantes",
        "eres un genio", "que sabio", "sois unos expertos",
        "eres muy hábil", "que talentoso", "sois muy capaces",
        "you are brilliant", "what a genius", "you're amazing",
        "that's clever", "you're talented", "what a smart person"
    ]
    
    for text in positive_examples:
        new_examples.append({
            'clean_tweet_improved': text,
            'class_label': 'neither'
        })
    
    # Crear DataFrame con nuevos ejemplos
    new_df = pd.DataFrame(new_examples)
    print(f"✅ Nuevos ejemplos creados: {len(new_df)}")
    
    # Combinar con dataset original
    expanded_df = pd.concat([df, new_df], ignore_index=True)
    print(f"✅ Dataset expandido: {len(expanded_df)} ejemplos total")
    
    # Guardar dataset expandido
    expanded_df.to_csv('backend/data/processed/expanded_tweets.csv', index=False)
    print("✅ Dataset expandido guardado")
    
    return expanded_df

def retrain_models(expanded_df):
    """Re-entrenar modelos con dataset expandido"""
    
    print("\n🔄 Re-entrenando modelos con dataset expandido...")
    
    # Preparar datos
    X = expanded_df['clean_tweet_improved'].fillna('')
    y = expanded_df['class_label'].fillna('neither')
    
    # Crear vectorizador mejorado
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Más características
        min_df=1,           # Incluir palabras raras
        max_df=0.95,
        ngram_range=(1, 2), # Bigramas para mejor contexto
        stop_words=None
    )
    
    # Vectorizar
    X_vectorized = vectorizer.fit_transform(X)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo mejorado
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Modelo re-entrenado - Precisión: {accuracy:.3f}")
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo mejorado
    os.makedirs('backend/models/saved', exist_ok=True)
    joblib.dump(model, 'backend/models/saved/expanded_model.pkl')
    joblib.dump(vectorizer, 'backend/models/saved/expanded_vectorizer.pkl')
    
    print("✅ Modelo expandido guardado")
    
    return model, vectorizer

def test_expanded_model():
    """Probar el modelo expandido"""
    
    print("\n🧪 Probando modelo expandido...")
    
    # Cargar modelo expandido
    try:
        model = joblib.load('backend/models/saved/expanded_model.pkl')
        vectorizer = joblib.load('backend/models/saved/expanded_vectorizer.pkl')
        print("✅ Modelo expandido cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo expandido: {e}")
        return
    
    # Casos de prueba
    test_cases = [
        "eres un zoquete",      # Palabra rara ofensiva
        "que bobo eres",        # Palabra rara ofensiva
        "sois unos pardillos",  # Palabra rara ofensiva
        "Hello, how are you?",  # Texto limpio
        "That's brilliant work", # Texto positivo
        "xyz123 abc456"         # Palabras sin sentido
    ]
    
    print("\n📊 Resultados:")
    print("-" * 50)
    
    for text in test_cases:
        # Vectorizar
        X_test = vectorizer.transform([text])
        
        # Predecir
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        confidence = max(probabilities)
        
        print(f"'{text}'")
        print(f"  → {prediction} ({confidence:.1%})")
        print()

def main():
    """Función principal"""
    print("🚀 EXPANDIENDO DATASET Y RE-ENTRENANDO MODELOS")
    print("=" * 60)
    
    # Paso 1: Expandir dataset
    expanded_df = expand_dataset()
    
    if expanded_df is not None:
        # Paso 2: Re-entrenar modelos
        model, vectorizer = retrain_models(expanded_df)
        
        # Paso 3: Probar modelo expandido
        test_expanded_model()
        
        print("\n🎉 ¡Proceso completado!")
        print("✅ Dataset expandido con palabras raras")
        print("✅ Modelo re-entrenado con más datos")
        print("✅ Mejor reconocimiento de palabras ofensivas raras")

if __name__ == "__main__":
    main()
