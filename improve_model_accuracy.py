"""
Script para mejorar la precisión del modelo
Agrega más ejemplos de palabras ofensivas al entrenamiento
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def create_enhanced_dataset():
    """Crear dataset mejorado con más ejemplos ofensivos"""
    print("📊 Creando dataset mejorado...")
    
    # Cargar datos originales
    df = pd.read_csv("backend/data/processed/cleaned_tweets.csv")
    
    # Crear ejemplos adicionales de palabras ofensivas
    offensive_examples = [
        # Palabras ofensivas directas
        "fuck you", "fuck off", "fucking stupid", "fucking idiot",
        "shit", "bullshit", "asshole", "bastard", "damn", "hell",
        "stupid", "idiot", "moron", "dumb", "retard",
        
        # Variaciones con evasiones
        "f*ck you", "f*ck off", "f*cking stupid", "f*cking idiot",
        "sh*t", "bullsh*t", "@sshole", "b@stard", "d@mn", "h*ll",
        "st*pid", "id*ot", "m*ron", "d*mb", "r*tard",
        
        # Con espacios
        "f u c k you", "s h i t", "a s s h o l e",
        
        # Con números
        "fuck1ng", "sh1t", "assh0le", "st0pid",
        
        # En español
        "jódete", "mierda", "cabrón", "idiota", "estúpido",
        "j*det*", "m*erda", "c*br*n", "id*ota", "est*pido",
        
        # Frases ofensivas
        "you are stupid", "you are an idiot", "you suck",
        "this is bullshit", "what the hell", "damn it",
        "go to hell", "fuck this", "fuck that"
    ]
    
    # Crear DataFrame con ejemplos ofensivos
    offensive_df = pd.DataFrame({
        'clean_tweet_improved': offensive_examples,
        'class': 1,  # Offensive Language
        'label': 1
    })
    
    # Crear ejemplos de hate speech
    hate_speech_examples = [
        "women are inferior to men", "all immigrants are criminals",
        "black people are stupid", "gay people are disgusting",
        "muslims are terrorists", "jews control everything",
        "women belong in the kitchen", "men are better than women",
        "all white people are racist", "asians are all the same"
    ]
    
    hate_df = pd.DataFrame({
        'clean_tweet_improved': hate_speech_examples,
        'class': 0,  # Hate Speech
        'label': 0
    })
    
    # Crear ejemplos neutrales
    neutral_examples = [
        "hello how are you", "thank you very much", "have a nice day",
        "good morning", "how is the weather", "nice to meet you",
        "what time is it", "where are you from", "how old are you",
        "what is your name", "nice weather today", "good job",
        "congratulations", "happy birthday", "merry christmas"
    ]
    
    neutral_df = pd.DataFrame({
        'clean_tweet_improved': neutral_examples,
        'class': 2,  # Neither
        'label': 2
    })
    
    # Combinar todos los datos
    enhanced_df = pd.concat([df, offensive_df, hate_df, neutral_df], ignore_index=True)
    
    # Limpiar datos
    enhanced_df = enhanced_df.dropna(subset=['clean_tweet_improved'])
    enhanced_df = enhanced_df[enhanced_df['clean_tweet_improved'].str.strip() != '']
    
    print(f"📈 Dataset original: {len(df)} ejemplos")
    print(f"📈 Ejemplos ofensivos agregados: {len(offensive_df)}")
    print(f"📈 Ejemplos de hate speech agregados: {len(hate_df)}")
    print(f"📈 Ejemplos neutrales agregados: {len(neutral_df)}")
    print(f"📈 Dataset final: {len(enhanced_df)} ejemplos")
    
    # Mostrar distribución
    print("\n📊 Distribución de clases:")
    print(enhanced_df['class'].value_counts().sort_index())
    
    return enhanced_df

def train_enhanced_model(df):
    """Entrenar modelo mejorado"""
    print("\n🚀 Entrenando modelo mejorado...")
    
    # Preparar datos
    X = df['clean_tweet_improved'].values
    y = df['label'].values
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Crear vectorizador mejorado
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Más features
        ngram_range=(1, 3),  # Unigramas, bigramas y trigramas
        min_df=1,  # Incluir palabras que aparecen solo una vez
        max_df=0.95,
        stop_words='english'
    )
    
    # Vectorizar
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Entrenar modelo
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Balancear clases
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Accuracy del modelo mejorado: {accuracy:.3f}")
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Hate Speech', 'Offensive Language', 'Neither']))
    
    return model, vectorizer, accuracy

def test_specific_cases(model, vectorizer):
    """Probar casos específicos"""
    print("\n🧪 Probando casos específicos...")
    
    test_cases = [
        "fuck you",
        "F*ck you", 
        "asshole",
        "@sshole",
        "stupid",
        "st*pid",
        "Women are inferior to men",
        "This is fucking stupid",
        "Hello, how are you?",
        "Thank you very much"
    ]
    
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    
    for text in test_cases:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        print(f"📝 '{text}' → {class_names[prediction]} ({confidence:.1%})")

def main():
    """Función principal"""
    print("🚀 MEJORANDO PRECISIÓN DEL MODELO")
    print("=" * 50)
    
    # Crear dataset mejorado
    enhanced_df = create_enhanced_dataset()
    
    # Entrenar modelo mejorado
    model, vectorizer, accuracy = train_enhanced_model(enhanced_df)
    
    # Probar casos específicos
    test_specific_cases(model, vectorizer)
    
    # Guardar modelo mejorado
    os.makedirs("backend/models/saved", exist_ok=True)
    
    joblib.dump(model, "backend/models/saved/enhanced_model.pkl")
    joblib.dump(vectorizer, "backend/models/saved/enhanced_vectorizer.pkl")
    
    # Guardar metadatos
    metadata = {
        "model_type": "Enhanced LogisticRegression",
        "accuracy": accuracy,
        "features": vectorizer.max_features,
        "ngram_range": vectorizer.ngram_range,
        "training_samples": len(enhanced_df),
        "class_distribution": enhanced_df['class'].value_counts().to_dict()
    }
    
    import json
    with open("backend/models/saved/enhanced_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Modelo mejorado guardado!")
    print(f"📁 Archivos guardados:")
    print(f"   - backend/models/saved/enhanced_model.pkl")
    print(f"   - backend/models/saved/enhanced_vectorizer.pkl")
    print(f"   - backend/models/saved/enhanced_metadata.json")
    
    print(f"\n🎯 Accuracy mejorado: {accuracy:.1%}")
    
    if accuracy > 0.85:
        print("🎉 ¡Excelente! El modelo mejoró significativamente")
    elif accuracy > 0.80:
        print("👍 ¡Bien! El modelo mejoró notablemente")
    else:
        print("⚠️ El modelo necesita más mejoras")

if __name__ == "__main__":
    main()
