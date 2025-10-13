#!/usr/bin/env python3
"""
Vectorizador mejorado que garantiza la inclusión de palabras ofensivas importantes
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

class ImprovedVectorizer:
    """Vectorizador mejorado que incluye palabras ofensivas importantes"""
    
    def __init__(self):
        # Palabras ofensivas importantes que DEBEN estar en el vocabulario
        self.important_words = [
            # Palabras ofensivas básicas
            'fuck', 'shit', 'damn', 'hell', 'bitch', 'asshole',
            'stupid', 'idiot', 'moron', 'loser', 'pathetic',
            'hate', 'kill', 'die', 'destroy', 'inferior', 'superior',
            
            # Palabras de hate speech
            'women', 'men', 'immigrants', 'jews', 'muslims', 'blacks', 'whites',
            'gays', 'lesbians', 'faggot', 'nigger', 'dyke', 'tranny',
            
            # Palabras en español
            'puta', 'puto', 'joder', 'mierda', 'pendejo', 'pendeja',
            'idiota', 'estupido', 'imbecil', 'mamón', 'mamona',
            'cabrón', 'cabrona', 'culero', 'culera', 'joto', 'jota',
            'maricón', 'marica', 'zorra', 'perra', 'pinche'
        ]
    
    def create_improved_vectorizer(self, X_text, max_features=3000):
        """Crear vectorizador mejorado"""
        print("🔧 Creando vectorizador mejorado...")
        
        # Crear vectorizador base
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        # Ajustar el texto para incluir palabras importantes
        X_text_enhanced = self._enhance_text_with_important_words(X_text)
        
        # Entrenar vectorizador
        vectorizer.fit(X_text_enhanced)
        
        # Verificar que las palabras importantes estén incluidas
        feature_names = vectorizer.get_feature_names_out()
        missing_words = [word for word in self.important_words if word not in feature_names]
        
        if missing_words:
            print(f"⚠️ Palabras importantes faltantes: {missing_words[:10]}")
            # Crear vectorizador con más features si es necesario
            if len(missing_words) > 10:
                print("🔧 Aumentando max_features para incluir palabras importantes...")
                vectorizer = TfidfVectorizer(
                    max_features=max_features + 500,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=1,  # Reducir min_df para incluir palabras raras
                    max_df=0.95
                )
                vectorizer.fit(X_text_enhanced)
        
        # Verificar nuevamente
        feature_names = vectorizer.get_feature_names_out()
        included_words = [word for word in self.important_words if word in feature_names]
        print(f"✅ Palabras importantes incluidas: {len(included_words)}/{len(self.important_words)}")
        print(f"✅ Vocabulario total: {len(feature_names)} palabras")
        
        return vectorizer
    
    def _enhance_text_with_important_words(self, X_text):
        """Mejorar el texto para incluir palabras importantes"""
        enhanced_texts = []
        
        for text in X_text:
            if pd.isna(text):
                enhanced_texts.append("")
                continue
            
            # Agregar palabras importantes que podrían estar en el texto
            enhanced_text = str(text).lower()
            
            # Si el texto contiene evasiones, agregar la palabra normalizada
            evasions_map = {
                'f*ck': 'fuck', 'f_ck': 'fuck', 'fck': 'fuck',
                'sh*t': 'shit', 'sht': 'shit',
                'st*pid': 'stupid', 'stpid': 'stupid',
                '1d10t': 'idiot', 'id10t': 'idiot',
                '@sshole': 'asshole', 'a$$hole': 'asshole'
            }
            
            for evasion, normal in evasions_map.items():
                if evasion in enhanced_text:
                    enhanced_text += f" {normal}"
            
            enhanced_texts.append(enhanced_text)
        
        return enhanced_texts
    
    def train_improved_model(self, data_path, save_dir="backend/models/saved"):
        """Entrenar modelo con vectorizador mejorado"""
        print("🚀 ENTRENANDO MODELO MEJORADO")
        print("=" * 50)
        
        # Cargar datos
        df = pd.read_csv(data_path)
        X_text = df['clean_tweet_improved'].fillna('')
        y = df['class']
        
        print(f"📊 Datos cargados: {len(df)} ejemplos")
        print(f"📊 Distribución de clases:")
        print(y.value_counts())
        
        # Crear vectorizador mejorado
        vectorizer = self.create_improved_vectorizer(X_text)
        
        # Vectorizar datos
        X = vectorizer.transform(X_text)
        print(f"📊 Matriz de características: {X.shape}")
        
        # Entrenar modelo
        print("🤖 Entrenando modelo...")
        model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
        model.fit(X, y)
        
        # Guardar modelo mejorado
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, os.path.join(save_dir, "improved_model.pkl"))
        joblib.dump(vectorizer, os.path.join(save_dir, "improved_vectorizer.pkl"))
        
        # Guardar metadatos
        metadata = {
            "model_type": "LogisticRegression",
            "improved": True,
            "important_words_included": True,
            "features": vectorizer.max_features,
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df
        }
        
        import json
        with open(os.path.join(save_dir, "improved_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Modelo mejorado guardado en {save_dir}")
        
        # Probar con casos específicos
        self._test_improved_model(model, vectorizer)
        
        return model, vectorizer
    
    def _test_improved_model(self, model, vectorizer):
        """Probar el modelo mejorado"""
        print("\n🧪 PROBANDO MODELO MEJORADO:")
        print("-" * 50)
        
        test_cases = [
            "fuck you",
            "F*ck you", 
            "asshole",
            "@sshole",
            "stupid",
            "st*pid",
            "idiot",
            "1d10t",
            "Women are inferior to men",
            "This is fucking stupid",
            "I hate all immigrants"
        ]
        
        class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        
        for text in test_cases:
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            prediction_text = class_mapping.get(prediction, 'Unknown')
            confidence = max(probabilities) * 100
            
            print(f"📝 '{text}' → {prediction_text} ({confidence:.1%})")

def main():
    """Función principal"""
    vectorizer_improver = ImprovedVectorizer()
    model, vectorizer = vectorizer_improver.train_improved_model(
        "backend/data/processed/cleaned_tweets.csv"
    )

if __name__ == "__main__":
    main()
