"""
Balanceador de clases para mejorar la precisión del modelo
Usa SMOTE para balancear las clases desbalanceadas
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

class ClassBalancer:
    """Balanceador de clases para mejorar la precisión del modelo"""
    
    def __init__(self):
        self.smote = SMOTE(random_state=42, k_neighbors=1)
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english', 
            ngram_range=(1,3),
            min_df=2,
            max_df=0.95
        )
        self.model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
    
    def load_and_balance_data(self, data_path):
        """Cargar datos y balancear clases"""
        print("📊 Cargando datos...")
        df = pd.read_csv(data_path)
        
        # Preparar datos
        X_text = df['clean_tweet_improved'].fillna('')
        y = df['class']
        
        print(f"📈 Distribución original:")
        print(y.value_counts())
        print(f"📊 Porcentajes:")
        print(y.value_counts(normalize=True) * 100)
        
        # Vectorizar texto
        print("\n🔤 Vectorizando texto...")
        X = self.vectorizer.fit_transform(X_text)
        
        # Aplicar SMOTE
        print("\n⚖️ Aplicando SMOTE para balancear clases...")
        X_balanced, y_balanced = self.smote.fit_resample(X, y)
        
        print(f"\n📈 Distribución después de SMOTE:")
        print(pd.Series(y_balanced).value_counts())
        print(f"📊 Porcentajes:")
        print(pd.Series(y_balanced).value_counts(normalize=True) * 100)
        
        return X_balanced, y_balanced, X_text, y
    
    def train_balanced_model(self, X_balanced, y_balanced):
        """Entrenar modelo con datos balanceados"""
        print("\n🤖 Entrenando modelo balanceado...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        
        print("\n📊 Métricas del modelo balanceado:")
        print(classification_report(y_test, y_pred))
        
        # Calcular overfitting
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        overfitting = abs(train_score - test_score)
        
        print(f"\n📈 Overfitting: {overfitting:.4f} ({overfitting*100:.2f}%)")
        if overfitting < 0.05:
            print("✅ Overfitting controlado (< 5%)")
        else:
            print("⚠️ Overfitting alto (> 5%)")
        
        return self.model, self.vectorizer
    
    def save_balanced_model(self, model, vectorizer, save_dir="backend/models/saved"):
        """Guardar modelo balanceado"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Guardar modelo
        joblib.dump(model, os.path.join(save_dir, "balanced_model.pkl"))
        joblib.dump(vectorizer, os.path.join(save_dir, "balanced_vectorizer.pkl"))
        
        # Guardar metadatos
        metadata = {
            "model_type": "LogisticRegression",
            "balanced": True,
            "smote_applied": True,
            "features": vectorizer.max_features,
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df
        }
        
        import json
        with open(os.path.join(save_dir, "balanced_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Modelo balanceado guardado en {save_dir}")
    
    def test_balanced_model(self, model, vectorizer, test_cases):
        """Probar modelo balanceado con casos de prueba"""
        print("\n🧪 Probando modelo balanceado...")
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n{i}. Texto: '{text}'")
            
            # Predecir
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Mapear predicción
            class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            prediction_text = class_mapping.get(prediction, 'Unknown')
            confidence = max(probabilities) * 100
            
            print(f"   Predicción: {prediction_text}")
            print(f"   Confianza: {confidence:.1f}%")
            print(f"   Probabilidades: {dict(zip(class_mapping.values(), probabilities))}")

def main():
    """Función principal para balancear clases"""
    print("🔄 BALANCEADOR DE CLASES - Mejorando Precisión del Modelo")
    print("=" * 60)
    
    # Inicializar balanceador
    balancer = ClassBalancer()
    
    # Cargar y balancear datos
    data_path = "backend/data/processed/cleaned_tweets.csv"
    X_balanced, y_balanced, X_text, y = balancer.load_and_balance_data(data_path)
    
    # Entrenar modelo balanceado
    model, vectorizer = balancer.train_balanced_model(X_balanced, y_balanced)
    
    # Guardar modelo balanceado
    balancer.save_balanced_model(model, vectorizer)
    
    # Probar con casos de prueba
    test_cases = [
        "Hello, how are you?",
        "You are an idiot",
        "Women are inferior to men",
        "This is fucking stupid",
        "I hate all immigrants"
    ]
    
    balancer.test_balanced_model(model, vectorizer, test_cases)
    
    print("\n✅ Balanceo de clases completado!")
    print("🎯 El modelo ahora debería tener mejor precisión en todas las clases")

if __name__ == "__main__":
    main()
