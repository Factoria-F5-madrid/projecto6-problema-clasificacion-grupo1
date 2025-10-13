"""
Entrenador de Transformers (BERT) para detección de hate speech
Implementa fine-tuning de BERT multilingüe para alcanzar nivel experto
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import joblib
import os
import json
from datetime import datetime

class TransformerTrainer:
    """Entrenador de Transformers para hate speech detection"""
    
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Mapeo de clases
        self.class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        self.label_mapping = {'Hate Speech': 0, 'Offensive Language': 1, 'Neither': 2}
        
    def load_and_prepare_data(self, data_path):
        """Cargar y preparar datos para Transformers"""
        print("📊 Cargando datos para Transformers...")
        
        # Cargar datos
        df = pd.read_csv(data_path)
        
        # Preparar datos
        df_clean = df.dropna(subset=['clean_tweet_improved', 'class'])
        df_clean = df_clean[df_clean['clean_tweet_improved'].str.strip() != '']
        
        # Mapear clases a números
        df_clean['label'] = df_clean['class'].map(self.label_mapping)
        
        print(f"📈 Datos cargados: {len(df_clean)} ejemplos")
        print(f"📊 Distribución de clases:")
        print(df_clean['label'].value_counts().sort_index())
        
        # Dividir datos
        train_df, val_df = train_test_split(
            df_clean, 
            test_size=0.2, 
            stratify=df_clean['label'], 
            random_state=42
        )
        
        print(f"📊 Datos de entrenamiento: {len(train_df)}")
        print(f"📊 Datos de validación: {len(val_df)}")
        
        return train_df, val_df
    
    def tokenize_data(self, train_df, val_df):
        """Tokenizar datos para Transformers"""
        print("🔤 Tokenizando datos...")
        
        # Cargar tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Crear datasets
        train_dataset = Dataset.from_pandas(train_df[['clean_tweet_improved', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['clean_tweet_improved', 'label']])
        
        # Función de tokenización
        def tokenize_function(examples):
            return self.tokenizer(
                examples['clean_tweet_improved'], 
                padding='max_length', 
                truncation=True, 
                max_length=128
            )
        
        # Aplicar tokenización
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        print("✅ Tokenización completada")
        return train_dataset, val_dataset
    
    def setup_model(self, num_labels=3):
        """Configurar modelo Transformer"""
        print(f"🤖 Configurando modelo: {self.model_name}")
        
        # Cargar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
        print("✅ Modelo configurado")
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Calcular métricas de evaluación"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, val_dataset, output_dir="./models/transformer"):
        """Entrenar modelo Transformer"""
        print("🚀 Iniciando entrenamiento de Transformer...")
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=100,
            eval_steps=100,
            warmup_steps=100,
            fp16=torch.cuda.is_available(),  # Usar FP16 si hay GPU
            dataloader_num_workers=4,
            remove_unused_columns=False
        )
        
        # Crear trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Entrenar
        print("🔥 Iniciando fine-tuning...")
        train_result = self.trainer.train()
        
        # Evaluar
        print("📊 Evaluando modelo...")
        eval_result = self.trainer.evaluate()
        
        print("✅ Entrenamiento completado!")
        print(f"📈 Métricas finales:")
        for key, value in eval_result.items():
            print(f"   {key}: {value:.4f}")
        
        return train_result, eval_result
    
    def save_model(self, output_dir="./models/transformer"):
        """Guardar modelo entrenado"""
        print(f"💾 Guardando modelo en {output_dir}")
        
        # Crear directorio
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar modelo y tokenizador
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Guardar metadatos
        metadata = {
            "model_name": self.model_name,
            "model_type": "Transformer",
            "num_labels": 3,
            "class_mapping": self.class_mapping,
            "label_mapping": self.label_mapping,
            "training_date": datetime.now().isoformat(),
            "framework": "transformers"
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Modelo guardado exitosamente")
    
    def predict(self, texts, model_path="./models/transformer"):
        """Hacer predicciones con el modelo Transformer"""
        if self.model is None or self.tokenizer is None:
            # Cargar modelo guardado
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Tokenizar textos
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Mover a GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Hacer predicciones
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # Convertir a CPU y numpy
        probabilities = probabilities.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        # Formatear resultados
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i],
                'prediction': self.class_mapping[pred],
                'confidence': float(max(probs)),
                'probabilities': {
                    self.class_mapping[j]: float(prob) 
                    for j, prob in enumerate(probs)
                }
            }
            results.append(result)
        
        return results

def main():
    """Función principal para entrenar Transformer"""
    print("🚀 ENTRENAMIENTO DE TRANSFORMER - NIVEL EXPERTO")
    print("=" * 60)
    
    # Inicializar trainer
    trainer = TransformerTrainer()
    
    # Cargar y preparar datos
    train_df, val_df = trainer.load_and_prepare_data("backend/data/processed/cleaned_tweets.csv")
    
    # Tokenizar datos
    train_dataset, val_dataset = trainer.tokenize_data(train_df, val_df)
    
    # Configurar modelo
    trainer.setup_model()
    
    # Entrenar modelo
    train_result, eval_result = trainer.train_model(train_dataset, val_dataset)
    
    # Guardar modelo
    trainer.save_model()
    
    # Probar con casos específicos
    print("\n🧪 PROBANDO MODELO TRANSFORMER:")
    print("-" * 50)
    
    test_cases = [
        "fuck you",
        "F*ck you", 
        "asshole",
        "@sshole",
        "stupid",
        "st*pid",
        "Women are inferior to men",
        "This is fucking stupid",
        "I hate all immigrants",
        "Hello, how are you?"
    ]
    
    results = trainer.predict(test_cases)
    
    for result in results:
        print(f"📝 '{result['text']}' → {result['prediction']} ({result['confidence']:.1%})")
    
    print("\n✅ Entrenamiento de Transformer completado!")
    print("🎯 Nivel experto alcanzado con comprensión semántica")

if __name__ == "__main__":
    main()
