# 🚨 Sistema de Clasificación de Hate Speech - Estructura Organizada

## 📁 Estructura del Proyecto

```
proyecto6-problema-clasificacion-grupo1/
├── 📄 main.py                          # Punto de entrada principal
├── 📄 requirements.txt                 # Dependencias del proyecto
├── 📄 README.md                        # Documentación principal
├── 📄 README_ORGANIZED.md              # Esta documentación
│
├── 🎨 frontend/                        # Interfaz de usuario
│   └── apps/                          # Aplicaciones Streamlit
│       ├── app_organized.py           # App completa con MLOps
│       ├── app_simple.py              # App simplificada
│       ├── app.py                     # App básica
│       └── start_app.py               # Selector de aplicaciones
│
├── ⚙️ backend/                         # Lógica de negocio
│   ├── config/                        # Configuraciones
│   │   ├── model_config.py           # Configuración de modelos
│   │   └── hybrid_config.py          # Configuración híbrida
│   ├── data/                          # Datos del proyecto
│   │   └── processed/                 # Datos procesados
│   ├── mlops/                         # MLOps y automatización
│   │   ├── ab_testing.py             # A/B Testing
│   │   ├── auto_model_replacement.py # Auto-reemplazo de modelos
│   │   └── data_drift_monitor.py     # Monitoreo de deriva
│   ├── models/                        # Modelos de ML
│   │   ├── saved/                    # Modelos guardados
│   │   ├── baseline_models.py        # Modelos base
│   │   ├── advanced_hybrid_system.py # Sistema híbrido avanzado
│   │   └── ultimate_hybrid_system.py # Sistema híbrido definitivo
│   ├── tests/                         # Tests del backend
│   └── utils/                         # Utilidades
│       ├── api_*.py                  # APIs externas
│       ├── robust_preprocessor.py    # Preprocesamiento
│       └── smart_evasion_detector.py # Detector de evasiones
│
├── 📊 data/                           # Datos del proyecto
│   ├── raw/                          # Datos originales
│   └── processed/                    # Datos procesados
│       ├── cleaned_tweets.csv        # Tweets limpios
│       └── expanded_tweets.csv       # Tweets expandidos
│
├── 🔧 scripts/                        # Scripts de utilidad
│   ├── mlops/                        # Scripts de MLOps
│   │   ├── diagnose_auto_replacement.py
│   │   └── fix_current_model.py
│   ├── testing/                      # Scripts de testing
│   │   ├── expand_dataset.py
│   │   ├── fix_model_thresholds.py
│   │   └── improve_model_accuracy.py
│   └── utilities/                    # Scripts de utilidad
│       ├── create_demo_models.py
│       ├── stronger_rules.py
│       └── stronger_rules_fixed.py
│
├── 📚 docs/                          # Documentación
│   ├── readmes/                      # READMEs específicos
│   ├── deployment/                   # Documentación de despliegue
│   └── MLOPS_README.md              # Documentación MLOps
│
└── ⚙️ config/                        # Archivos de configuración
    ├── hatespeech.png               # Imagen del banner
    └── APP_VERSIONS_README.md       # Documentación de versiones
```

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Usando el archivo principal (Recomendado)
```bash
python main.py
```

### Opción 2: Ejecutando aplicaciones directamente
```bash
# Aplicación organizada (completa con MLOps)
streamlit run frontend/apps/app_organized.py --server.port 8501

# Aplicación simple
streamlit run frontend/apps/app_simple.py --server.port 8502

# Aplicación básica
streamlit run frontend/apps/app.py --server.port 8503
```

### Opción 3: Usando el selector de aplicaciones
```bash
python frontend/apps/start_app.py
```

## 🧪 Scripts de Utilidad

### MLOps
```bash
# Diagnosticar sistema de auto-reemplazo
python scripts/mlops/diagnose_auto_replacement.py

# Arreglar modelo actual
python scripts/mlops/fix_current_model.py
```

### Testing
```bash
# Expandir dataset
python scripts/testing/expand_dataset.py

# Mejorar precisión del modelo
python scripts/testing/improve_model_accuracy.py
```

### Utilidades
```bash
# Crear modelos de demostración
python scripts/utilities/create_demo_models.py
```

## 📋 Características del Proyecto

### ✅ Nivel Esencial
- ✅ Modelo de ML funcional (Logistic Regression, XGBoost, BERT)
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Overfitting < 5%
- ✅ Aplicación Streamlit productiva
- ✅ Métricas de clasificación completas

### ✅ Nivel Medio
- ✅ Modelos ensemble (Random Forest, XGBoost)
- ✅ Validación cruzada K-Fold
- ✅ Optimización de hiperparámetros
- ✅ Sistema de feedback en tiempo real
- ✅ Pipeline de re-entrenamiento

### ✅ Nivel Avanzado
- ✅ Versión dockerizada
- ✅ Guardado en bases de datos
- ✅ Despliegue en Render
- ✅ Tests unitarios

### ✅ Nivel Experto
- ✅ Modelos de redes neuronales (BERT)
- ✅ A/B Testing para comparar modelos
- ✅ Monitoreo de Data Drift
- ✅ Auto-reemplazo de modelos

## 🔧 Tecnologías Utilizadas

- **Machine Learning:** scikit-learn, XGBoost, BERT, Transformers
- **Frontend:** Streamlit
- **Backend:** Python, FastAPI
- **MLOps:** A/B Testing, Data Drift Monitoring, Auto-replacement
- **Datos:** Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn, Plotly
- **Despliegue:** Docker, Render

## 📊 Métricas del Sistema

- **Precisión:** 93.8%
- **Accuracy:** 85.5% - 90.4%
- **F1-Score:** 87.0% - 91.6%
- **Tiempo de análisis:** < 2 segundos
- **Soporte multilingüe:** Español e Inglés

## 🎯 Funcionalidades Principales

1. **Detección Híbrida:** Combina reglas específicas + ML
2. **Detección de Evasiones:** Leet speak, palabras espaciadas, Unicode
3. **MLOps Completo:** A/B Testing, Data Drift, Auto-replacement
4. **Interfaz Intuitiva:** Múltiples versiones de aplicación
5. **Análisis en Tiempo Real:** Métricas y visualizaciones
6. **Sistema Robusto:** Manejo de errores y fallbacks

## 📝 Notas de Desarrollo

- **Estructura modular:** Cada componente en su lugar
- **Importaciones relativas:** Funcionan desde cualquier ubicación
- **Documentación completa:** READMEs específicos para cada módulo
- **Tests incluidos:** Scripts de prueba y validación
- **Configuración centralizada:** Archivos de config en carpeta dedicada

---

**¡El proyecto está completamente organizado y listo para producción!** 🚀✨
