# 🚀 ML Optimization Framework

## 📋 Descripción del Proyecto

Este framework está diseñado para **optimizar modelos de Machine Learning** mediante validación cruzada y ajuste de hiperparámetros. Es **compatible** con cualquier modelo que crees.

## 👥 División de Trabajo

### **Compañero B (Ciprian) - Modelado Base:**
- Crea modelos iniciales (Logistic Regression, Naive Bayes, SVM, etc.)
- Hace evaluación básica con métricas simples
- **Entrega:** Modelos entrenados + métricas básicas

### **Compañero C (Barb) - Optimización:**
- Toma los modelos base de Ciprian
- Aplica validación cruzada y optimización de hiperparámetros
- Controla el overfitting (<5%)
- **Entrega:** Modelos optimizados + control de overfitting

## 📁 Estructura del Proyecto

```
backend/
├── models/
│   ├── cross_validation.py      # ✅ Sistema de validación cruzada
│   ├── hyperparameter_tuning.py # 🔄 Framework de optimización (próximo)
│   ├── model_evaluation.py      # 📊 Evaluación detallada (próximo)
│   └── automation_pipeline.py   # 🤖 Pipeline automático (próximo)
├── config/
│   └── model_config.py          # ✅ Configuraciones de modelos
└── utils/
    ├── data_preprocessing.py    # 🔄 Preprocesamiento genérico (próximo)
    └── experiment_logger.py     # 📝 Logging de experimentos (próximo)
```

## ✅ **Archivos Completados:**

### 1. `config/model_config.py`
**¿Qué hace?**
- Define 7 algoritmos de ML listos para usar
- Contiene hiperparámetros predefinidos para cada modelo
- Configuración de validación cruzada

**¿Cómo lo usa Ciprian?**
```python
from backend.config.model_config import ModelConfig

# Obtener un modelo
model = ModelConfig.get_model('random_forest')

# Obtener hiperparámetros para optimización
params = ModelConfig.get_hyperparameters('random_forest')
```

### 2. `models/cross_validation.py`
**¿Qué hace?**
- Sistema de validación cruzada (K-Fold, Stratified K-Fold)
- Control automático de overfitting (<5%)
- Métricas completas: accuracy, precision, recall, F1, ROC-AUC
- Comparación automática de modelos

**¿Cómo lo usa Ciprian?**
```python
from backend.models.cross_validation import CrossValidator

# Crear validador
validator = CrossValidator(cv_folds=5)

# Evaluar un modelo
results = validator.evaluate_model(model, X, y)

# Comparar múltiples modelos
comparison = validator.compare_models(models_dict, X, y)
```

### 3. `models/hyperparameter_tuning.py` ✅ **NUEVO**
**¿Qué hace?**
- **GridSearchCV**: Búsqueda exhaustiva de hiperparámetros (más lento, más preciso)
- **RandomizedSearchCV**: Búsqueda aleatoria de hiperparámetros (más rápido)
- **Comparación**: Compara ambos métodos automáticamente
- **Compatible** con cualquier modelo de sklearn

**¿Cómo lo usa Ciprian?**
```python
from backend.models.hyperparameter_tuning import HyperparameterTuner
from backend.config.model_config import ModelConfig

# Crear tuner
tuner = HyperparameterTuner(cv_folds=5)

# Obtener modelo y parámetros
model = ModelConfig.get_model('random_forest')
params = ModelConfig.get_hyperparameters('random_forest')

# Opción 1: Grid Search (más lento, más preciso)
grid_results = tuner.grid_search(model, params, X, y)

# Opción 2: Randomized Search (más rápido)
random_results = tuner.randomized_search(model, params, X, y, n_iter=100)

# Opción 3: Comparar ambos métodos
comparison = tuner.compare_methods(model, params, X, y)
```

**¿Cuándo usar cada método?**
- **GridSearchCV**: Cuando tienes pocos hiperparámetros y tiempo suficiente
- **RandomizedSearchCV**: Cuando tienes muchos hiperparámetros o poco tiempo
- **compare_methods()**: Para decidir cuál método usar en tu caso específico

## 🧪 **Testing the Framework:**

Para probar que todo funciona correctamente:

```bash
# Desde la raíz del proyecto
python backend/tests/test_simple.py
```

Este test verifica:
- ✅ CrossValidator funciona correctamente
- ✅ HyperparameterTuner (GridSearch y RandomizedSearch) funciona
- ✅ Control de overfitting funciona
- ✅ Todos los imports y dependencias están correctos

## 🔄 **Próximos Pasos:**

1. ✅ **HyperparameterTuner** - Optimización automática de hiperparámetros
2. **ModelEvaluator** - Evaluación detallada con visualizaciones
3. **DataPreprocessor** - Preprocesamiento genérico de datos
4. **ExperimentLogger** - Sistema de logging de experimentos
5. **MLPipeline** - Pipeline completo de automatización

## 🤝 **Flujo de Trabajo Recomendado:**

1. **Ciprian** crea modelos base en su rama
2. **Barb** toma esos modelos y los optimiza con este framework
3. **Integración** de resultados en la rama principal

## 📚 **Documentación Adicional:**

- [Scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

