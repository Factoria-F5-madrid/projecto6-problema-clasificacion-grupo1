# 🗄️ Diagrama Entidad-Relación - Hate Speech Detector

## 📊 **DIAGRAMA ER**

```mermaid
erDiagram
    PREDICTIONS {
        int id PK
        datetime timestamp
        text text
        string classification
        float confidence
        string model_used
        json preprocessing_info
        string user_ip
        string session_id
    }
    
    MODEL_METRICS {
        int id PK
        datetime timestamp
        string model_name
        float accuracy
        float precision_score
        float recall_score
        float f1_score
        float confidence_avg
        float response_time_ms
        string evaluation_type
        int dataset_size
    }
    
    MODEL_REPLACEMENTS {
        int id PK
        datetime timestamp
        string old_model
        string new_model
        float old_score
        float new_score
        float improvement
        string reason
        string triggered_by
    }
    
    DRIFT_DETECTION {
        int id PK
        datetime timestamp
        float drift_score
        float kl_divergence
        float ks_statistic
        float p_value
        string alert_level
        int features_analyzed
        int sample_size
    }
    
    AB_TESTING {
        int id PK
        datetime timestamp
        string test_id
        string model_a
        string model_b
        float traffic_split
        int model_a_predictions
        int model_b_predictions
        float model_a_accuracy
        float model_b_accuracy
        float statistical_significance
        string recommendation
    }
    
    PREDICTIONS ||--o{ MODEL_METRICS : "uses"
    MODEL_METRICS ||--o{ MODEL_REPLACEMENTS : "triggers"
    DRIFT_DETECTION ||--o{ MODEL_REPLACEMENTS : "triggers"
    AB_TESTING ||--o{ MODEL_METRICS : "evaluates"
```

## 🏗️ **NORMALIZACIÓN DE DATOS**

### **1NF (Primera Forma Normal)**
- ✅ Todos los atributos son atómicos
- ✅ No hay grupos repetitivos
- ✅ Cada tabla tiene una clave primaria

### **2NF (Segunda Forma Normal)**
- ✅ Cumple 1NF
- ✅ No hay dependencias parciales
- ✅ Todos los atributos no clave dependen completamente de la clave primaria

### **3NF (Tercera Forma Normal)**
- ✅ Cumple 2NF
- ✅ No hay dependencias transitivas
- ✅ Los atributos no clave no dependen de otros atributos no clave

## 📋 **DESCRIPCIÓN DE ENTIDADES**

### **PREDICTIONS**
- **Propósito**: Almacena todas las predicciones realizadas por el sistema
- **Clave primaria**: `id` (autoincremental)
- **Relaciones**: Referencia a modelos a través de `model_used`

### **MODEL_METRICS**
- **Propósito**: Almacena métricas de rendimiento de cada modelo
- **Clave primaria**: `id` (autoincremental)
- **Relaciones**: Un modelo puede tener múltiples evaluaciones

### **MODEL_REPLACEMENTS**
- **Propósito**: Registra el historial de reemplazos automáticos de modelos
- **Clave primaria**: `id` (autoincremental)
- **Relaciones**: Referencia a modelos antiguos y nuevos

### **DRIFT_DETECTION**
- **Propósito**: Almacena resultados de detección de drift en los datos
- **Clave primaria**: `id` (autoincremental)
- **Relaciones**: Puede triggerar reemplazos de modelos

### **AB_TESTING**
- **Propósito**: Registra experimentos A/B entre modelos
- **Clave primaria**: `id` (autoincremental)
- **Relaciones**: Evalúa rendimiento de modelos

## 🔗 **RELACIONES**

1. **PREDICTIONS → MODEL_METRICS**: Un modelo puede tener múltiples predicciones y métricas
2. **MODEL_METRICS → MODEL_REPLACEMENTS**: Las métricas pueden triggerar reemplazos
3. **DRIFT_DETECTION → MODEL_REPLACEMENTS**: El drift puede triggerar reemplazos
4. **AB_TESTING → MODEL_METRICS**: Los tests A/B evalúan métricas de modelos

## 📊 **ÍNDICES RECOMENDADOS**

```sql
-- Índices para optimizar consultas frecuentes
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_classification ON predictions(classification);
CREATE INDEX idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX idx_drift_detection_timestamp ON drift_detection(timestamp);
CREATE INDEX idx_ab_testing_test_id ON ab_testing(test_id);
```

## 🎯 **VENTAJAS DEL DISEÑO**

- **✅ Escalabilidad**: Diseño normalizado para grandes volúmenes
- **✅ Flexibilidad**: Fácil agregar nuevos campos
- **✅ Integridad**: Claves foráneas y restricciones
- **✅ Performance**: Índices optimizados
- **✅ Auditoría**: Timestamps en todas las tablas
- **✅ MLOps**: Soporte completo para monitoreo y reemplazo automático
