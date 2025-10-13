# 🏆 Proyecto de Clasificación: Detector de Discurso de Odio y Lenguaje Ofensivo

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar y productivizar un modelo de Machine Learning capaz de clasificar tweets en tres categorías: **Hate Speech (Discurso de Odio)**, **Offensive Language (Lenguaje Ofensivo)** y **Neither (Neutral)**. Dada la naturaleza crítica de la clase minoritaria (*Hate Speech*), el modelo fue optimizado para un alto **Recall** en dicha categoría, garantizando la detección de contenido de alto riesgo.

## 📦 Estructura del Repositorio

| Directorio/Archivo | Contenido |
| :--- | :--- |
| `backend/` | Contiene la lógica de modelado (código de *training*), datos y modelos guardados. |
| `backend/data/` | Datos crudos (`raw/`) y limpios (`processed/`). |
| `backend/models/` | Modelos serializados (`modelo_final.joblib`). |
| `frontend/` | Código fuente de la aplicación web (Streamlit). |
| `notebooks/` | Análisis exploratorio de datos (EDA) y desarrollo de *baseline*. |
| `Dockerfile` | Receta para construir la imagen de Docker para la productivización. |
| `requirements.txt` | Lista de dependencias necesarias para el proyecto. |
| `app.py` | Aplicación principal de Streamlit para la predicción en tiempo real. |

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas |
| :--- | :--- |
| **Machine Learning** | Python, Scikit-learn (Gradient Boosting Classifier), Pandas, NumPy. |
| **NLP** | NLTK, TF-IDF. |
| **Productivización** | Streamlit, Docker. |
| **Control de Versiones** | Git, GitHub (Flujo GitFlow). |
| **Gestión de Proyecto** | Trello. |

## 🚀 Guía de Instalación y Ejecución

### 1. Clonar el Repositorio

```bash
git clone [https://github.com/Factoria-F5-madrid/projecto6-problema-clasificacion-grupo1.git](https://github.com/Factoria-F5-madrid/projecto6-problema-clasificacion-grupo1.git)
cd projecto6-problema-clasificacion-grupo1
```

### 2. Ejecución Local (Recomendado: Docker)

Utilizar Docker asegura que el modelo y la aplicación corran en el mismo entorno que usamos para el despliegue.

#### 1. construir la imagen

```bash
docker build -t clasificacion-tweets:v1.1 .
```

#### 2. Ejecutar el contenedor 

```bash

docker run -p 8501:8501 clasificacion-tweets:v1.1
```

#### 3. acceder a la aplicación

Abre tu navegador y navega a http://localhost:8501.


### 3. Ejecución Local (Entorno Virtual)

```bash
# Crear y activar entorno virtual (venv)
python -m venv venv
# ... activar venv según tu OS ...

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación Streamlit
streamlit run app.py
```

📈 Rendimiento y Métricas Clave
El modelo final logró un F1-Score de 84.1% para la clase crítica de Hate Speech en el conjunto de validación, con un overfitting controlado en el 4.4% (diferencia entre F1-Score de training y validation).
