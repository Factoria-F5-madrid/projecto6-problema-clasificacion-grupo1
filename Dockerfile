# ===========================================
# DOCKERFILE PARA SISTEMA DE DETECCIÓN DE HATE SPEECH
# ===========================================

# Usar Python 3.11 como imagen base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorio para logs
RUN mkdir -p logs

# Exponer puerto 8501 (Streamlit por defecto)
EXPOSE 8501

# Variables de entorno
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Inicializar base de datos (opcional)
RUN python scripts/database/init_database.py || echo "Base de datos no inicializada - continuando sin logs persistentes"

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "frontend/apps/app_organized.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
