#!/bin/bash
# ===========================================
# SCRIPT DE CONSTRUCCIÓN Y EJECUCIÓN DOCKER
# ===========================================

echo "🐳 Construyendo imagen Docker para Hate Speech Detector..."

# Construir la imagen
docker build -t hate-speech-detector .

echo "✅ Imagen construida exitosamente!"

echo "🚀 Iniciando contenedor..."

# Ejecutar el contenedor
docker run -d \
  --name hate-speech-app \
  -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  hate-speech-detector

echo "✅ Contenedor iniciado!"
echo "🌐 Aplicación disponible en: http://localhost:8501"
echo ""
echo "📋 Comandos útiles:"
echo "  - Ver logs: docker logs hate-speech-app"
echo "  - Parar: docker stop hate-speech-app"
echo "  - Eliminar: docker rm hate-speech-app"
echo "  - Entrar al contenedor: docker exec -it hate-speech-app bash"
