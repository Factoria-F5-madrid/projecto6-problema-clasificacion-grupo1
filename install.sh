#!/bin/bash
# ===========================================
# SCRIPT DE INSTALACIÓN PARA RENDER
# ===========================================

echo "🚀 Instalando dependencias..."

# Instalar dependencias
pip install -r requirements.txt

echo "✅ Dependencias instaladas"

# Inicializar base de datos (opcional)
echo "🗄️ Inicializando base de datos..."
python3 scripts/database/init_database.py || echo "⚠️ Base de datos no inicializada - continuando sin logs persistentes"

echo "✅ Instalación completada"
