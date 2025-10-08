# project6_classification_task_team1

## Descripción del Proyecto
Proyecto de clasificación desarrollado por el equipo 1 de Factoría F5 Madrid.

## Estructura del Proyecto
```
.
├── backend/
│   ├── data/          # Datos del proyecto
│   ├── models/        # Modelos de machine learning
│   ├── notebooks/     # Jupyter notebooks
│   └── tests/         # Tests del backend
├── frontend/          # Código del frontend
├── requirements.txt   # Dependencias de Python
└── README.md          # Este archivo
```

## Cómo Añadir Colaboradores al Proyecto

### Opción 1: Añadir Colaboradores desde GitHub (Para Administradores)

1. Ve a la página del repositorio en GitHub
2. Haz clic en **Settings** (Configuración) en la barra superior
3. En el menú lateral, selecciona **Collaborators** (Colaboradores)
4. Haz clic en el botón **Add people** (Añadir personas)
5. Escribe el nombre de usuario de GitHub o el email de la persona que quieres añadir
6. Selecciona el nivel de permisos:
   - **Read**: Solo lectura
   - **Write**: Puede hacer push y merge
   - **Admin**: Control total del repositorio
7. Haz clic en **Add [nombre de usuario] to this repository**

### Opción 2: Trabajar con el Proyecto (Para Colaboradores)

Si eres un colaborador que acaba de ser añadido:

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/Factoria-F5-madrid/projecto6-problema-clasificacion-grupo1.git
   cd projecto6-problema-clasificacion-grupo1
   ```

2. **Configura tu entorno de desarrollo:**
   ```bash
   # Crea un entorno virtual de Python
   python -m venv venv
   
   # Activa el entorno virtual
   # En Linux/Mac:
   source venv/bin/activate
   # En Windows:
   # venv\Scripts\activate
   
   # Instala las dependencias
   pip install -r requirements.txt
   ```

3. **Crea una rama para tu trabajo:**
   ```bash
   git checkout -b feature/tu-funcionalidad
   ```

4. **Haz tus cambios y súbelos:**
   ```bash
   git add .
   git commit -m "Descripción de tus cambios"
   git push origin feature/tu-funcionalidad
   ```

5. **Crea un Pull Request:**
   - Ve al repositorio en GitHub
   - Haz clic en **Pull Requests** > **New Pull Request**
   - Selecciona tu rama y describe los cambios realizados

## Guía de Contribución

### Flujo de Trabajo

1. Actualiza tu rama principal antes de empezar:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Crea una nueva rama para cada funcionalidad o corrección:
   ```bash
   git checkout -b tipo/descripcion-breve
   ```
   
   Tipos de rama recomendados:
   - `feature/` - Nueva funcionalidad
   - `fix/` - Corrección de errores
   - `docs/` - Cambios en documentación
   - `test/` - Añadir o modificar tests

3. Realiza commits descriptivos y frecuentes

4. Sube tus cambios y crea un Pull Request

5. Espera la revisión de otros colaboradores

### Buenas Prácticas

- Escribe mensajes de commit claros y descriptivos
- Mantén tus Pull Requests pequeños y enfocados
- Comenta tu código cuando sea necesario
- Actualiza la documentación si haces cambios significativos
- Ejecuta los tests antes de hacer push

## Instalación y Configuración

### Requisitos Previos
- Python 3.x
- Git

### Instalación
```bash
# Clona el repositorio
git clone https://github.com/Factoria-F5-madrid/projecto6-problema-clasificacion-grupo1.git

# Entra al directorio
cd projecto6-problema-clasificacion-grupo1

# Crea y activa el entorno virtual
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate  # En Windows

# Instala las dependencias
pip install -r requirements.txt
```

## Contacto y Soporte

Para preguntas o problemas:
- Abre un [Issue](https://github.com/Factoria-F5-madrid/projecto6-problema-clasificacion-grupo1/issues) en GitHub
- Contacta con los administradores del equipo

## Licencia

Este proyecto es parte del programa de Factoría F5 Madrid.