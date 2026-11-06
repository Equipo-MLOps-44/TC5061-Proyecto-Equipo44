
# Imagen base con Python 3.11
FROM python:3.11-slim

# Configuracion de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias de la API
COPY api/requirements-api.txt /app/api/requirements-api.txt
RUN pip install --no-cache-dir -r /app/api/requirements-api.txt

# Copiar el codigo de FastAPI
COPY api/ /app/api/

# Copiar el modelo descargado desde MLflow Registry
COPY models/serving/ /app/models/serving/

# Puerto donde funcionara FastAPI
EXPOSE 8000

# Iniciar la API cuando arranque el contenedor
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
