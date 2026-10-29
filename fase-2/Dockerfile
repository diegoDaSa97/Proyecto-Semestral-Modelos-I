# Imagen base con la misma versión mayor de Python usada en la Fase 1 (3.13).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero: aprovecha la caché de capas de Docker si el código cambia.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código y datos de ejemplo. Los modelos NO se incluyen: se generan con train.py
# y se montan como volumen para poder versionarlos fuera del contenedor.
COPY train.py predict.py ./
COPY data/ ./data/
RUN mkdir -p /app/models /app/output

# Se elige el script al ejecutar: `docker run ... churn-fase2 train.py`
ENTRYPOINT ["python"]
CMD ["train.py", "--help"]
