# Construir desde la raíz del repositorio:
# docker build -f fase-3/Dockerfile -t churn-api .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models \
    DATA_PATH=/app/data/Telco-Customer-Churn.csv \
    LEGACY_MODEL_PATH=/app/legacy_models/churn_model_pipeline.joblib

WORKDIR /app

COPY fase-3/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# API de Fase 3 y scripts reutilizados de Fase 2.
COPY fase-3/app/ ./app/
COPY fase-2/train.py ./train.py
COPY fase-2/predict.py ./predict.py
COPY fase-2/data/ ./data/

# Modelo inicial de Fase 1: permite iniciar la API antes del primer /train.
COPY fase-1/models/churn_model_pipeline.joblib ./legacy_models/churn_model_pipeline.joblib

RUN mkdir -p /app/models
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
