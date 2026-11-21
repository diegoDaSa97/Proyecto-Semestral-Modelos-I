# Construir desde la raíz del repositorio:
# docker build -f fase-4/Dockerfile -t churn-api-fase4 .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models \
    DATA_PATH=/app/data/Telco-Customer-Churn.csv \
    LEGACY_MODEL_PATH=/app/legacy_models/churn_model_pipeline.joblib \
    LOG_PATH=/app/logs/predicciones.csv

WORKDIR /app

COPY fase-4/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# API de Fase 4 (registro) + API de Fase 3 reutilizada sin duplicar endpoints.
COPY fase-4/app/ ./app/
COPY fase-3/app/ ./fase3_app/

# Scripts y datos reutilizados de Fase 2, y modelo inicial de Fase 1.
COPY fase-2/train.py ./train.py
COPY fase-2/predict.py ./predict.py
COPY fase-2/data/ ./data/
COPY fase-1/models/churn_model_pipeline.joblib ./legacy_models/churn_model_pipeline.joblib

# Cliente de demostración y script de monitoreo, útiles dentro del contenedor.
COPY fase-4/monitor.py ./monitor.py
COPY fase-4/client/ ./client/

RUN mkdir -p /app/models /app/logs
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
