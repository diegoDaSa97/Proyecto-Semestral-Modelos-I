# syntax=docker/dockerfile:1
#
# Imagen de la Fase 2 - Proyecto Integrador "Modelos y Simulación de Sistemas I"
#
# Requisitos de la especificación técnica que cubre este archivo:
#   §11.1 - Imagen base de Python con versión explícita (3.13), nunca `latest`.
#   §11.1 - Directorio de trabajo, dependencias instaladas y código copiado.
#   §11.1 - Permite ejecutar train.py y predict.py desde la misma imagen.
#   §11.1 - No se entrena el modelo durante `docker build`.
#   §11.2 - Datos, modelos, métricas y predicciones se montan desde el host.
#
# Importante: NO se define ENTRYPOINT. El comando se escribe completo en cada
# ejecución (`docker run ... imagen python train.py ...`) y `CMD` solo provee
# una ayuda por defecto, de modo que no bloquea los comandos de la entrega.

FROM python:3.13-slim AS base

LABEL org.opencontainers.image.title="churn-fase2" \
      org.opencontainers.image.description="Entrenamiento y predicción de churn - Fase 2" \
      org.opencontainers.image.source="https://github.com/diegoDaSa97/Proyecto-Semestral-Modelos-I"

# Salida sin buffer (logs en vivo), sin .pyc y sin caché de pip dentro de la imagen.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero: si solo cambia el código, Docker reutiliza esta capa.
# Las versiones están fijadas en requirements.txt (compatibles con los .joblib).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación y datos (el dataset es el mismo de la Fase 1).
COPY train.py predict.py ./
COPY data/ ./data/

# Carpetas de salida. En ejecución se sobrescriben con volúmenes del host para
# que modelos, métricas y predicciones persistan fuera del contenedor.
RUN mkdir -p /app/models /app/outputs

# ---------------------------------------------------------------------------
# Etapa opcional para ejecutar pytest dentro de la imagen (no es la etapa por
# defecto). Uso:
#   docker build --target test -t churn-fase2:test .
#   docker run --rm churn-fase2:test
# ---------------------------------------------------------------------------
FROM base AS test

COPY requirements-dev.txt pytest.ini ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "-q"]

# ---------------------------------------------------------------------------
# Etapa final (la que se construye por defecto): imagen de ejecución.
# ---------------------------------------------------------------------------
FROM base AS runtime

# Sin ENTRYPOINT: el comando real se indica en `docker run`.
CMD ["python", "train.py", "--help"]
