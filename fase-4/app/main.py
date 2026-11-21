"""API de la Fase 4: extiende la API de la Fase 3 con registro de predicciones.

En lugar de duplicar los endpoints, esta aplicación carga el objeto ``app`` de
``fase-3/app/main.py`` y le añade un middleware que registra una fila por cada
solicitud ``POST /predict``. Así se conserva una única implementación de la
predicción y del entrenamiento.

El registro se guarda en un CSV persistente (``LOG_PATH``) con las columnas
``fecha_hora, estado_http, version, prediccion``. Las solicitudes rechazadas
por validación (HTTP 422) también se registran, con versión y predicción vacías.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("churn_api.fase4")

APP_DIR = Path(__file__).resolve().parent
FASE4_DIR = APP_DIR.parent
PROJECT_ROOT = FASE4_DIR.parent

CAMPOS_REGISTRO = ("fecha_hora", "estado_http", "version", "prediccion")


def _localizar_api_fase3() -> Path:
    """Devuelve la ruta de la API de Fase 3 en ejecución local o en Docker."""
    candidatos = (
        PROJECT_ROOT / "fase-3" / "app" / "main.py",
        FASE4_DIR / "fase3_app" / "main.py",
        Path("/app/fase3_app/main.py"),
    )
    for candidato in candidatos:
        if candidato.is_file():
            return candidato
    raise FileNotFoundError(
        "No se encontró fase-3/app/main.py. La Fase 4 reutiliza la API de la Fase 3."
    )


def _cargar_api_fase3() -> ModuleType:
    """Carga la API de Fase 3 con un nombre de módulo propio para no colisionar."""
    ruta = _localizar_api_fase3()
    especificacion = importlib.util.spec_from_file_location("fase3_api", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo preparar la carga de la API de Fase 3: {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["fase3_api"] = modulo
    especificacion.loader.exec_module(modulo)
    logger.info("API de Fase 3 cargada desde %s", ruta)
    return modulo


fase3_api = _cargar_api_fase3()
app = fase3_api.app

# Ruta del registro. En Docker se define LOG_PATH=/app/logs/predicciones.csv y
# la carpeta se monta como volumen para conservar el historial entre ejecuciones.
LOG_PATH = Path(
    os.getenv("LOG_PATH", str(FASE4_DIR / "logs" / "predicciones.csv"))
)
_lock_registro = threading.Lock()


def _ahora() -> str:
    """Fecha y hora local en formato ISO 8601, con segundos."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def registrar_evento(
    estado_http: int,
    version: str = "",
    prediccion: str = "",
) -> None:
    """Añade una fila al CSV, creando el encabezado si el archivo es nuevo."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock_registro:
        archivo_nuevo = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        with LOG_PATH.open("a", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(
                archivo, fieldnames=CAMPOS_REGISTRO, lineterminator="\n"
            )
            if archivo_nuevo:
                escritor.writeheader()
            escritor.writerow(
                {
                    "fecha_hora": _ahora(),
                    "estado_http": int(estado_http),
                    "version": version,
                    "prediccion": prediccion,
                }
            )


def _registrar_sin_interrumpir(
    estado_http: int,
    version: str = "",
    prediccion: str = "",
) -> None:
    """El monitoreo nunca debe tumbar una predicción por un fallo de escritura."""
    try:
        registrar_evento(estado_http, version, prediccion)
    except Exception:  # el registro nunca debe tumbar una predicción
        logger.exception("No se pudo escribir el registro de predicciones en %s", LOG_PATH)


def _extraer_resultado(cuerpo: bytes) -> tuple[str, str]:
    """Obtiene la versión y las etiquetas predichas de una respuesta exitosa.

    Admite las dos formas de ``/predict``: el lote (``model_version`` +
    ``predictions``) y el registro único (``version`` + ``prediccion``).
    """
    try:
        datos: dict[str, Any] = json.loads(cuerpo)
    except (json.JSONDecodeError, TypeError):
        logger.warning("La respuesta de /predict no se pudo interpretar como JSON")
        return "", ""

    if not isinstance(datos, dict):
        return "", ""

    if "predictions" in datos or "model_version" in datos:
        version = str(datos.get("model_version") or "")
        predicciones = datos.get("predictions") or []
        if not isinstance(predicciones, list):
            return version, ""
        etiquetas = [
            str(prediccion.get("churn_prediction", ""))
            for prediccion in predicciones
            if isinstance(prediccion, dict)
        ]
        return version, ";".join(etiquetas)

    # Registro único: {"prediccion": 1, "version": "..."}
    version = str(datos.get("version") or "")
    prediccion = datos.get("prediccion")
    return version, "" if prediccion is None else str(prediccion)


class RegistroPrediccionesMiddleware(BaseHTTPMiddleware):
    """Registra una única fila por cada intento de ``POST /predict``."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method != "POST" or request.url.path.rstrip("/") != "/predict":
            return await call_next(request)

        try:
            response = await call_next(request)
        except Exception:
            # Un error no controlado termina en HTTP 500 y también debe quedar registrado.
            _registrar_sin_interrumpir(500)
            raise

        cuerpo = b""
        async for bloque in response.body_iterator:
            cuerpo += bloque

        version, prediccion = "", ""
        if response.status_code < 400:
            version, prediccion = _extraer_resultado(cuerpo)

        _registrar_sin_interrumpir(response.status_code, version, prediccion)

        # Se reconstruye la respuesta después de leer el cuerpo para el registro.
        encabezados = {
            clave: valor
            for clave, valor in response.headers.items()
            if clave.lower() != "content-length"
        }
        return Response(
            content=cuerpo,
            status_code=response.status_code,
            headers=encabezados,
            background=response.background,
        )


app.add_middleware(RegistroPrediccionesMiddleware)

__all__ = ["app", "fase3_api", "registrar_evento", "LOG_PATH", "CAMPOS_REGISTRO"]
