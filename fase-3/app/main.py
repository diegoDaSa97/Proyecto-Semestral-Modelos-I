"""API REST para el modelo Telco Customer Churn (Fase 3).

La API reutiliza los módulos train.py y predict.py de la Fase 2. Primero busca
modelos versionados en fase-2/models y, si todavía no existen, utiliza el
pipeline inicial almacenado en fase-1/models.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("churn_api")

# Resolución de rutas compatible con ejecución local y con Docker.
APP_DIR = Path(__file__).resolve().parent
API_DIR = APP_DIR.parent
PROJECT_ROOT = API_DIR.parent

if (PROJECT_ROOT / "fase-2" / "train.py").is_file():
    PHASE2_DIR = PROJECT_ROOT / "fase-2"
    DEFAULT_MODEL_DIR = PHASE2_DIR / "models"
    DEFAULT_DATA_PATH = PHASE2_DIR / "data" / "Telco-Customer-Churn.csv"
    DEFAULT_LEGACY_MODEL = (
        PROJECT_ROOT / "fase-1" / "models" / "churn_model_pipeline.joblib"
    )
else:
    # En la imagen Docker los scripts de Fase 2 se copian directamente a /app.
    PROJECT_ROOT = API_DIR
    PHASE2_DIR = PROJECT_ROOT
    DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
    DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
    DEFAULT_LEGACY_MODEL = (
        PROJECT_ROOT / "legacy_models" / "churn_model_pipeline.joblib"
    )

if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

# Importaciones deliberadas de la Fase 2: no se duplica la lógica de entrenamiento
# ni la preparación de variables que ya fue probada en esa fase.
from train import COLUMNAS_NUMERICAS, entrenar  # noqa: E402
from predict import encontrar_modelo_mas_reciente, predecir  # noqa: E402

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
DATA_PATH = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))
LEGACY_MODEL_PATH = Path(
    os.getenv("LEGACY_MODEL_PATH", str(DEFAULT_LEGACY_MODEL))
)


class PredictRequest(BaseModel):
    """Registro único o lote de clientes nuevos que se desean evaluar.

    Se aceptan las dos formas descritas en el enunciado:

    - **Registro único** (forma principal): las variables predictoras van
      directamente en la raíz, por ejemplo
      ``{"tenure": 2, "gender": "Female", ...}``. La respuesta es
      ``{"prediccion": 1, "version": "20261120_103934"}``.
    - **Lote** (opcional): ``{"customers": [ {...}, {...} ]}``. Se conserva para
      enviar varios clientes en una sola petición.

    ``threshold`` es opcional en ambos casos.
    """

    model_config = ConfigDict(extra="allow")

    customers: list[dict[str, Any]] | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Lote opcional de clientes. Si se omite, el cuerpo es un registro único.",
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Umbral de probabilidad para clasificar abandono.",
    )


class TrainRequest(BaseModel):
    """Parámetros permitidos para entrenar una nueva versión."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["random_forest", "logistic_regression"] = "random_forest"
    test_size: float = Field(default=0.20, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0)
    version: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
        pattern=r"^\d{8}_\d{6}$",
        description="Versión opcional en formato AAAAMMDD_HHMMSS, compatible con el orden cronológico de Fase 2.",
    )


app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Servicio REST para consultar el estado del modelo, predecir la fuga de "
        "clientes y entrenar nuevas versiones utilizando los componentes de la Fase 2."
    ),
    version="1.0.0",
)

app.state.model = None
app.state.model_version = None
app.state.model_path = None


def _buscar_archivo_modelo() -> Path:
    """Busca la versión configurada, la más reciente de Fase 2 o el modelo de Fase 1.

    La variable de entorno ``MODEL_VERSION`` permite fijar una versión concreta
    (por ejemplo ``20261120_103934`` o ``fase-1-inicial``). Si no se define, se
    conserva el comportamiento de la Fase 3: la versión más reciente de Fase 2
    y, en su defecto, el artefacto inicial de Fase 1.
    """
    version_configurada = os.getenv("MODEL_VERSION", "").strip()
    if version_configurada:
        if version_configurada == "fase-1-inicial":
            if LEGACY_MODEL_PATH.is_file():
                return LEGACY_MODEL_PATH
            raise FileNotFoundError(
                "Se configuró MODEL_VERSION=fase-1-inicial, pero no existe el modelo inicial de Fase 1."
            )

        ruta_version = MODEL_DIR / f"churn_model_{version_configurada}.joblib"
        if ruta_version.is_file():
            return ruta_version
        raise FileNotFoundError(
            f"No se encontró la versión configurada '{version_configurada}' en '{MODEL_DIR}'. "
            "Revise MODEL_VERSION o seleccione una versión disponible."
        )

    try:
        return encontrar_modelo_mas_reciente(MODEL_DIR)
    except FileNotFoundError:
        if LEGACY_MODEL_PATH.is_file():
            return LEGACY_MODEL_PATH
        raise FileNotFoundError(
            "No hay un modelo disponible. Ejecute POST /train o revise la ruta del modelo."
        )


def _version_del_modelo(ruta: Path) -> str:
    """Obtiene la versión desde los metadatos o desde el nombre del archivo."""
    if ruta.resolve() == LEGACY_MODEL_PATH.resolve():
        return "fase-1-inicial"

    ruta_meta = ruta.with_suffix(".json")
    if ruta_meta.is_file():
        try:
            metadatos = json.loads(ruta_meta.read_text(encoding="utf-8"))
            if metadatos.get("version"):
                return str(metadatos["version"])
        except (OSError, json.JSONDecodeError):
            logger.warning("No fue posible leer los metadatos de %s", ruta_meta)

    return ruta.stem.removeprefix("churn_model_")


def _cargar_modelo_desde_disco() -> tuple[Any, str, Path]:
    ruta = _buscar_archivo_modelo()
    modelo = joblib.load(ruta)
    return modelo, _version_del_modelo(ruta), ruta


def obtener_modelo_actual() -> tuple[Any, str]:
    """Devuelve el modelo en memoria o lo carga desde el almacenamiento."""
    if app.state.model is not None:
        return app.state.model, app.state.model_version or "sin-version"

    try:
        modelo, version, ruta = _cargar_modelo_desde_disco()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:  # archivo corrupto o incompatibilidad de serialización
        logger.exception("No se pudo cargar el modelo almacenado")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo existe, pero no pudo cargarse. Revise su versión y dependencias.",
        ) from error

    app.state.model = modelo
    app.state.model_version = version
    app.state.model_path = str(ruta)
    logger.info("Modelo cargado: %s (versión %s)", ruta, version)
    return modelo, version


@app.get("/health", tags=["Servicio"])
def health() -> dict[str, Any]:
    """Comprueba que la API responde y reporta si hay un modelo disponible."""
    try:
        _, version = obtener_modelo_actual()
        return {
            "status": "ok",
            "model_available": True,
            "model_version": version,
        }
    except HTTPException:
        return {
            "status": "degraded",
            "model_available": False,
            "model_version": None,
        }


def _columnas_esperadas(modelo: Any) -> list[str]:
    """Nombres de las variables de entrada que espera el modelo entrenado."""
    columnas = list(getattr(modelo, "feature_names_in_", []))
    if not columnas:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo almacenado no expone los nombres de sus variables de entrada.",
        )
    return columnas


def _verificar_columnas(modelo: Any, df: pd.DataFrame) -> None:
    """Comprueba que llegaron todas las variables y nombra las que falten."""
    faltantes = [c for c in _columnas_esperadas(modelo) if c not in df.columns]
    if faltantes:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Faltan variables obligatorias para realizar la predicción.",
                "missing_features": faltantes,
            },
        )


def _verificar_tipos(registros: list[dict[str, Any]]) -> None:
    """Rechaza valores no numéricos en las columnas numéricas, nombrando el campo.

    Se aceptan números y cadenas numéricas (compatibles con el CSV original).
    Un valor nulo, vacío o en blanco se considera faltante y se imputa con la
    regla guardada en la Fase 2, por lo que no es un error. Un valor como
    ``"abc"`` se rechaza con HTTP 422 indicando la variable.
    """
    for registro in registros:
        for columna in COLUMNAS_NUMERICAS:
            valor = registro.get(columna)
            if valor is None or (isinstance(valor, str) and valor.strip() == ""):
                continue
            try:
                convertido = pd.to_numeric(
                    pd.Series([valor]), errors="coerce"
                ).iloc[0]
            except (TypeError, ValueError):
                convertido = float("nan")
            if pd.isna(convertido):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": f"La variable '{columna}' debe ser numérica.",
                        "invalid_feature": columna,
                    },
                )


def _predecir_registros(
    modelo: Any, registros: list[dict[str, Any]], umbral: float
) -> pd.DataFrame:
    """Valida y predice un lote de registros con el mismo preprocesamiento."""
    df = pd.DataFrame(registros)
    _verificar_columnas(modelo, df)
    _verificar_tipos(registros)
    try:
        return predecir(modelo, df, umbral=umbral)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict", tags=["Predicción"])
def predict_endpoint(request: PredictRequest) -> dict[str, Any]:
    """Predice el abandono para un registro único o para un lote de clientes.

    - Registro único (por defecto): el cuerpo contiene las variables predictoras
      en la raíz y la respuesta es ``{"prediccion", "version"}``.
    - Lote: el cuerpo incluye ``customers`` y la respuesta es
      ``{"model_version", "count", "predictions"}``.
    """
    modelo, version = obtener_modelo_actual()

    if request.customers is not None:
        resultado = _predecir_registros(modelo, request.customers, request.threshold)
        return {
            "model_version": version,
            "count": int(len(resultado)),
            "predictions": jsonable_encoder(resultado.to_dict(orient="records")),
        }

    registro = request.model_extra or {}
    if not registro:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No se recibieron variables predictoras en el registro.",
                "missing_features": _columnas_esperadas(modelo),
            },
        )

    resultado = _predecir_registros(modelo, [registro], request.threshold)
    fila = resultado.iloc[0]
    return {
        "prediccion": int(fila["churn_prediction"]),
        "version": version,
    }


@app.post("/train", tags=["Entrenamiento"])
def train_endpoint(request: TrainRequest) -> dict[str, Any]:
    """Entrena y activa una nueva versión usando el dataset del servidor."""
    if request.version and (MODEL_DIR / f"churn_model_{request.version}.joblib").exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ya existe una versión del modelo: churn_model_{request.version}.joblib",
        )

    try:
        metadatos = entrenar(
            data_path=DATA_PATH,
            models_dir=MODEL_DIR,
            tipo_modelo=request.model,
            test_size=request.test_size,
            seed=request.seed,
            version=request.version,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No está disponible el dataset de entrenamiento: {error}",
        ) from error
    except FileExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    ruta_modelo = MODEL_DIR / metadatos["model_file"]
    try:
        modelo_nuevo = joblib.load(ruta_modelo)
    except Exception as error:
        logger.exception("El entrenamiento terminó, pero no se pudo cargar el artefacto nuevo")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El modelo se entrenó, pero no fue posible activarlo.",
        ) from error

    # La siguiente petición /predict utiliza inmediatamente la versión recién entrenada.
    app.state.model = modelo_nuevo
    app.state.model_version = str(metadatos["version"])
    app.state.model_path = str(ruta_modelo)

    return {
        "message": "Entrenamiento completado y modelo activado correctamente.",
        "version": metadatos["version"],
        "model_type": metadatos["model_type"],
        "metrics": metadatos["metrics"],
        "model_file": metadatos["model_file"],
    }
