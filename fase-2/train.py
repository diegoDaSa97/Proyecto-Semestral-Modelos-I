"""Entrena el modelo de churn (Telco Customer Churn) desde la línea de comandos.

Reproduce el flujo de la Fase 1 (partición estratificada antes de transformar,
imputación/escalado/One-Hot dentro de un Pipeline) y guarda el modelo con una
versión basada en la fecha y hora de entrenamiento, junto con un archivo de
metadatos (.json).

Ejemplo:
    python train.py
    python train.py --model logistic_regression --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODELS_DIR = BASE_DIR / "models"

COLUMNA_OBJETIVO = "Churn"
COLUMNA_ID = "customerID"
COLUMNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
MODELOS_DISPONIBLES = ("random_forest", "logistic_regression")

logger = logging.getLogger("train")


def limpiar_predictoras(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza aplicada igual en entrenamiento y en predicción.

    - Elimina el identificador del cliente y la variable objetivo, si existen.
    - Convierte `TotalCharges` a numérico (los espacios en blanco pasan a NaN).
    - Convierte `SeniorCitizen` a numérico (0/1), como se usó en la Fase 1.

    No calcula estadísticos: la imputación y el escalado viven en el Pipeline.
    """
    df = df.drop(columns=[COLUMNA_ID, COLUMNA_OBJETIVO], errors="ignore").copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    return df


def cargar_datos(ruta: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el CSV y devuelve (X, y) con y en {0, 1}."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta}")
    df = pd.read_csv(ruta)
    if COLUMNA_OBJETIVO not in df.columns:
        raise ValueError(f"El dataset no contiene la columna objetivo '{COLUMNA_OBJETIVO}'.")
    y = (df[COLUMNA_OBJETIVO] == "Yes").astype(int)
    X = limpiar_predictoras(df)
    faltantes = [c for c in COLUMNAS_NUMERICAS if c not in X.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas numéricas esperadas: {faltantes}")
    return X, y


def construir_pipeline(columnas: list[str], tipo_modelo: str, seed: int) -> Pipeline:
    """Pipeline completo (preprocesamiento + clasificador), igual al de la Fase 1."""
    columnas_cat = [c for c in columnas if c not in COLUMNAS_NUMERICAS]

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                COLUMNAS_NUMERICAS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                columnas_cat,
            ),
        ]
    )

    if tipo_modelo == "random_forest":
        clasificador = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
    elif tipo_modelo == "logistic_regression":
        clasificador = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Modelo no soportado: {tipo_modelo}. Opciones: {MODELOS_DISPONIBLES}")

    return Pipeline([("preprocesador", preprocesador), ("clasificador", clasificador)])


def evaluar(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Métricas sobre el conjunto de prueba (umbral 0.5)."""
    probabilidades = pipeline.predict_proba(X_test)[:, 1]
    predicciones = (probabilidades >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilidades)), 4),
        "accuracy": round(float(accuracy_score(y_test, predicciones)), 4),
        "precision": round(float(precision_score(y_test, predicciones)), 4),
        "recall": round(float(recall_score(y_test, predicciones)), 4),
        "f1": round(float(f1_score(y_test, predicciones)), 4),
    }


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def entrenar(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
    tipo_modelo: str = "random_forest",
    test_size: float = 0.20,
    seed: int = 42,
    version: str | None = None,
) -> dict:
    """Entrena, evalúa y guarda el modelo versionado. Devuelve los metadatos."""
    version = version or datetime.now().strftime("%Y%m%d_%H%M%S")

    X, y = cargar_datos(data_path)
    logger.info("Datos cargados: %d filas, %d predictoras", len(X), X.shape[1])

    # La partición se hace ANTES de ajustar cualquier transformación (sin fuga de datos).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipeline = construir_pipeline(list(X.columns), tipo_modelo, seed)
    pipeline.fit(X_train, y_train)
    metricas = evaluar(pipeline, X_test, y_test)
    logger.info("Métricas en prueba: %s", metricas)

    models_dir.mkdir(parents=True, exist_ok=True)
    nombre_base = f"churn_model_{version}"
    ruta_modelo = models_dir / f"{nombre_base}.joblib"
    ruta_meta = models_dir / f"{nombre_base}.json"
    if ruta_modelo.exists():
        raise FileExistsError(f"Ya existe una versión del modelo: {ruta_modelo.name}")

    joblib.dump(pipeline, ruta_modelo)

    metadatos = {
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": tipo_modelo,
        "hyperparameters": pipeline.named_steps["clasificador"].get_params(),
        "metrics": metricas,
        "seed": seed,
        "test_size": test_size,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
        "dataset": {"file": data_path.name, "sha256": _sha256(data_path)},
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "model_file": ruta_modelo.name,
    }
    ruta_meta.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Modelo guardado en: %s", ruta_modelo)
    logger.info("Metadatos guardados en: %s", ruta_meta)
    return metadatos


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de churn y lo guarda versionado.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Ruta al CSV de entrenamiento.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Carpeta donde se guarda el modelo.")
    parser.add_argument("--model", choices=MODELOS_DISPONIBLES, default="random_forest",
                        help="Algoritmo a entrenar (por defecto, el elegido en la Fase 1).")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporción de prueba (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--version", default=None,
                        help="Versión del modelo. Por defecto, fecha y hora: AAAAMMDD_HHMMSS.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parsear_argumentos(argv)
    try:
        entrenar(args.data, args.models_dir, args.model, args.test_size, args.seed, args.version)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
