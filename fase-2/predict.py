"""Genera predicciones de churn a partir de un CSV con clientes nuevos.

Usa por defecto la versión más reciente del modelo guardada en `models/`.
El CSV debe tener las mismas columnas predictoras del dataset original
(`customerID` y `Churn` son opcionales y se ignoran como predictoras).

Ejemplos:
    python predict.py --input data/ejemplo_nuevos_clientes.csv
    python predict.py --input data/ejemplo_nuevos_clientes.csv --output output/predicciones.csv
    python predict.py --input clientes.csv --model-path models/churn_model_20261027_210000.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

# Se reutiliza la MISMA limpieza del entrenamiento para evitar diferencias entre train y predict.
from train import COLUMNA_ID, MODELS_DIR, limpiar_predictoras

logger = logging.getLogger("predict")


def encontrar_modelo_mas_reciente(models_dir: Path) -> Path:
    """Devuelve el .joblib más reciente (la versión AAAAMMDD_HHMMSS ordena cronológicamente)."""
    modelos = sorted(models_dir.glob("churn_model_*.joblib"))
    if not modelos:
        raise FileNotFoundError(
            f"No hay modelos en '{models_dir}'. Entrene primero con: python train.py"
        )
    return modelos[-1]


def cargar_modelo(ruta: Path):
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta}")
    return joblib.load(ruta)


def predecir(modelo, df_crudo: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    """Devuelve un DataFrame con la predicción y la probabilidad de churn por fila.

    Si el CSV trae `customerID`, se conserva como primera columna del resultado.
    """
    if not 0.0 <= umbral <= 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1.")

    X = limpiar_predictoras(df_crudo)
    esperadas = list(modelo.feature_names_in_)
    faltantes = [c for c in esperadas if c not in X.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")
    X = X[esperadas]  # mismo orden de columnas que en el entrenamiento

    probabilidades = modelo.predict_proba(X)[:, 1]
    resultado = pd.DataFrame(
        {
            "churn_prediction": (probabilidades >= umbral).astype(int),
            "churn_probability": probabilidades.round(4),
        },
        index=df_crudo.index,
    )
    if COLUMNA_ID in df_crudo.columns:
        resultado.insert(0, COLUMNA_ID, df_crudo[COLUMNA_ID])
    return resultado


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predice churn para clientes nuevos.")
    parser.add_argument("--input", type=Path, required=True, help="CSV con los clientes a predecir.")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV de salida. Si se omite, se imprime en pantalla.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Modelo .joblib específico. Por defecto, el más reciente de --models-dir.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Carpeta con los modelos.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Umbral de probabilidad para clasificar churn (por defecto 0.5).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)
    args = parsear_argumentos(argv)
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {args.input}")
        ruta_modelo = args.model_path or encontrar_modelo_mas_reciente(args.models_dir)
        logger.info("Usando modelo: %s", ruta_modelo.name)
        modelo = cargar_modelo(ruta_modelo)

        df = pd.read_csv(args.input)
        if df.empty:
            raise ValueError("El archivo de entrada no tiene filas.")
        resultado = predecir(modelo, df, args.threshold)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(args.output, index=False)
        logger.info("Predicciones guardadas en: %s (%d filas)", args.output, len(resultado))
    else:
        print(resultado.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
