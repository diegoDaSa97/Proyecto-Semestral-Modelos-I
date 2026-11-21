"""Fixtures y datos sintéticos para las pruebas de la Fase 4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main


def generar_dataset(n: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contrato = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.60, 0.20, 0.20]
    )
    mensual = rng.uniform(20, 115, n).round(2)
    probabilidad = 0.12 + 0.58 * ((contrato == "Month-to-month") & (tenure < 24))
    churn = np.where(rng.random(n) < probabilidad, "Yes", "No")

    def yes_no() -> np.ndarray:
        return rng.choice(["Yes", "No"], n)

    datos = pd.DataFrame(
        {
            "customerID": [f"TEST-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yes_no(),
            "Dependents": yes_no(),
            "tenure": tenure,
            "PhoneService": yes_no(),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": contrato,
            "PaperlessBilling": yes_no(),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": mensual,
            "TotalCharges": (mensual * tenure).round(2).astype(str),
            "Churn": churn,
        }
    )
    datos.loc[datos.index[:3], "TotalCharges"] = " "
    return datos


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """Cliente FastAPI con rutas temporales, dataset sintético y registro aislado."""
    ruta_datos = tmp_path / "Telco-Customer-Churn.csv"
    generar_dataset().to_csv(ruta_datos, index=False)
    carpeta_modelos = tmp_path / "models"
    ruta_registro = tmp_path / "logs" / "predicciones.csv"

    monkeypatch.setattr(main.fase3_api, "DATA_PATH", ruta_datos)
    monkeypatch.setattr(main.fase3_api, "MODEL_DIR", carpeta_modelos)
    monkeypatch.setattr(
        main.fase3_api, "LEGACY_MODEL_PATH", tmp_path / "modelo_legacy_inexistente.joblib"
    )
    monkeypatch.setattr(main, "LOG_PATH", ruta_registro)
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    main.app.state.model = None
    main.app.state.model_version = None
    main.app.state.model_path = None

    with TestClient(main.app) as client:
        yield client

    main.app.state.model = None
    main.app.state.model_version = None
    main.app.state.model_path = None
