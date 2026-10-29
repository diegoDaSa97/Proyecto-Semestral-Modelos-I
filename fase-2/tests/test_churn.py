"""Pruebas básicas de train.py y predict.py con un dataset sintético pequeño.

Usan datos generados en memoria (misma estructura que Telco Customer Churn),
por lo que corren en pocos segundos y no dependen del CSV real.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

import predict
import train

COLUMNAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def generar_datos(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético con señal: más churn con contrato mensual y poca antigüedad."""
    rng = np.random.default_rng(seed)
    yes_no = lambda: rng.choice(["Yes", "No"], n)
    tenure = rng.integers(0, 73, n)
    contrato = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.2, 0.2])
    mensual = rng.uniform(20, 110, n).round(2)
    prob = 0.15 + 0.5 * (contrato == "Month-to-month") * (tenure < 24)
    churn = np.where(rng.random(n) < prob, "Yes", "No")
    df = pd.DataFrame({
        "customerID": [f"ID-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yes_no(), "Dependents": yes_no(), "tenure": tenure,
        "PhoneService": yes_no(),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": yes_no(), "OnlineBackup": yes_no(), "DeviceProtection": yes_no(),
        "TechSupport": yes_no(), "StreamingTV": yes_no(), "StreamingMovies": yes_no(),
        "Contract": contrato, "PaperlessBilling": yes_no(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges": mensual,
        "TotalCharges": (mensual * tenure).round(2).astype(str),
        "Churn": churn,
    })
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # como en el dataset real
    df.loc[0, "TotalCharges"] = " "  # garantiza al menos un faltante
    return df


@pytest.fixture()
def csv_entrenamiento(tmp_path):
    ruta = tmp_path / "datos.csv"
    generar_datos().to_csv(ruta, index=False)
    return ruta


@pytest.fixture()
def modelo_entrenado(csv_entrenamiento, tmp_path):
    models_dir = tmp_path / "models"
    train.entrenar(csv_entrenamiento, models_dir, version="20260101_000000")
    return models_dir


# ---------- limpieza de datos ----------

def test_limpieza_convierte_espacios_a_nan_y_quita_columnas_no_predictoras():
    limpio = train.limpiar_predictoras(generar_datos(50))
    assert "customerID" not in limpio.columns and "Churn" not in limpio.columns
    assert pd.api.types.is_numeric_dtype(limpio["TotalCharges"])
    assert limpio["TotalCharges"].isna().sum() >= 1


# ---------- train.py ----------

def test_train_genera_modelo_y_metadatos_versionados(modelo_entrenado):
    modelo = modelo_entrenado / "churn_model_20260101_000000.joblib"
    meta = modelo_entrenado / "churn_model_20260101_000000.json"
    assert modelo.exists() and meta.exists()

    datos = json.loads(meta.read_text(encoding="utf-8"))
    assert datos["version"] == "20260101_000000"
    assert datos["model_type"] == "random_forest"
    assert 0.5 < datos["metrics"]["roc_auc"] <= 1.0
    assert datos["n_train"] + datos["n_test"] == 400
    assert "scikit-learn" in datos["environment"]
    assert len(datos["dataset"]["sha256"]) == 64


def test_train_es_reproducible_con_la_misma_semilla(csv_entrenamiento, tmp_path):
    a = train.entrenar(csv_entrenamiento, tmp_path / "a", seed=42, version="v1")
    b = train.entrenar(csv_entrenamiento, tmp_path / "b", seed=42, version="v1")
    assert a["metrics"] == b["metrics"]


def test_train_no_sobrescribe_una_version_existente(csv_entrenamiento, tmp_path):
    train.entrenar(csv_entrenamiento, tmp_path, version="v1")
    with pytest.raises(FileExistsError):
        train.entrenar(csv_entrenamiento, tmp_path, version="v1")


def test_train_falla_si_no_existe_el_dataset(tmp_path):
    assert train.main(["--data", str(tmp_path / "no_existe.csv"),
                       "--models-dir", str(tmp_path)]) == 1


def test_pipeline_no_ajusta_con_datos_de_prueba(csv_entrenamiento, tmp_path):
    """La mediana imputada debe provenir solo del conjunto de entrenamiento."""
    X, y = train.cargar_datos(csv_entrenamiento)
    X_tr, X_te, y_tr, _ = train.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = train.construir_pipeline(list(X.columns), "random_forest", 42).fit(X_tr, y_tr)
    imputador = pipe.named_steps["preprocesador"].named_transformers_["num"].named_steps["imputer"]
    esperado = X_tr[train.COLUMNAS_NUMERICAS].median().to_numpy()
    assert np.allclose(imputador.statistics_, esperado)


# ---------- predict.py ----------

def test_predict_devuelve_una_fila_por_cliente_con_probabilidades_validas(modelo_entrenado):
    modelo = joblib.load(modelo_entrenado / "churn_model_20260101_000000.joblib")
    nuevos = generar_datos(30, seed=1).drop(columns=["Churn"])
    resultado = predict.predecir(modelo, nuevos)

    assert len(resultado) == 30
    assert list(resultado.columns) == ["customerID", "churn_prediction", "churn_probability"]
    assert resultado["churn_probability"].between(0, 1).all()
    assert set(resultado["churn_prediction"]) <= {0, 1}


def test_predict_tolera_nulos_y_categorias_desconocidas(modelo_entrenado):
    modelo = joblib.load(modelo_entrenado / "churn_model_20260101_000000.joblib")
    nuevos = generar_datos(5, seed=2).drop(columns=["Churn"])
    nuevos.loc[0, "TotalCharges"] = " "
    nuevos.loc[1, "TotalCharges"] = None
    nuevos.loc[2, "Contract"] = "Contrato nuevo nunca visto"
    resultado = predict.predecir(modelo, nuevos)
    assert resultado["churn_probability"].notna().all()


def test_predict_falla_con_mensaje_claro_si_faltan_columnas(modelo_entrenado):
    modelo = joblib.load(modelo_entrenado / "churn_model_20260101_000000.joblib")
    incompleto = generar_datos(5).drop(columns=["Churn", "Contract", "tenure"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        predict.predecir(modelo, incompleto)


def test_predict_usa_el_modelo_mas_reciente(csv_entrenamiento, tmp_path):
    train.entrenar(csv_entrenamiento, tmp_path, version="20250101_000000")
    train.entrenar(csv_entrenamiento, tmp_path, version="20260101_000000")
    elegido = predict.encontrar_modelo_mas_reciente(tmp_path)
    assert elegido.name == "churn_model_20260101_000000.joblib"


def test_predict_sin_modelos_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        predict.encontrar_modelo_mas_reciente(tmp_path)


def test_predict_cli_extremo_a_extremo(modelo_entrenado, tmp_path):
    entrada = tmp_path / "nuevos.csv"
    generar_datos(10, seed=3).drop(columns=["Churn"]).to_csv(entrada, index=False)
    salida = tmp_path / "out" / "pred.csv"

    codigo = predict.main(["--input", str(entrada), "--output", str(salida),
                           "--models-dir", str(modelo_entrenado)])

    assert codigo == 0
    assert len(pd.read_csv(salida)) == 10


def test_predict_cli_falla_si_no_existe_la_entrada(modelo_entrenado, tmp_path):
    assert predict.main(["--input", str(tmp_path / "x.csv"),
                         "--models-dir", str(modelo_entrenado)]) == 1
