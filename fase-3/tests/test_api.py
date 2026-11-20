"""Pruebas de integración de los endpoints REST de la Fase 3."""

from __future__ import annotations

from conftest import generar_dataset


def entrenar_modelo(client, version: str = "20261120_120001") -> dict:
    respuesta = client.post(
        "/train",
        json={"model": "random_forest", "test_size": 0.2, "seed": 42, "version": version},
    )
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()


def test_health_informa_servicio_activo_sin_modelo(api_client):
    respuesta = api_client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "status": "degraded",
        "model_available": False,
        "model_version": None,
    }


def test_train_entrena_y_activa_una_version(api_client):
    resultado = entrenar_modelo(api_client)

    assert resultado["message"].startswith("Entrenamiento completado")
    assert resultado["version"] == "20261120_120001"
    assert resultado["model_type"] == "random_forest"
    assert 0 <= resultado["metrics"]["roc_auc"] <= 1
    assert resultado["model_file"].endswith(".joblib")

    health = api_client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_available"] is True
    assert health["model_version"] == "20261120_120001"


def test_predict_devuelve_probabilidad_y_clasificacion(api_client):
    entrenar_modelo(api_client)
    cliente = generar_dataset(n=1, seed=99).drop(columns=["Churn"])
    solicitud = {"customers": cliente.to_dict(orient="records"), "threshold": 0.5}

    respuesta = api_client.post("/predict", json=solicitud)

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["model_version"] == "20261120_120001"
    assert cuerpo["count"] == 1
    prediccion = cuerpo["predictions"][0]
    assert prediccion["churn_prediction"] in (0, 1)
    assert 0 <= prediccion["churn_probability"] <= 1
    assert "customerID" in prediccion


def test_predict_rechaza_variables_obligatorias_ausentes(api_client):
    entrenar_modelo(api_client)

    respuesta = api_client.post(
        "/predict",
        json={"customers": [{"gender": "Female"}], "threshold": 0.5},
    )

    assert respuesta.status_code == 422
    assert "missing_features" in respuesta.json()["detail"]


def test_predict_rechaza_umbral_fuera_de_rango(api_client):
    respuesta = api_client.post(
        "/predict",
        json={"customers": [{}], "threshold": 1.5},
    )

    assert respuesta.status_code == 422


def test_predict_sin_modelo_responde_503(api_client):
    respuesta = api_client.post("/predict", json={"customers": [{}]})

    assert respuesta.status_code == 503
    assert "No hay un modelo disponible" in respuesta.json()["detail"]


def test_train_no_sobrescribe_una_version_existente(api_client):
    entrenar_modelo(api_client, version="20261120_120002")

    respuesta = api_client.post("/train", json={"version": "20261120_120002"})

    assert respuesta.status_code == 409
    assert "Ya existe una versión" in respuesta.json()["detail"]


def test_train_rechaza_parametros_invalidos(api_client):
    respuesta = api_client.post("/train", json={"test_size": 1.0})

    assert respuesta.status_code == 422
