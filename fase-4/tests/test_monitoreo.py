"""Pruebas del registro de /predict y de las métricas de operación (Fase 4)."""

from __future__ import annotations

import csv
import shutil

from conftest import generar_dataset

import monitor
from app import main


def entrenar_modelo(client, version: str = "20261121_120001") -> dict:
    respuesta = client.post(
        "/train",
        json={
            "model": "random_forest",
            "test_size": 0.2,
            "seed": 42,
            "version": version,
        },
    )
    assert respuesta.status_code == 200, respuesta.text
    return respuesta.json()


def leer_registro() -> list[dict[str, str]]:
    assert main.LOG_PATH.is_file(), "El registro no se creó"
    with main.LOG_PATH.open("r", encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def solicitud_valida(customer_id: str = "DEMO-001") -> dict:
    cliente = generar_dataset(n=1, seed=99).drop(columns=["Churn"])
    cliente.loc[cliente.index[0], "customerID"] = customer_id
    return {"customers": cliente.to_dict(orient="records"), "threshold": 0.5}


def test_health_responde_200_sin_modelo(api_client):
    respuesta = api_client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "degraded"


def test_predict_valido_registra_version_y_prediccion(api_client):
    entrenar_modelo(api_client)

    respuesta = api_client.post("/predict", json=solicitud_valida())

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    filas = leer_registro()
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "200"
    assert filas[0]["version"] == cuerpo["model_version"] == "20261121_120001"
    assert filas[0]["prediccion"] == str(cuerpo["predictions"][0]["churn_prediction"])
    assert filas[0]["fecha_hora"]


def test_predict_invalido_tambien_se_registra(api_client):
    entrenar_modelo(api_client)

    respuesta = api_client.post(
        "/predict",
        json={"customers": [{"gender": "Female"}], "threshold": 0.5},
    )

    assert respuesta.status_code == 422
    assert "missing_features" in respuesta.json()["detail"]
    filas = leer_registro()
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "422"
    assert filas[0]["version"] == ""
    assert filas[0]["prediccion"] == ""


def test_predict_sin_modelo_registra_503(api_client):
    respuesta = api_client.post("/predict", json=solicitud_valida())

    assert respuesta.status_code == 503
    filas = leer_registro()
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "503"
    assert filas[0]["version"] == ""
    assert filas[0]["prediccion"] == ""


def test_predict_registro_unico_se_registra(api_client):
    entrenar_modelo(api_client)
    registro = (
        generar_dataset(n=1, seed=99)
        .drop(columns=["Churn"])
        .to_dict(orient="records")[0]
    )

    respuesta = api_client.post("/predict", json=registro)

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"prediccion", "version"}
    filas = leer_registro()
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "200"
    assert filas[0]["version"] == cuerpo["version"] == "20261121_120001"
    assert filas[0]["prediccion"] == str(cuerpo["prediccion"])


def test_dos_validas_y_una_invalida_producen_tres_filas(api_client):
    entrenar_modelo(api_client)

    assert api_client.post("/predict", json=solicitud_valida("DEMO-001")).status_code == 200
    assert api_client.post("/predict", json=solicitud_valida("DEMO-002")).status_code == 200
    invalida = solicitud_valida("DEMO-003")
    del invalida["customers"][0]["tenure"]
    assert api_client.post("/predict", json=invalida).status_code == 422

    filas = leer_registro()
    assert len(filas) == 3
    total, fallidas, porcentaje = monitor.calcular_metricas(filas)
    assert total == 3
    assert fallidas == 1
    assert round(porcentaje, 1) == 33.3
    assert monitor.formatear_porcentaje(porcentaje) == "33,3 %"


def test_monitor_sin_solicitudes(tmp_path):
    ruta = tmp_path / "vacio.csv"

    reporte = monitor.generar_reporte(ruta)

    assert "Solicitudes totales: 0" in reporte
    assert "Porcentaje de fallos: sin solicitudes" in reporte


def test_seleccionar_una_version_anterior(api_client, monkeypatch):
    entrenar_modelo(api_client, version="20261121_120001")
    entrenar_modelo(api_client, version="20261121_120002")
    assert api_client.get("/health").json()["model_version"] == "20261121_120002"

    # Simula un reinicio con MODEL_VERSION fijada a la versión anterior.
    monkeypatch.setenv("MODEL_VERSION", "20261121_120001")
    main.app.state.model = None
    main.app.state.model_version = None
    main.app.state.model_path = None

    health = api_client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == "20261121_120001"

    respuesta = api_client.post("/predict", json=solicitud_valida())
    assert respuesta.status_code == 200
    assert respuesta.json()["model_version"] == "20261121_120001"
    assert leer_registro()[-1]["version"] == "20261121_120001"


def test_seleccionar_modelo_inicial_de_fase1(api_client, monkeypatch, tmp_path):
    entrenar_modelo(api_client)
    origen = main.fase3_api.MODEL_DIR / "churn_model_20261121_120001.joblib"
    destino = tmp_path / "legacy.joblib"
    shutil.copyfile(origen, destino)
    monkeypatch.setattr(main.fase3_api, "LEGACY_MODEL_PATH", destino)
    monkeypatch.setenv("MODEL_VERSION", "fase-1-inicial")
    main.app.state.model = None
    main.app.state.model_version = None
    main.app.state.model_path = None

    health = api_client.get("/health")

    assert health.status_code == 200
    assert health.json()["model_version"] == "fase-1-inicial"
