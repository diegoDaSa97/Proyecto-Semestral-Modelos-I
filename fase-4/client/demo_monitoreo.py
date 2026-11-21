"""Cliente de demostración de la Fase 4.

Envía a la API dos solicitudes válidas y una inválida (sin la variable
obligatoria ``tenure``), y luego muestra el reporte del registro. Sirve para
comprobar que cada intento produce exactamente una fila.

Uso (con la API en marcha):
    python fase-4/client/demo_monitoreo.py
    python fase-4/client/demo_monitoreo.py --url http://localhost:8000
    python fase-4/client/demo_monitoreo.py --log fase-4/logs/predicciones.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from monitor import LOG_POR_DEFECTO, generar_reporte  # noqa: E402

CLIENTE_BASE = {
    "customerID": "DEMO-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.9,
    "TotalCharges": 179.8,
}


def enviar(url: str, cuerpo: dict) -> tuple[int, dict]:
    """Envía un JSON y devuelve (código HTTP, cuerpo de la respuesta)."""
    solicitud = urllib.request.Request(
        url,
        data=json.dumps(cuerpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=60) as respuesta:
            return respuesta.status, json.loads(respuesta.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Envía 2 predicciones válidas y 1 inválida, y resume el registro."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL base de la API.")
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_POR_DEFECTO,
        help="Ruta del CSV de registro que se mostrará al final.",
    )
    parser.add_argument(
        "--sin-monitor",
        action="store_true",
        help="Solo envía las solicitudes; no imprime el reporte.",
    )
    argumentos = parser.parse_args(argv)

    cliente_1 = copy.deepcopy(CLIENTE_BASE)
    cliente_2 = copy.deepcopy(CLIENTE_BASE)
    cliente_2.update(
        {
            "customerID": "DEMO-002",
            "tenure": 36,
            "Contract": "Two year",
            "MonthlyCharges": 45.5,
            "TotalCharges": 1638.0,
        }
    )
    cliente_invalido = copy.deepcopy(CLIENTE_BASE)
    cliente_invalido["customerID"] = "DEMO-003"
    cliente_invalido.pop("tenure")  # variable obligatoria ausente

    # Cada caso es un registro único: las variables van en la raíz del cuerpo.
    casos = [
        ("válida 1", {**cliente_1, "threshold": 0.5}),
        ("válida 2", {**cliente_2, "threshold": 0.5}),
        ("inválida (sin tenure)", {**cliente_invalido, "threshold": 0.5}),
    ]

    print(f"API: {argumentos.url}")
    for nombre, cuerpo in casos:
        estado, respuesta = enviar(f"{argumentos.url.rstrip('/')}/predict", cuerpo)
        if estado == 200 and "prediccion" in respuesta:
            detalle = f"version={respuesta['version']} prediccion={respuesta['prediccion']}"
        elif estado == 200:
            prediccion = respuesta["predictions"][0]
            detalle = (
                f"version={respuesta['model_version']} "
                f"churn_prediction={prediccion['churn_prediction']} "
                f"churn_probability={prediccion['churn_probability']}"
            )
        else:
            detalle = json.dumps(respuesta.get("detail", respuesta), ensure_ascii=False)
        print(f"- {nombre}: HTTP {estado} | {detalle}")

    if not argumentos.sin_monitor:
        print()
        print(generar_reporte(argumentos.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
