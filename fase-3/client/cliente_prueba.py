"""Cliente sencillo para probar la API REST desde la terminal.

Ejemplos:
    python client/cliente_prueba.py
    python client/cliente_prueba.py --entrenar
    python client/cliente_prueba.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parents[1]
REQUEST_FILE = BASE_DIR / "examples" / "solicitud_predict.json"


def llamar_api(url: str, metodo: str = "GET", payload: dict | None = None) -> dict:
    datos = None if payload is None else json.dumps(payload).encode("utf-8")
    solicitud = Request(
        url,
        data=datos,
        headers={"Content-Type": "application/json"},
        method=metodo,
    )
    try:
        with urlopen(solicitud, timeout=120) as respuesta:
            return json.loads(respuesta.read().decode("utf-8"))
    except HTTPError as error:
        detalle = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"La API respondió HTTP {error.code}: {detalle}") from error
    except URLError as error:
        raise RuntimeError(f"No fue posible conectar con la API: {error.reason}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente de prueba de Telco Customer Churn API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL base de la API")
    parser.add_argument(
        "--entrenar", action="store_true",
        help="Ejecuta POST /train antes de probar POST /predict.",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    try:
        print("\n1. Consultando GET /health")
        print(json.dumps(llamar_api(f"{base_url}/health"), indent=2, ensure_ascii=False))

        if args.entrenar:
            print("\n2. Ejecutando POST /train")
            resultado_train = llamar_api(
                f"{base_url}/train",
                "POST",
                {"model": "random_forest", "test_size": 0.2, "seed": 42},
            )
            print(json.dumps(resultado_train, indent=2, ensure_ascii=False))

        print("\n3. Ejecutando POST /predict")
        solicitud = json.loads(REQUEST_FILE.read_text(encoding="utf-8"))
        resultado = llamar_api(f"{base_url}/predict", "POST", solicitud)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
