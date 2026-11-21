"""Calcula métricas de operación a partir del registro de la API (Fase 4).

Lee el CSV generado por ``fase-4/app/main.py`` y muestra:

- total de solicitudes registradas;
- total y porcentaje de solicitudes fallidas (HTTP 4xx o 5xx).

Uso:
    python monitor.py
    python monitor.py --log logs/predicciones.csv
    python monitor.py --log examples/registro_muestra.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_POR_DEFECTO = Path(
    os.getenv("LOG_PATH", str(BASE_DIR / "logs" / "predicciones.csv"))
)


def leer_registro(ruta: Path) -> list[dict[str, str]]:
    """Devuelve las filas de datos del CSV; un archivo inexistente equivale a vacío."""
    if not ruta.is_file():
        return []
    with ruta.open("r", encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        return [
            fila
            for fila in lector
            if fila.get("estado_http") not in (None, "")
        ]


def calcular_metricas(filas: list[dict[str, str]]) -> tuple[int, int, float | None]:
    """Calcula (total, fallidas, porcentaje). Porcentaje None si no hay filas."""
    total = len(filas)
    fallidas = 0
    for fila in filas:
        try:
            estado = int(fila["estado_http"])
        except (KeyError, TypeError, ValueError):
            continue
        if 400 <= estado <= 599:
            fallidas += 1
    porcentaje = None if total == 0 else (fallidas / total) * 100
    return total, fallidas, porcentaje


def formatear_porcentaje(porcentaje: float | None) -> str:
    """Formato en español: 33,3 % o «sin solicitudes»."""
    if porcentaje is None:
        return "sin solicitudes"
    return f"{porcentaje:.1f} %".replace(".", ",")


def generar_reporte(ruta: Path) -> str:
    filas = leer_registro(ruta)
    total, fallidas, porcentaje = calcular_metricas(filas)
    estado_archivo = "" if ruta.is_file() else " (archivo inexistente, se interpreta como vacío)"
    return "\n".join(
        [
            f"Registro: {ruta}{estado_archivo}",
            f"Solicitudes totales: {total}",
            f"Solicitudes fallidas (4xx/5xx): {fallidas}",
            f"Porcentaje de fallos: {formatear_porcentaje(porcentaje)}",
        ]
    )


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula total de solicitudes y porcentaje de fallos de la API."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_POR_DEFECTO,
        help="Ruta del CSV de registro (por defecto LOG_PATH o logs/predicciones.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argumentos = parsear_argumentos(argv)
    print(generar_reporte(argumentos.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
