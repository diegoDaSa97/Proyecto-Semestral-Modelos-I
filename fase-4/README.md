# Fase 4: Monitoreo – Telco Customer Churn

**Asignatura:** Modelos y Simulación de Sistemas I  
**Proyecto:** Predicción de fuga de clientes (*Customer Churn*) en telecomunicaciones.

## 1. Objetivo

Registrar el uso real de la API construida en la Fase 3 para saber cuántas solicitudes se atienden, qué proporción falla y qué versión del modelo está prediciendo. La Fase 4 no cambia el modelo ni el algoritmo: añade observabilidad sobre la API existente.

## 2. Estructura de la carpeta

```text
fase-4/
├── app/
│   ├── __init__.py
│   └── main.py                 # Carga la API de Fase 3 y añade el registro de /predict
├── client/
│   └── demo_monitoreo.py       # Envía 2 solicitudes válidas y 1 inválida
├── examples/
│   └── registro_muestra.csv    # Muestra del registro generado en la demostración
├── logs/
│   └── .gitkeep                # Aquí se escribe predicciones.csv (ignorado por Git)
├── tests/
│   ├── conftest.py             # Dataset sintético y registro aislado
│   └── test_monitoreo.py       # Pruebas del registro, métricas y selección de versión
├── Dockerfile                  # Imagen de la API con registro habilitado
├── Dockerfile.test             # Imagen auxiliar con pytest y httpx
├── monitor.py                  # Total de solicitudes y porcentaje de fallos
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 3. Relación con la Fase 3

`fase-4/app/main.py` **reutiliza** el objeto `app` de `fase-3/app/main.py`; no copia los endpoints. Así, `/health`, `/predict` y `/train` siguen teniendo una única implementación. Sobre esa aplicación se registra un middleware que:

1. Detecta cada `POST /predict`.
2. Espera la respuesta final del framework (incluidos los rechazos 422 de validación).
3. Extrae la versión y la predicción cuando la respuesta es exitosa. Admite las dos formas de `/predict`: el registro único (`{"prediccion", "version"}`) y el lote (`{"model_version", ..., "predictions"}`).
4. Añade **una sola fila** al CSV de registro.

La Fase 3 también incorpora la variable `MODEL_VERSION` para fijar la versión activa; la Fase 4 la usa para documentar el cambio entre versiones.

## 4. Registro de solicitudes

Cada intento de `POST /predict` genera una fila en `fase-4/logs/predicciones.csv` (o en la ruta indicada por `LOG_PATH`). Las columnas son:

| Columna | Contenido |
|---|---|
| `fecha_hora` | Fecha y hora en formato ISO 8601, con segundos (zona horaria local; UTC dentro del contenedor). |
| `estado_http` | Código HTTP devuelto (`200`, `422`, `503`, etc.). |
| `version` | Versión del modelo que produjo la respuesta. Vacía si no se usó modelo. |
| `prediccion` | Etiqueta predicha (`0` o `1`). Vacía en errores. Si se envía un lote, las etiquetas se separan con `;`. |

Propiedades del registro:

- Se abre en modo de agregar, por lo que **se conserva entre ejecuciones**.
- El encabezado se escribe solo al crear el archivo.
- Las solicitudes rechazadas por validación (HTTP 422) también se registran, con `version` y `prediccion` vacías.
- Un error no controlado se registra como HTTP 500 antes de propagarse.
- Si la escritura del registro falla, la predicción no se interrumpe; el error queda en el log del servidor.

## 5. Ejecución con Docker

El contexto de construcción es la **raíz del repositorio**, porque la imagen reutiliza las fases 1, 2 y 3.

```bash
docker build -f fase-4/Dockerfile -t churn-api-fase4 .
```

Iniciar la API con los modelos de Fase 2 y el registro persistente montados desde el host:

```bash
docker run -d --name churn-api-fase4 -p 8000:8000 \
  -v "${PWD}/fase-2/models:/app/models" \
  -v "${PWD}/fase-4/logs:/app/logs" \
  churn-api-fase4
```

En PowerShell se puede escribir en una sola línea:

```powershell
docker run -d --name churn-api-fase4 -p 8000:8000 -v "${PWD}/fase-2/models:/app/models" -v "${PWD}/fase-4/logs:/app/logs" churn-api-fase4
```

Comprobar el estado y la versión activa:

```bash
curl http://localhost:8000/health
```

La documentación interactiva queda en `http://localhost:8000/docs`.

### 5.1. Variables de entorno

| Variable | Valor por defecto en la imagen | Uso |
|---|---|---|
| `MODEL_DIR` | `/app/models` | Carpeta de versiones de Fase 2. |
| `DATA_PATH` | `/app/data/Telco-Customer-Churn.csv` | Dataset para `/train`. |
| `LEGACY_MODEL_PATH` | `/app/legacy_models/churn_model_pipeline.joblib` | Modelo inicial de Fase 1. |
| `LOG_PATH` | `/app/logs/predicciones.csv` | Archivo CSV de registro. |
| `MODEL_VERSION` | vacío (última versión) | Fija una versión concreta. |

## 6. Script de monitoreo

`monitor.py` lee el CSV y muestra el total de solicitudes y el porcentaje de fallos (HTTP 4xx o 5xx).

```bash
# Registro en vivo
python fase-4/monitor.py --log fase-4/logs/predicciones.csv

# Muestra incluida en el repositorio
python fase-4/monitor.py --log fase-4/examples/registro_muestra.csv

# Dentro del contenedor
docker exec churn-api-fase4 python monitor.py
```

Salida esperada con tres solicitudes y una fallida:

```text
Registro: fase-4/logs/predicciones.csv
Solicitudes totales: 3
Solicitudes fallidas (4xx/5xx): 1
Porcentaje de fallos: 33,3 %
```

Si el archivo no existe o está vacío:

```text
Solicitudes totales: 0
Solicitudes fallidas (4xx/5xx): 0
Porcentaje de fallos: sin solicitudes
```

## 7. Demostración: dos válidas y una inválida

El cliente de demostración envía dos predicciones válidas (como registros únicos) y una tercera a la que le falta `tenure`, y luego ejecuta el reporte:

```bash
# Con Python en el host y la API publicada en el puerto 8000
python fase-4/client/demo_monitoreo.py

# Sin Python en el host, desde dentro del contenedor
docker exec churn-api-fase4 python client/demo_monitoreo.py
```

La ruta del registro se puede cambiar con `--log`. Con `--sin-monitor` solo se envían las solicitudes.

## 8. Versiones y cómo volver a la anterior

Versiones disponibles en el repositorio:

| Identificador | Origen | Algoritmo | ROC-AUC de referencia |
|---|---|---|---|
| `20261120_103934` | Fase 2, `fase-2/models/` | Random Forest | 0.8426 |
| `fase-1-inicial` | Fase 1, `fase-1/models/churn_model_pipeline.joblib` | Pipeline inicial | No registrada |

Por defecto la API usa la versión más reciente de `fase-2/models/`; en esta entrega es **`20261120_103934`**. Para volver a la versión inicial de Fase 1, inicie el contenedor con:

```powershell
docker run -d --name churn-api-fase4-v1 -p 8001:8000 -e MODEL_VERSION=fase-1-inicial -e LOG_PATH=/app/logs/rollback.csv -v "${PWD}/fase-2/models:/app/models" -v "${PWD}/fase-4/logs:/app/logs" churn-api-fase4
```

Se usa `LOG_PATH=/app/logs/rollback.csv` para no mezclar las pruebas de la versión anterior con el registro principal. Para fijar cualquier versión de Fase 2 use su identificador, por ejemplo `-e MODEL_VERSION=20261120_103934`. El valor se lee al cargar el modelo, así que el cambio se aplica al reiniciar el servicio. Si la versión no existe, `/health` responde `degraded` y `/predict` responde HTTP 503 con el nombre de la versión buscada.

## 9. Pruebas automatizadas

```bash
# Opción 1: imagen auxiliar de pruebas
docker build -f fase-4/Dockerfile.test -t churn-api-fase4-test .
docker run --rm churn-api-fase4-test

# Opción 2: entorno local con Python
cd fase-4
pip install -r requirements-dev.txt
pytest -v
```

Las pruebas comprueban:

- `/health` responde 200.
- Una predicción válida (lote) devuelve predicción y versión, y deja una fila con estado 200.
- Una predicción válida (registro único) se registra con su `version` y su `prediccion`.
- Una solicitud sin `tenure` se rechaza con 422 y también se registra (versión y predicción vacías).
- Una predicción sin modelo se registra como 503.
- Dos válidas y una inválida producen exactamente tres filas y una tasa de fallos de 33,3 %.
- El monitor informa «sin solicitudes» cuando el registro está vacío.
- `MODEL_VERSION` permite volver a una versión anterior.

## 10. Reporte

### 10.1. Solicitudes enviadas

Se partió de un registro vacío y se ejecutó el cliente de demostración contra la API con la versión `20261120_103934`. Cada caso es un registro único:

| Intento | Cliente | Resultado |
|---|---|---|
| 1 | `DEMO-001` (válido) | HTTP 200, `prediccion` = `1` |
| 2 | `DEMO-002` (válido) | HTTP 200, `prediccion` = `0` |
| 3 | `DEMO-003` (sin `tenure`) | HTTP 422, `missing_features: ["tenure"]` |

### 10.2. Registro obtenido

```csv
fecha_hora,estado_http,version,prediccion
2026-11-21T21:41:56+00:00,200,20261120_103934,1
2026-11-21T21:41:56+00:00,200,20261120_103934,0
2026-11-21T21:41:56+00:00,422,,
```

La marca de tiempo queda en UTC porque la API se ejecutó dentro del contenedor; en una ejecución local conserva la zona horaria del equipo.

### 10.3. Indicadores

- **Solicitudes totales:** 3
- **Solicitudes fallidas (4xx/5xx):** 1
- **Porcentaje de fallos:** 33,3 %

### 10.4. Interpretación

La API procesó dos registros con la versión `20261120_103934`: `DEMO-001` con `prediccion` = `1` y `DEMO-002` con `prediccion` = `0`. El tercer intento, `DEMO-003`, se rechazó con HTTP 422 porque faltaba la variable `tenure`. La tasa de fallos fue 1/3 (33,3 %); el rechazo era el resultado esperado de la prueba de validación y no indica un problema del modelo.

### 10.5. Política de reentrenamiento

Cuando se acumulen 100 clientes nuevos con su resultado real de fuga, el equipo de modelo revisará el desempeño de la versión activa. Un integrante designado ejecutará `POST /train` para generar una candidata y comparará la ROC-AUC de la candidata y de la versión activa sobre el mismo conjunto reservado. La candidata se activará solo si mejora la ROC-AUC de la versión activa; en caso contrario se conserva la versión actual y se documenta la decisión.

### 10.6. Versiones identificadas

- **Versión activa durante la demostración:** `20261120_103934` (Fase 2, Random Forest).
- **Versión anterior disponible:** `fase-1-inicial` (Fase 1).
- **Forma de volver a la anterior:** `MODEL_VERSION=fase-1-inicial` al iniciar el contenedor (sección 8).
- **Muestra del registro:** `fase-4/examples/registro_muestra.csv`.

### 10.7. Entregables del monitoreo

- **Script de métricas:** `fase-4/monitor.py`.
- **Código de registro:** `fase-4/app/main.py` (middleware `RegistroPrediccionesMiddleware`).
- **Muestra del registro utilizado:** `fase-4/examples/registro_muestra.csv`.
- **Registro en vivo:** `fase-4/logs/predicciones.csv`.

## 11. Consideraciones

- El porcentaje de fallos mide el funcionamiento del servicio, no la calidad predictiva. Para evaluar el modelo se necesitan resultados reales de los clientes.
- Un error 422 por una variable ausente se corrige en el cliente que envía los datos; no implica que el modelo esté desactualizado.
- El registro puede contener datos identificables si el cliente envía `customerID`; en un entorno real debe protegerse y aplicarse una política de retención.
- El despliegue en la nube, los tableros gráficos y el reentrenamiento automático están fuera del alcance de esta entrega.
