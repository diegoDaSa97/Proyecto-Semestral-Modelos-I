# Fase 3: API REST – Telco Customer Churn

**Asignatura:** Modelos y Simulación de Sistemas I  
**Proyecto:** Predicción de fuga de clientes (*Customer Churn*) en telecomunicaciones.

## 1. Objetivo

Exponer mediante una API REST el modelo de clasificación desarrollado en la Fase 1 y convertido en una aplicación reutilizable en la Fase 2. Esta fase no crea un algoritmo nuevo: reutiliza `train.py`, `predict.py` y el pipeline almacenado.

La API se construye con **FastAPI** y permite consultar el estado del servicio, generar predicciones para clientes nuevos y entrenar una nueva versión del modelo.

## 2. Estructura de la carpeta

```text
fase-3/
├── app/
│   ├── __init__.py
│   └── main.py                 # Endpoints /health, /predict y /train
├── client/
│   └── cliente_prueba.py       # Cliente de prueba por terminal
├── examples/
│   └── solicitud_predict.json  # Ejemplo de entrada para /predict
├── tests/
│   ├── conftest.py             # Dataset sintético y fixtures
│   └── test_api.py             # Pruebas de integración
├── Dockerfile                 # Imagen de la API
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 3. Relación con las fases anteriores

- La Fase 1 aporta el pipeline inicial `fase-1/models/churn_model_pipeline.joblib`.
- La Fase 2 aporta `train.py` y `predict.py`, que se importan directamente para evitar duplicar la lógica de preparación de datos, entrenamiento y predicción.
- La API busca primero el modelo versionado más reciente en `fase-2/models/`. Si esa carpeta todavía no contiene un modelo, utiliza el artefacto inicial de Fase 1.
- Cuando se ejecuta `/train`, el nuevo modelo y sus metadatos se guardan en `fase-2/models/` y la API activa esa versión para las siguientes predicciones.

## 4. Instalación y ejecución local

Se recomienda Python 3.13, la misma versión mayor utilizada en la Fase 2.

Desde la carpeta `fase-3`:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

La API queda disponible en `http://127.0.0.1:8000`.

Documentación interactiva: `http://127.0.0.1:8000/docs`  
Documentación OpenAPI: `http://127.0.0.1:8000/openapi.json`

## 5. Endpoints

### 5.1. Estado del servicio – `GET /health`

Comprueba que el servicio responde e informa si pudo cargar un modelo.

Ejemplo:

```bash
curl http://127.0.0.1:8000/health
```

Respuesta cuando el modelo está disponible:

```json
{
  "status": "ok",
  "model_available": true,
  "model_version": "fase-1-inicial"
}
```

Si la API está activa pero no encuentra un artefacto de modelo, responde HTTP 200 con `status: degraded` y `model_available: false`. En ese caso se puede entrenar mediante `/train`.

### 5.2. Predicción – `POST /predict`

Recibe una lista de clientes. Cada cliente debe contener las 19 variables predictoras usadas durante el entrenamiento. `customerID` es opcional y se conserva en la respuesta. La variable objetivo `Churn` no se debe enviar.

Ejemplo de solicitud:

```json
{
  "threshold": 0.5,
  "customers": [
    {
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
      "TotalCharges": 179.8
    }
  ]
}
```

Ejemplo con cURL (desde la raíz del repositorio):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @fase-3/examples/solicitud_predict.json
```

Respuesta de ejemplo (los valores de probabilidad dependen del modelo activo):

```json
{
  "model_version": "fase-1-inicial",
  "count": 1,
  "predictions": [
    {
      "customerID": "DEMO-001",
      "churn_prediction": 1,
      "churn_probability": 0.73
    }
  ]
}
```

`churn_prediction` vale `1` cuando la probabilidad alcanza o supera el umbral y `0` cuando no lo alcanza. El umbral predeterminado es `0.5` y se puede modificar entre `0` y `1`.

### 5.3. Entrenamiento – `POST /train`

Ejecuta el entrenamiento con el dataset que está en el servidor, guarda un modelo versionado y lo activa para las siguientes predicciones. No recibe rutas de archivos desde el cliente.

Solicitud:

```json
{
  "model": "random_forest",
  "test_size": 0.2,
  "seed": 42
}
```

También se permite `logistic_regression` como alternativa. `version` es opcional y, si se envía, debe respetar el formato `AAAAMMDD_HHMMSS`; si no se envía, `train.py` genera la versión con fecha y hora. Este formato mantiene el orden cronológico utilizado por el selector de modelos de la Fase 2.

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model":"random_forest","test_size":0.2,"seed":42}'
```

La respuesta incluye la versión, el algoritmo, las métricas calculadas por `train.py` y el nombre del archivo `.joblib` creado.

## 6. Validación y manejo de errores

| Situación | Código HTTP | Comportamiento |
|---|---:|---|
| Solicitud correcta | 200 | Devuelve estado, predicciones o resultado del entrenamiento. |
| JSON o parámetros fuera de rango | 422 | FastAPI/Pydantic indica los campos inválidos. |
| Faltan variables predictoras | 422 | Informa los nombres de las columnas faltantes. |
| No existe un modelo para predecir | 503 | Indica que se debe entrenar o revisar la ruta del modelo. |
| No existe el dataset de entrenamiento | 503 | Informa que el recurso de entrenamiento no está disponible. |
| Se intenta repetir una versión | 409 | Evita sobrescribir un modelo existente. |
| Parámetros de entrenamiento no aceptados | 400 | Devuelve el mensaje de validación del proceso de entrenamiento. |

La solicitud de predicción acepta entre 1 y 500 clientes por petición y un umbral entre 0 y 1. Los campos adicionales no utilizados por el pipeline no participan en la predicción; las variables requeridas deben estar presentes.

## 7. Cliente de prueba

El cliente usa la biblioteca estándar de Python, por lo que no necesita instalar dependencias adicionales. Con la API en ejecución, desde la raíz del repositorio:

```bash
# Consulta /health y prueba /predict con el JSON de ejemplo
python fase-3/client/cliente_prueba.py

# Ejecuta /train y luego prueba /predict
python fase-3/client/cliente_prueba.py --entrenar
```

Se puede indicar otra URL con `--url`, por ejemplo `--url http://localhost:8000`.

## 8. Pruebas automatizadas

Desde la carpeta `fase-3`:

```bash
pip install -r requirements-dev.txt
pytest -v
```

Las pruebas utilizan un dataset sintético con la misma estructura del conjunto Telco Customer Churn, por lo que no dependen de datos externos. Comprueban:

- Estado del servicio cuando no hay un modelo cargado.
- Entrenamiento y activación de una versión.
- Predicciones y rango de probabilidades.
- Rechazo de variables obligatorias ausentes.
- Validación del umbral.
- Respuesta cuando no hay un modelo disponible.
- Protección contra sobrescritura de versiones.
- Validación de parámetros de entrenamiento.

## 9. Docker

**Importante:** el contexto de construcción es la raíz del repositorio porque la imagen reutiliza archivos de las fases 1 y 2.

Desde la raíz del proyecto:

```bash
docker build -f fase-3/Dockerfile -t churn-api .

docker run --rm -p 8000:8000 \
  -v "${PWD}/fase-2/models:/app/models" \
  churn-api
```

En PowerShell, el comando puede escribirse en una sola línea:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/fase-2/models:/app/models" churn-api
```

La imagen contiene el modelo inicial de Fase 1 para que la API pueda iniciar sin ejecutar primero `/train`. El volumen conserva fuera del contenedor las versiones nuevas generadas por `/train`.

## 10. Consideraciones de uso

- `/train` está pensado para demostración académica y entornos controlados. En un despliegue público debe protegerse con autenticación y limitarse su frecuencia, porque consume recursos de CPU.
- El artefacto `joblib` debe proceder de este repositorio o de una fuente confiable; no se deben cargar archivos serializados de terceros.
- La API no modifica el dataset ni cambia el algoritmo seleccionado en la Fase 1. Solo publica las capacidades ya construidas.
