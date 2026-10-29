# Fase 2: Scripts y Docker – Telco Customer Churn
**Asignatura:** Modelos y Simulación de Sistemas I

En esta fase **no se desarrolla un modelo nuevo**: se convierte el modelo de la Fase 1 en una aplicación
reutilizable que se entrena y predice desde la línea de comandos, empaquetada en Docker y con pruebas automáticas.

## 1. Contenido de la carpeta

```
fase-2/
├── train.py                 # Entrena, evalúa y guarda el modelo versionado
├── predict.py               # Genera predicciones desde un CSV
├── Dockerfile               # Imagen reproducible (Python 3.13)
├── requirements.txt         # Dependencias con versiones fijas
├── requirements-dev.txt     # Dependencias + pytest
├── pytest.ini
├── .dockerignore
├── data/
│   ├── Telco-Customer-Churn.csv         # Dataset de entrenamiento
│   └── ejemplo_nuevos_clientes.csv      # Ejemplo para probar predict.py (incluye un TotalCharges vacío)
├── models/                  # Modelos versionados (.joblib + .json)
└── tests/test_churn.py      # Pruebas con pytest
```

## 2. Requisitos y versiones
Python 3.13. Las versiones están fijadas porque un modelo serializado con `joblib` solo es confiable
con la versión de scikit-learn con que se entrenó:

| Paquete | Versión |
|---------|---------|
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| joblib | 1.6.0 |

Instalación local:
```bash
cd fase-2
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install -r requirements-dev.txt
```

## 3. Entrenamiento: `train.py`
```bash
python train.py                                  # Random Forest (modelo elegido en la Fase 1)
python train.py --model logistic_regression      # alternativa
python train.py --seed 7 --test-size 0.25
```
Opciones: `--data`, `--models-dir`, `--model`, `--test-size`, `--seed`, `--version`.

El script reproduce la Fase 1: partición 80/20 estratificada con `random_state=42` **antes** de cualquier
transformación; imputación, escalado y One-Hot dentro de un `Pipeline` ajustado solo con entrenamiento;
Random Forest con `n_estimators=100`, `max_depth=8`.
Resultado de verificación (idéntico a la Fase 1): **ROC-AUC 0.8426, accuracy 80.48 %**.

### Versionado del modelo
Cada ejecución crea dos archivos con la fecha y hora de entrenamiento como versión (`AAAAMMDD_HHMMSS`):
```
models/churn_model_20261027_210500.joblib   # pipeline completo (acepta datos crudos)
models/churn_model_20261027_210500.json     # metadatos
```
Los metadatos incluyen: tipo de modelo, hiperparámetros, métricas, semilla, tamaños de partición,
columnas de entrada, hash SHA-256 del dataset y versiones de Python y librerías.
Nunca se sobrescribe una versión existente, y `predict.py` usa por defecto la más reciente.

## 4. Predicción: `predict.py`
```bash
python predict.py --input data/ejemplo_nuevos_clientes.csv
python predict.py --input data/ejemplo_nuevos_clientes.csv --output output/predicciones.csv
python predict.py --input clientes.csv --model-path models/churn_model_20261027_210500.joblib --threshold 0.4
```
El CSV debe traer las 19 columnas predictoras (`customerID` y `Churn` son opcionales). La salida contiene
`customerID` (si existe), `churn_prediction` (0/1) y `churn_probability`. Los `TotalCharges` vacíos se
imputan igual que en el entrenamiento y las categorías no vistas se ignoran sin romper la predicción.
Si faltan columnas, el archivo no existe o no hay modelos, el script termina con un mensaje claro (código de salida 1).

## 5. Pruebas
```bash
pytest
```
Se usan datos sintéticos con la misma estructura del dataset (corren en pocos segundos). Verifican: limpieza
de `TotalCharges`, generación del modelo y sus metadatos, reproducibilidad con la misma semilla, que no se
sobrescriban versiones, que el imputador se ajuste solo con entrenamiento, forma y rango de las predicciones,
manejo de nulos y categorías desconocidas, error claro ante columnas faltantes, selección del modelo más
reciente y ejecución completa de la CLI.

## 6. Docker
Los modelos no se incluyen en la imagen: se generan con `train.py` y se montan como volumen para
conservar el versionado fuera del contenedor.
```bash
docker build -t churn-fase2 .

# Entrenar (el modelo queda en ./models del equipo anfitrión)
docker run --rm -v "${PWD}/models:/app/models" churn-fase2 train.py

# Predecir
docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/output:/app/output" \
  churn-fase2 predict.py --input data/ejemplo_nuevos_clientes.csv --output output/predicciones.csv
```
En PowerShell escriba cada comando en una sola línea (sin `\`).
Para predecir sobre un CSV propio, móntelo, por ejemplo: `-v "${PWD}/mis_datos:/app/mis_datos"`.

## 7. Decisiones de diseño y limitaciones
- `train.py` entrena por defecto solo el Random Forest ya elegido en la Fase 1, en lugar de repetir la
  comparación de modelos: en el notebook el mejor modelo se escogió con el ROC-AUC del conjunto de prueba,
  lo que filtra información de prueba hacia la decisión.
- `predict.py` importa la limpieza de `train.py` para que entrenamiento y predicción traten los datos igual.
- La imagen no usa usuario no root para evitar problemas de permisos con volúmenes montados; sería una mejora para producción.
- Un `.joblib` solo debe cargarse si proviene de una fuente confiable.
