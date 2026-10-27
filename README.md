# Fase 1: Modelo Predictivo - Telco Customer Churn
**Asignatura:** Modelos y Simulación de Sistemas I

## 1. Descripción del Problema
Este proyecto aborda la predicción de fuga de clientes (*Customer Churn*) en el sector de telecomunicaciones. El objetivo es identificar oportunamente a aquellos usuarios con alta probabilidad de cancelar sus servicios para permitir la ejecución de estrategias proactivas de retención comercial.

* **Tipo de tarea:** Clasificación binaria supervisada.
* **Variable objetivo:** `Churn` (1: cliente cancela el servicio, 0: cliente permanece).

## 2. Descripción del Conjunto de Datos
* **Fuente:** Telco Customer Churn (Kaggle).
* **Muestras:** 7.043 observaciones.
* **Variables predictoras:** 19 variables (atributos sociodemográficos, servicios contratados, modalidad contractual, antigüedad `tenure` y método de pago).
* **Tratamiento de valores faltantes:** La variable `TotalCharges` presentaba 11 valores vacíos (0.156% del dataset), los cuales son imputados utilizando la mediana estimada exclusivamente en el conjunto de entrenamiento.

## 3. Prevención de Fuga de Información (Data Leakage)
Para garantizar un flujo metodológico riguroso y reproducible:
1. La partición de datos en entrenamiento (80%) y prueba (20%) estratificada se realizó antes de efectuar cualquier transformación o cálculo estadístico.
2. Todas las etapas de transformación numérica (imputación por mediana y escalado estándar) y categórica (imputación por moda y codificación One-Hot) se encapsularon dentro de un `Pipeline` y un `ColumnTransformer` de `scikit-learn`. Los parámetros estadísticos se ajustaron únicamente sobre `X_train` (`fit_transform`) y se aplicaron sin recalcular sobre `X_test` (`transform`).

## 4. Resultados del Modelo
Se entrenaron y compararon dos algoritmos sobre los datos de prueba:
* **Random Forest:** ROC-AUC: 0.8426 | Accuracy: 80.48%
* **Regresión Logística:** ROC-AUC: 0.8419 | Accuracy: 80.55%

Se seleccionó **Random Forest** como el modelo final por alcanzar el mejor desempeño global en la métrica de discriminación **ROC-AUC (0.8426)**.

## 5. Artefacto Almacenado
El pipeline final completo está almacenado en:
`models/churn_model_pipeline.joblib`

Este artefacto recibe directamente DataFrames con las variables crudas originales y produce tanto las etiquetas predichas como las probabilidades de abandono.

# Fase 2: Modelo Predictivo - Telco Customer Churn
### 1. Implementación de versionamiento con Git Flow
Se implementa para mejorar el desempeño en el ambito de desarrollo del modelo. 