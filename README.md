# Proyecto-Machine-Learning

Parte 1 de la competencia de Aprendizaje de Máquina 2026-20: clasificación de sentimientos (positivo, negativo o neutral) en reseñas de productos en español, usando solo técnicas clásicas de scikit-learn.

## Estructura

- `data/`: archivos de la competencia (`train.csv`, `eval.csv`, `sample_submission.csv`).
- `notebooks/`: un notebook por modelo, más la exploración de datos.
- `transformaciones.py`: funciones de transformación del texto que usan los pipelines.
- `models/`: pipelines entrenados, guardados con joblib.
- `submissions/`: archivos CSV enviados a Kaggle.

## Resultados en Kaggle (puntaje público)

| Modelo | Archivo | Puntaje |
|---|---|---|
| SVM lineal, conectores y posición | `submission_svm_lineal_conectores.csv` | 0.87555 |
| Regresión Logística V2 | `submission_regresion_logistica_v2.csv` | 0.86555 |
| SVM lineal, representación V2 | `submission_svm_lineal_v2.csv` | 0.85666 |
| Regresión Logística V1 | `submission_regresion_logistica.csv` | 0.83222 |

## Entorno

Todos los modelos se entrenaron con scikit-learn 1.9.0, que necesita Python 3.10 o superior (usamos Python 3.12). Las versiones de las librerías están en `requirements.txt`:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para cargar los modelos sin advertencias hay que usar esa misma versión de scikit-learn.

## Cargar un modelo

Los pipelines usan funciones de `transformaciones.py`, así que ese archivo debe estar en la misma carpeta (o en el `sys.path`) al cargar el modelo:

```python
import joblib
import pandas as pd

modelo = joblib.load("models/svm_lineal_conectores.joblib")
eval_data = pd.read_csv("data/eval.csv")
predicciones = modelo.predict(eval_data["text"])
```
