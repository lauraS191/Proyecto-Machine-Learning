# Proyecto-Machine-Learning

Parte 1 de la competencia de Aprendizaje de Máquina 2026-20: clasificación de sentimientos (positivo, negativo o neutral) en reseñas de productos en español, usando solo técnicas clásicas de scikit-learn.

## Estructura

- `data/`: archivos de la competencia (`train.csv`, `eval.csv`, `sample_submission.csv`).
- `notebooks/`: un notebook por modelo, más la exploración de datos.
- `transformaciones.py`: funciones de transformación del texto que usan los pipelines.
- `models/`: pipelines entrenados, guardados con joblib.
- `submissions/`: archivos CSV enviados a Kaggle.

## Resultados en Kaggle (puntaje público)

| Modelo | Notebook | Archivo | Puntaje |
|---|---|---|---|
| SVM lineal, conectores por cláusulas | `03_SVM.ipynb` | `submission_svm_lineal_clausulas.csv` | **0.88333** |
| SVM lineal, conectores y posición | `03_SVM.ipynb` | `submission_svm_lineal_conectores.csv` | 0.87555 |
| Regresión Logística V2 | `02_Regresion_Logistica.ipynb` | `submission_regresion_logistica_v2.csv` | 0.86555 |
| SVM lineal, representación V2 | `03_SVM.ipynb` | `submission_svm_lineal_v2.csv` | 0.85666 |
| Regresión Logística V1 | `02_Regresion_Logistica.ipynb` | `submission_regresion_logistica.csv` | 0.83222 |
| Naive Bayes, TF-IDF + MultinomialNB por cláusulas | `04_Naive_Bayes.ipynb` | `naive_bayes.csv` | Pendiente |
| Línea base de la competencia | | `submission_baseline_nb.csv` | 0.65555 |

Para la entrega en Bloque Neón se usa solo el notebook y el modelo del envío con mejor puntaje público. Por ahora es el SVM con cláusulas: `notebooks/03_SVM.ipynb` y `models/svm_lineal_clausulas.joblib`, que necesita `transformaciones.py` para cargarse.

## Entorno

Todo el equipo usa el mismo entorno, para que los modelos guardados se puedan cargar en cualquier computador: Python 3.12 y scikit-learn 1.9.0 (esta versión necesita Python 3.10 o superior). Las versiones de las librerías están en `requirements.txt`.

En Windows, desde la raíz del proyecto:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS o Linux:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Después, en Jupyter o VS Code, se selecciona el kernel del `.venv` antes de ejecutar los notebooks. Si un modelo se guarda con otras versiones de scikit-learn o joblib, puede mostrar advertencias o no cargar en otro computador, así que cada notebook debe volver a ejecutarse con este entorno antes de guardar su modelo.

## Cargar un modelo

Los pipelines usan funciones de `transformaciones.py`, así que ese archivo debe estar en la misma carpeta (o en el `sys.path`) al cargar el modelo:

```python
import joblib
import pandas as pd

modelo = joblib.load("models/svm_lineal_conectores.joblib")
eval_data = pd.read_csv("data/eval.csv")
predicciones = modelo.predict(eval_data["text"])
```

## Experimento de Naive Bayes (Alejandro Abril)

El notebook [04_Naive_Bayes.ipynb](notebooks/04_Naive_Bayes.ipynb) compara BoW y TF-IDF con MultinomialNB y ComplementNB. Las tablas, la matriz de confusión, los errores y la conclusión se ven dentro del notebook.

### Ejecución

1. Ubicar `train.csv` en `data/`. Para generar el envío, agregar `eval.csv` y `sample_submission.csv`.
2. Preparar el entorno común descrito en la sección Entorno.
3. Abrir el notebook, seleccionar ese entorno y ejecutar todas las celdas desde la raíz del proyecto o desde `notebooks/`.

La búsqueda compara el texto completo, la última oración, los conectores por oraciones y los conectores por cláusulas. Ajusta con más detalle los dos candidatos de mayor accuracy de CV. Usa cinco particiones y dos procesos; si el computador tiene poca memoria, cambiar `N_JOBS` a `1`.

La referencia conserva el stemming en español de P3. Las nuevas representaciones reutilizan `extraer_oraciones_lote`, `marcar_conectores_lote` y `marcar_clausulas_lote` del equipo, combinadas con el texto completo y n-gramas de palabras y caracteres. Las funciones anteriores de `transformaciones.py` siguen intactas; solo se agregó `preparar_textos_nb` para reproducir la referencia.

### Selección y resultados

- Se usa una división estratificada 80/20 con semilla 42 y cinco particiones de CV con la misma semilla. GridSearchCV usa accuracy; entre los finalistas de las cuatro familias se compara también F1 macro y desviación.
- La última ejecución eligió **TF-IDF + MultinomialNB por cláusulas**, con `alpha=0.1`: **87,61% de accuracy en CV y 87,71% en validación**, frente al 72,25% anterior. F1 macro de validación: 0,8814. Ese 20% ya se había consultado.
- Las 157 configuraciones evaluadas, la matriz de confusión y los errores se pueden consultar dentro del notebook. Quedaron 295 errores de las 2.400 reseñas de validación.
- El notebook se volvió a ejecutar completo en Python 3.12 y con las versiones de `requirements.txt`. El modelo recargado produce las mismas 3.000 predicciones que el CSV.
- `models/naive_bayes.joblib`: modelo entrenado con todo `train.csv`, incluida la preparación del texto.
- `submissions/naive_bayes.csv`: archivo `id,answer` para subir manualmente a Kaggle.

Cada ejecución reemplaza esos dos archivos. El análisis y los resultados se conservan dentro del notebook. Para cargar el modelo hay que tener disponible `transformaciones.py`.

### Entrega de la Parte 1

La sección 4.2 del PDF pide el notebook y el modelo entrenado guardado con pickle o joblib. Para Naive Bayes, estos archivos son `notebooks/04_Naive_Bayes.ipynb` y `models/naive_bayes.joblib`. El CSV de `submissions/` se utiliza en Kaggle.

El grupo debe entregar únicamente el notebook y el modelo que correspondan a su mejor envío público de Kaggle. Este experimento es el candidato de Naive Bayes; su elección como modelo final depende de la comparación con los demás envíos.

Después de subir el CSV, hay que anotar el score de Kaggle y compararlo con los envíos del resto del grupo.

La sección 1 del notebook relaciona cada paso con las secciones y ejercicios de `MaterialDeClase-ISIS-2611/202620`: preparación y representación de texto de P3, entrenamiento y evaluación de P7, y comparación de modelos del Laboratorio 2. También explica las adaptaciones a las reseñas en español.

La elección de MultinomialNB y ComplementNB viene del acuerdo del grupo. En la carpeta 202620 no se encontró una práctica específica de Naive Bayes; sus parámetros se explican como ajustes de los clasificadores y se referencian a la documentación de scikit-learn. Las referencias están al final del notebook.
