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

El notebook [04_Naive_Bayes.ipynb](notebooks/04_Naive_Bayes.ipynb) compara BoW y TF-IDF con MultinomialNB y ComplementNB. Los resultados, las gráficas, los errores y la conclusión quedan en el notebook.

### Cómo ejecutarlo

1. Ubicar `train.csv` en `data/`. Para generar el envío, agregar `eval.csv` y `sample_submission.csv`.
2. Preparar el entorno común de Python 3.12 descrito arriba.
3. Abrir el notebook, seleccionar ese entorno y ejecutar todas las celdas desde la raíz del proyecto o desde `notebooks/`.

La ejecución completa toma tiempo: además de la búsqueda global, repite las dos etapas dentro de cada una de las cinco particiones externas. Usa dos procesos; si falta memoria, cambiar `N_JOBS` a `1`.

### Qué se probó

La primera búsqueda compara las cuatro combinaciones de representación y clasificador: `alpha=[0.01, 0.03, 0.1, 0.3, 1]`, unigramas o unigramas con bigramas, y `min_df=[1, 2, 3]`. Después se refinan las dos mejores combinaciones con conteos binarios o TF logarítmico, una o dos palabras de conector, una o dos oraciones finales y cuatro opciones de pesos.

Los bloques reúnen el texto completo, las marcas de cláusulas y las últimas oraciones como palabras y caracteres. Se reutilizan las transformaciones del equipo. Las adaptaciones a las reseñas se explican por separado de lo visto en clase.

La base es [MaterialDeClase-ISIS-2611/202620](https://github.com/alejandroabrilm/MaterialDeClase-ISIS-2611/tree/main/202620): calidad de datos de P1/P2, representación de texto de P3, pipelines y evaluación anidada de P7, y comparación y bootstrap del Laboratorio 2. En 202620 no hay una práctica específica de Naive Bayes; el notebook explica sus supuestos y parámetros con la documentación de scikit-learn.

### Resultados de esta ejecución

- Se conserva la división estratificada 80/20 y la semilla 42. La selección usa accuracy de CV; los desempates usan F1 macro y menor desviación.
- La referencia anterior se reprodujo: **87,61% en CV** y **87,71% en validación**.
- Ganador: **TF-IDF + MultinomialNB**, con `alpha=0.1`, n-gramas `[1, 2]` y `min_df=1`. Obtuvo **87,81% en CV** y **87,71% en el 20% reutilizado**; F1 macro: **0.8814**. Quedaron **295 errores de 2.400 reseñas**.
- La evaluación anidada obtuvo **87.45% ± 1.09 puntos** de accuracy externa, frente a **87.61%** de la referencia. Comparo estos valores con la búsqueda global para revisar cuánto optimismo introduce la selección de parámetros. El signo ± corresponde a la desviación entre folds.
- Las **184 configuraciones**, la eliminación de cada bloque, las métricas por clase y ejemplos de errores están dentro del notebook. La CV anidada repite las 184 configuraciones en cada partición externa.
- El modelo recargado reproduce las 3.000 predicciones del CSV; una segunda ejecución secuencial de la CV del ganador comprueba las métricas de la búsqueda.

`eval.csv` se usa únicamente al generar las predicciones finales. El 20% ya se había consultado en versiones anteriores, por eso se reporta como evaluación complementaria. La evaluación anidada mide la búsqueda definida en esta versión; el diseño de características también recibió información de experimentos anteriores.

La meta de **90% público en Kaggle sigue pendiente**. El CSV de esta versión está listo para subir; no se le atribuye el puntaje de otros modelos. La sección de la rúbrica registra la línea base de 0,65555 y la evidencia del grupo consultada en Kaggle, con fecha.

### Archivos para entregar

- `models/naive_bayes.joblib`: pipeline entrenado con las 12.000 reseñas.
- `submissions/naive_bayes.csv`: archivo de 3.000 filas con columnas `id,answer`.
- `notebooks/04_Naive_Bayes.ipynb`: código, resultados y explicación.

Cada ejecución reemplaza el modelo y el CSV. Para cargar el modelo debe estar disponible `transformaciones.py`; la función de preparación de Naive Bayes se conserva para el ejemplo de stemming de P3.

La sección 4.2 del PDF pide el notebook y el modelo guardado. El CSV se usa en Kaggle. El equipo escogerá su entrega final según el mejor envío público; este trabajo aporta el candidato de Naive Bayes.
