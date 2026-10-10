# Proyecto-Machine-Learning

Clasificación de sentimientos en reseñas de productos en español, Parte 1 del curso 2026-20.

## Experimento de Naive Bayes — Alejandro Abril

El notebook [04_Naive_Bayes.ipynb](notebooks/04_Naive_Bayes.ipynb) compara BoW y TF-IDF con MultinomialNB y ComplementNB. Las tablas, la matriz de confusión, los errores y la conclusión se ven dentro del notebook.

### Ejecución

1. Descargar los datos de la competencia y ubicar `train.csv` en `data/`. Para generar el envío, agregar `eval.csv` y, si está disponible, `sample_submission.csv`.
2. Preparar un entorno Python desde la raíz del proyecto:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements-naive-bayes.txt
   ```

3. Abrir el notebook en Jupyter o VS Code, seleccionar ese entorno y ejecutar todas las celdas desde la raíz del proyecto o desde `notebooks/`.

La búsqueda compara distintas preparaciones del texto y ajusta con más detalle las dos combinaciones que mejor salen. Usa validación cruzada con cinco particiones y dos procesos; si el computador tiene poca memoria, cambiar `N_JOBS` a `1`.

Se prueban limpieza, stemming en español y stop words, tomando como base P3. Se conservan palabras como “no” y “pero”, que pueden cambiar el sentido de una opinión. La preparación del texto queda incluida en el modelo guardado.

### Selección y resultados

- Se usa una división estratificada 80/20 con semilla 42. El modelo se elige por accuracy de validación cruzada, con F1 macro y menor desviación como desempates.
- La última ejecución eligió **TF-IDF + MultinomialNB con stemming**: 71,50% de accuracy en validación cruzada y 72,25% en el 20% de validación. Ese 20% ya se había consultado en la primera versión.
- Las 305 configuraciones evaluadas se pueden consultar en una tabla desplegable dentro del notebook. Las nuevas ejecuciones muestran los resultados ahí mismo.
- `models/naive_bayes.joblib`: modelo entrenado con todo `train.csv`, incluida la preparación del texto.
- `submissions/naive_bayes.csv`: archivo `id,answer` para subir manualmente a Kaggle.

Cada ejecución reemplaza esos dos archivos. El análisis y los resultados se conservan dentro del notebook.

### Entrega de la Parte 1

La sección 4.2 del PDF pide el notebook y el modelo entrenado guardado con pickle o joblib. Para Naive Bayes, estos archivos son `notebooks/04_Naive_Bayes.ipynb` y `models/naive_bayes.joblib`. El CSV de `submissions/` se utiliza en Kaggle.

El grupo debe entregar únicamente el notebook y el modelo que correspondan a su mejor envío público de Kaggle. Este experimento es el candidato de Naive Bayes; su elección como modelo final depende de la comparación con los demás envíos.

Después de subir el CSV, hay que anotar el score de Kaggle y compararlo con los envíos del resto del grupo.

La sección 1 del notebook relaciona cada paso con las secciones y ejercicios de `MaterialDeClase-ISIS-2611/202620`: preparación y representación de texto de P3, entrenamiento y evaluación de P7, y comparación de modelos del Laboratorio 2. También explica las adaptaciones a las reseñas en español.

La elección de MultinomialNB y ComplementNB viene del acuerdo del grupo. En la carpeta 202620 no se encontró una práctica específica de Naive Bayes; sus parámetros se explican como ajustes de los clasificadores y se referencian a la documentación de scikit-learn. Las referencias y la ayuda de IA utilizada están al final del notebook.
