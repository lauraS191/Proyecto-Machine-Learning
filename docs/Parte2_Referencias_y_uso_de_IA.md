# Parte 2: referencias de modelos y datos, y uso de IA generativa

Proyecto del curso Aprendizaje de Máquina (ISIS-2611), Universidad de los Andes, 2026-20.
Notebook: `notebooks/P2_XLMR_Aumentacion.ipynb`.

## 1. Modelos preentrenados utilizados

| Uso en el proyecto | Modelo | Fuente | Licencia |
|---|---|---|---|
| Modelo final y de las iteraciones 2 y 3 (*fine-tuning* completo) | XLM-RoBERTa large (`FacebookAI/xlm-roberta-large`) | https://huggingface.co/FacebookAI/xlm-roberta-large | MIT |
| Línea base, iteración 1 (embeddings congelados + MLP) | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Apache 2.0 |
| Aumentación de datos (traducción automática) | NLLB-200 distilled 600M (`facebook/nllb-200-distilled-600M`) | https://huggingface.co/facebook/nllb-200-distilled-600M | CC-BY-NC 4.0 (uso no comercial; el proyecto es académico) |
| Detección de idioma | langid.py | https://github.com/saffsd/langid.py | BSD |

Artículos de referencia:

- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L. & Stoyanov, V. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL 2020.
- Reimers, N. & Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP 2020.
- NLLB Team, Costa-jussà, M. R. et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. arXiv:2207.04672.
- Lui, M. & Baldwin, T. (2012). *langid.py: An Off-the-shelf Language Identification Tool*. ACL 2012 System Demonstrations.
- Sennrich, R., Haddow, B. & Birch, A. (2016). *Improving Neural Machine Translation Models with Monolingual Data*. ACL 2016. Es el origen de la aumentación por traducción.

### Arquitecturas utilizadas

- **Clasificador (modelo entregado):** XLM-RoBERTa large, un transformer *encoder-only* de la familia de BERT y RoBERTa, con una cabeza de clasificación de 3 salidas. Es el único modelo que interviene al predecir, como indica el comunicado del curso sobre el uso de arquitecturas *encoder-only*.
- **Línea base:** `paraphrase-multilingual-mpnet-base-v2`, también *encoder-only*, con un MLP encima.
- **NLLB-200 (solo para la aumentación):** es un modelo de traducción *encoder-decoder*, no *decoder-only*. Lo usamos únicamente para generar traducciones de reseñas de `train.csv` como datos sintéticos de entrenamiento, algo que la sección 4 del enunciado permite de forma explícita ("traducciones o paráfrasis generadas"). No clasifica, no etiqueta ninguna reseña (las sintéticas heredan la etiqueta de su original), no toca `eval.csv` y no forma parte del modelo entregado.
- En ninguna parte del proyecto usamos modelos *decoder-only* ni modelos generativos para clasificar o etiquetar.

## 2. Datos

- **Datos de la competencia:** `train.csv`, `eval.csv` y `sample_submission.csv` de la competencia de Kaggle *Parte 2 - Competencia Aprendizaje de Máquina 2026-20*. `eval.csv` solo se usó para generar predicciones, nunca para entrenar ni para elegir modelos.
- **Datos externos:** no se usaron.
- **Datos sintéticos (aumentación):** se generaron a partir de `train.csv` con el modelo NLLB-200 de la tabla anterior. Se tomaron al azar, con semilla 42, 6.000 reseñas en español, y cada una recibió una transformación:
  - 1.500 traducidas al inglés, 1.500 al portugués y 1.500 al francés. Se traducen oración por oración, con decodificación voraz.
  - 1.500 con *code-switching* español-inglés: cerca de la mitad de las oraciones de cada reseña, elegidas al azar, se traducen al inglés.
  - Cada reseña sintética hereda la etiqueta de su original.
  - Se descartaron las traducciones defectuosas (más cortas que el 50 % del original o idénticas a él).
  - En la comparación con y sin aumentación se excluyeron las reseñas sintéticas cuyo original estaba en la partición de validación.
  - El detalle completo y el código están en la sección 8 del notebook, y el resultado se guarda en `aumentacion_traduccion.csv`.

## 3. Uso de IA generativa

> **Nota para el grupo:** completar y ajustar esta sección para que refleje exactamente cómo se usó la IA. Agregar el uso de cada integrante.

- **Herramienta:** Claude (Anthropic), mediante Claude Code en VS Code.
- **Para qué se usó:**
  - Analizar el enunciado de la Parte 2 y las prácticas del curso (P14 perceptrón multicapa, P15 embeddings, P16 RNN/CNN) para proponer un enfoque coherente con lo visto en clase.
  - Explorar los datos, en particular la distribución de idiomas de `train` frente a `eval`.
  - Escribir el borrador del código del notebook (carga de datos, detector de idioma, línea base, bucle de *fine-tuning* en PyTorch, aumentación por traducción y análisis de errores) y de los textos explicativos.
  - Redactar este documento.
- **Qué hizo el grupo:** revisar y ejecutar el notebook en GPU, verificar los resultados, completar las conclusiones con los números obtenidos, subir los envíos a Kaggle y elegir el modelo final según el puntaje público.
- **Qué no se hizo con IA:** ningún modelo generativo etiquetó ni clasificó los textos de `eval.csv`. Todas las predicciones enviadas salen del modelo entregado, entrenado solo con `train.csv` y los datos sintéticos descritos arriba.
