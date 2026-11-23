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

## 3. Uso de herramientas de IA generativa

**Herramienta utilizada:** Claude (Anthropic), asistente de IA conversacional, usado a lo largo del desarrollo de la Parte 2 desde Claude Code dentro de Visual Studio Code.

**Tipo de uso:** apoyo en la generación inicial y depuración de código (detector de idioma con langid y regla de code-switching, entrenamiento de XLM-RoBERTa en PyTorch, aumentación con NLLB-200 y análisis de errores por idioma), explicación conceptual (arquitecturas encoder-only frente a encoder-decoder, puntaje público frente a privado en Kaggle, validación del modelo final, prueba de McNemar) y apoyo en la redacción del notebook y de este documento.

### Prompts principales utilizados

- Apoyo para plantear el enfoque a partir del enunciado y de las prácticas P14, P15 y P16 del curso, buscando un modelo multilingüe competitivo y coherente con lo visto en clase.
- Apoyo para construir el detector de idioma (langid más una regla para las mezclas español-inglés) y ajustar sus umbrales revisando a mano los falsos positivos.
- Apoyo para escribir la línea base con embeddings congelados y mean pooling, como en P15, y el MLP con dropout y early stopping, como en P14.
- Apoyo para escribir el bucle de fine-tuning de XLM-RoBERTa large en PyTorch (AdamW, calentamiento lineal, precisión mixta, label smoothing y selección de la mejor época), de forma que cupiera en una GPU T4 de Kaggle.
- Apoyo para implementar la aumentación con NLLB-200, traduciendo oración por oración y armando el code-switching, sin que ninguna reseña sintética viniera de la partición de validación.
- Pregunta sobre cómo comparar con y sin aumentación de forma justa, lo que llevó a usar la misma partición, la misma semilla y la prueba de McNemar.
- Apoyo para organizar el notebook por iteraciones, de modo que cada una generara un envío distinto para Kaggle.
- Consulta sobre por qué el modelo final no tiene accuracy de validación y cómo justificarlo.
- Consulta sobre el comunicado de arquitecturas encoder-only y si el uso de NLLB en la aumentación lo cumplía.
- Apoyo para redactar los textos del notebook a partir de los resultados ya obtenidos y para redactar este documento.

### Trabajo del grupo

Definimos el objetivo del modelo y las prácticas del curso en las que debía basarse, trabajamos el código del notebook con el apoyo descrito, ejecutamos y supervisamos el entrenamiento en Kaggle con GPU, revisamos los resultados de cada iteración, subimos los cinco envíos y comparamos sus puntajes, escogimos el envío 5 (0.90471) como modelo final, verificamos el cumplimiento del enunciado y del comunicado sobre arquitecturas, y preparamos la entrega.

Ningún modelo generativo se usó para etiquetar o clasificar los textos de `eval.csv`. Todas las predicciones enviadas a Kaggle salen del modelo entregado, entrenado solo con `train.csv` y los datos sintéticos descritos arriba.
