# Parte 2. Sentimiento multilingüe

Implementación para Python 3.11/3.12 y GPU CUDA de Kaggle o Colab. El notebook
`notebooks/08_Parte2_Deep_Learning.ipynb` utiliza los mismos módulos que esta CLI.
Los resultados competitivos se completan al ejecutar los entrenamientos.

## Preparación

Desde la raíz del repositorio, crear un entorno independiente.

```bash
python3.12 -m venv .venv-parte2
source .venv-parte2/bin/activate
python -m pip install -r parte2/requirements.txt
python -m parte2 --help
```

En Kaggle/Colab activar GPU e Internet antes de instalar. Conservar el wheel CUDA
compatible con el runtime. Después de instalar, reiniciar el kernel si ya se había
importado PyTorch, NumPy o pandas. Las versiones directas están fijadas y cada
experimento captura todas las versiones efectivamente instaladas. La verificación
local usa CPU. Las versiones CUDA necesitan validación en la primera corrida.

```bash
python -m parte2 prepare \
  --train '/ruta/train (1).csv' \
  --eval '/ruta/eval (1).csv' \
  --sample '/ruta/sample_submission (1).csv'
```

Este comando verifica las 20.466/6.781 filas, copia los archivos a `data/parte2`,
detecta idiomas y congela una división aproximada 70/15/15 con semilla 42. Usa
estratos de etiqueta, procedencia e idioma, con retroceso a etiqueta/procedencia
en estratos con menos de 20 grupos. MinHash propone pares similares y Jaccard de
trigramas de palabras >=0,85 confirma los grupos. Esta búsqueda es aproximada.
Las componentes conexas permanecen juntas en las particiones.

Los manifiestos, distribuciones y muestra de revisión quedan en
`artifacts/parte2/prepared`. Revisar `language_review.csv`, especialmente `es-en`
e `incierto`. Si se requieren correcciones, crear un CSV `id,language` y repetir
`prepare --language-overrides ... --prepared artifacts/parte2/prepared_v2` antes
de entrenar. Todos los comandos posteriores deben usar esa misma carpeta.

El idioma es metadato de análisis. El clasificador procesa siempre el texto
completo. `eval.csv` participa en la auditoría de IDs y coincidencias exactas,
y posteriormente en inferencia. La agrupación, los estratos y la tokenización
de entrenamiento usan únicamente `train.csv`.

## Cuatro experimentos iniciales

```bash
python -m parte2 train --name xlmr_1e5 --experiment xlmr_1e5
python -m parte2 train --name xlmr_2e5 --experiment xlmr_2e5
python -m parte2 train --name mdeberta_1e5 --experiment mdeberta_1e5
python -m parte2 train --name mdeberta_2e5 --experiment mdeberta_2e5
```

Los modelos y revisiones están fijados en `parte2/experiments.json`. Cada corrida
guarda configuración, commit, estado del árbol, versiones, métricas por clase,
idioma y procedencia, probabilidades de desarrollo, tiempo y memoria CUDA. Se
selecciona el checkpoint con mejor accuracy de desarrollo.

En mDeBERTa se usa el tokenizador SentencePiece original (`use_fast=False`) para
conservar byte fallback. El bundle registra esta opción y la reutiliza al cargar.

El batch efectivo es 32. Ante OOM se reinicia con menor batch físico, la misma
semilla y más acumulación. Cada época virtual consume `ceil(n_original/32)`
actualizaciones. Así la aumentación conserva el mismo presupuesto de updates.
La duración también depende del hardware y del padding. Comparar tiempos antes
de comprometer la cuota restante. Reservar 30 % para repeticiones y refits.

La longitud inicial es 256 tokens. Si más del 1 % de las fuentes de entrenamiento
se trunca, se amplía a 384 y se registra el truncamiento restante. Se usa padding
dinámico, checkpointing de gradientes y BF16 o FP16 según CUDA. La semilla se fija
y se solicitan algoritmos deterministas con advertencias. Distinto hardware puede
producir variaciones numéricas, por lo que se verifica el CSV con el bundle real.

Para repetir un candidato, usar otro nombre y `--seed 123`. La opción `--epochs 5`
permite ampliar una configuración cuyo historial siga mejorando. Cada corrida
completa es inmutable. Si una sesión se interrumpe, conservar sus registros y
repetir con otro nombre. Los checkpoints permiten inspección y recuperación
manual, pero la CLI reinicia el experimento completo.

## Aumentación y revisión

```bash
python -m parte2 augment --strategy orthographic \
  --output artifacts/parte2/orth_train.csv
python -m parte2 augment --strategy translation --limit 2000 \
  --output artifacts/parte2/mt_train.csv
```

La variante ortográfica conserva ñ y el orden de palabras. La traducción usa
M2M100 con revisión fija, beam search de cuatro haces y destinos en/pt/fr.
Selecciona fuentes monolingües por turnos entre clase e idioma, sin repetir
fuentes. Las reseñas mixtas e inciertas se conservan como datos originales.
Cada variante guarda ID y hash de fuente, idioma, etiqueta heredada y estrategia.

La traducción guarda una caché cada 20 ejemplos. Para continuar una generación
interrumpida usar el mismo comando con `--resume`. Una interrupción anterior al
primer guardado requiere comenzar de nuevo.

Completar `decision` con `accept` o `reject` y `reason` en las 100 filas de
`mt_train_review.csv`. Revisar especialmente negaciones, ironía y concesiones.
Si se rechaza más del 5 %, corregir la generación y repetir la revisión.

```bash
python -m parte2 approve-translations \
  --variants artifacts/parte2/mt_train.csv \
  --review artifacts/parte2/mt_train_review.csv \
  --output artifacts/parte2/mt_train_approved.csv
```

Se excluyen las variantes rechazadas de la muestra. Las restantes conservan su
etiqueta fuente. La auditoría muestral no certifica cada traducción individual.

Suponiendo que gana `xlmr_2e5`, comparar con los mismos parámetros y semilla.

```bash
python -m parte2 train --name xlmr_orth --experiment xlmr_2e5 \
  --variants artifacts/parte2/orth_train.csv
python -m parte2 train --name xlmr_mt --experiment xlmr_2e5 \
  --variants artifacts/parte2/mt_train_approved.csv
```

Si ambas estrategias aportan, probar una corrida con ambos CSV en `--variants`.
Comparar accuracy global y sobre datos nuevos, F1 macro, métricas por idioma y
tiempo. Repetir las dos configuraciones más fuertes con otra semilla.

```bash
python -m parte2 errors --run artifacts/parte2/runs/xlmr_orth \
  --output artifacts/parte2/error_review.csv
python -m parte2 compare \
  --first artifacts/parte2/runs/xlmr_orth \
  --second artifacts/parte2/runs/mdeberta_2e5 \
  --output artifacts/parte2/runs/ensemble
```

Etiquetar los 100 errores con negación, ironía, comparación, concesión,
sentimiento implícito o mezcla de idiomas. El ensamble prueba pesos 0,25/0,50/0,75
y se guarda si mejora accuracy y ningún idioma/procedencia con n>=30 pierde más
de dos puntos frente al mejor modelo individual. También registra solapamiento
de errores e intervalo bootstrap pareado de la ganancia.

## Congelación y comprobación final

Congelar sólo cuando terminen las decisiones sobre modelos y aumentación.
Incluir los componentes individuales de cualquier ensamble que se quiera refitar.

```bash
python -m parte2 freeze \
  --candidates artifacts/parte2/runs/xlmr_orth artifacts/parte2/runs/mdeberta_2e5 \
  --output artifacts/parte2/selection.json
python -m parte2 holdout --selection artifacts/parte2/selection.json
```

La consulta genera un recibo en la carpeta de particiones. Una segunda selección
o una segunda consulta completa se rechazan. Una consulta interrumpida admite
continuación de la misma selección. Este control de proceso depende de conservar
los manifiestos y recibos originales. Las reseñas de holdout no se incluyen en
los archivos de revisión de errores de desarrollo.

## Refits y CSV para Kaggle

Los refits reutilizan arquitectura, semilla, hiperparámetros y número de épocas
del checkpoint seleccionado. Recrear las variantes para el alcance `full`.

```bash
python -m parte2 augment --strategy orthographic --scope full \
  --selection artifacts/parte2/selection.json \
  --output artifacts/parte2/orth_full.csv
python -m parte2 train --name xlmr_orth_full --full \
  --source-run artifacts/parte2/runs/xlmr_orth \
  --selection artifacts/parte2/selection.json \
  --variants artifacts/parte2/orth_full.csv
python -m parte2 predict \
  --bundle artifacts/parte2/runs/xlmr_orth_full/bundle \
  --output submissions/parte2/xlmr_orth.csv
python -m parte2 verify \
  --bundle artifacts/parte2/runs/xlmr_orth_full/bundle \
  --submission submissions/parte2/xlmr_orth.csv
```

Las traducciones full requieren su propia muestra de revisión y aprobación.
Un modelo sin aumentación se refita con el mismo comando `train --full`, omitiendo
`--variants`. Para un ensamble, refitar primero cada componente congelado y usar
`full-ensemble --source-run ... --selection ... --members ... --name ...` con los
componentes full en el mismo orden que `run.json`. Este comando copia todos los
modelos y conserva los pesos del ensamble seleccionado en desarrollo.

Los paquetes de inferencia son autónomos y contienen pesos safetensors,
tokenizador, etiquetas, configuración, revisiones e inventario SHA-256. Se
verifica ese inventario antes de cargar. `predict` exige 6.781 IDs en el orden
de la plantilla. `verify` vuelve a cargar el paquete y compara todas las etiquetas.

Realizar al menos cinco envíos diferentes, guardar su referencia y registrar
únicamente los scores observados en Kaggle.

```bash
python -m parte2 register --submission submissions/parte2/xlmr_orth.csv \
  --score SCORE_OBSERVADO --reference ID_DEL_ENVIO --notes 'Descripción del cambio'
```

El registro rechaza CSV repetidos y actualiza `best_public.json`. Entregar el
bundle exacto del mejor envío público, su notebook y el documento de referencias.
Anotar en el notebook la línea base de Parte 2, los límites de Kaggle, la hora de
cierre y la selección de envíos finales. Estos valores necesitan consulta al
registrar el equipo en la competencia nueva.

## Entrega y sustentación

Exportar el notebook ejecutado con resultados, el bundle del mejor envío y
`parte2/REFERENCIAS.md` actualizado con el uso real de las herramientas. Guardar
los manifiestos, configuraciones, revisión de traducciones y registro de envíos.
Completar la tabla de aportes del notebook con nombres, commits y experimentos.

Preparar explicación de transferencia multilingüe, particiones por grupos,
accuracy frente a F1, control de aumentación, errores por idioma y motivo de
selección del candidato final. Evitar afirmar mejoras hasta medirlas.

Para ejecutar las pruebas de regresión, usar `python -m unittest discover -s parte2/tests -v`.
