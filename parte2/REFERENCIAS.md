# Fuentes, licencias y uso de IA generativa

## Modelos y herramientas

| Recurso | Fuente | Licencia declarada | Uso |
|---|---|---|---|
| XLM-R base | https://huggingface.co/FacebookAI/xlm-roberta-base | MIT | Encoder para clasificación multilingüe |
| mDeBERTa-v3 base | https://huggingface.co/microsoft/mdeberta-v3-base | MIT | Encoder alternativo para clasificación |
| M2M100 418M | https://huggingface.co/facebook/m2m100_418M | MIT | Generación de traducciones del entrenamiento |
| Lingua | https://github.com/pemistahl/lingua-py | Apache-2.0 | Metadatos de idioma y detección de mezcla español-inglés |
| Transformers | https://github.com/huggingface/transformers | Apache-2.0 | Tokenización, entrenamiento y guardado |
| PyTorch | https://github.com/pytorch/pytorch | BSD-3-Clause | Entrenamiento e inferencia |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | Particiones y métricas |

Revisiones de los modelos consultadas el 10 de noviembre de 2026.

- XLM-R base `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- mDeBERTa `a0484667b22365f84929a935b5e50a51f71f159d`.
- M2M100 `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`.

Las fichas y licencias se enlazan para permitir obtener los recursos. Conservar
los avisos de licencia al distribuir pesos y código. Los datos originales del
curso se obtienen de la competencia y los archivos suministrados al grupo. Su
licencia de redistribución pública no se ha establecido. Los nuevos CSV y sus
variantes están excluidos de Git.

## Construcción de datos sintéticos

El comando `augment` genera variantes de fuentes incluidas en la partición de
entrenamiento de cada corrida. La traducción usa M2M100, destinos inglés,
portugués y francés, cuatro haces, generación determinista y hasta 512 tokens
nuevos. La selección recorre clase e idioma fuente, con destinos balanceados.
Se conserva la etiqueta original y se registran fuente, hash, estrategia,
idioma, revisión y parámetros. Una muestra de 100 traducciones exige revisión
humana. Se excluyen las rechazadas y se descarta el lote si la tasa de rechazo
supera 5 %. El manifiesto aprobado conserva el hash de la revisión.

La variación ortográfica elimina diacríticos conservando ñ, cambia a minúsculas
y duplica espacios entre palabras. Mantiene el orden de palabras. El efecto de
estas modificaciones debe reportarse mediante la comparación experimental.

## Uso de IA generativa en el desarrollo

Codex asistió en la lectura del enunciado, inspección de datos y del repositorio,
el diseño del protocolo experimental, la implementación de módulos Python,
notebook, documentación y pruebas de regresión. El equipo debe revisar el código,
ejecutar los experimentos y describir en el notebook las decisiones que adoptó.

La implementación incluye traducción generativa local de ejemplos etiquetados
del entrenamiento mediante M2M100. Su uso efectivo, volumen y revisión se
documentarán con los artefactos de las corridas. Las predicciones de evaluación
se generan con los clasificadores entregados.

Completar antes de entregar con las herramientas adicionales realmente usadas,
quién revisó las traducciones, fecha de cada revisión, filas aceptadas/rechazadas,
experimentos ejecutados y bundle que produjo el envío elegido.
