import re

def extraer_final(texto, proporcion=0.25):
    palabras = texto.split()
    cantidad = max(1, int(len(palabras) * proporcion))
    return " ".join(palabras[-cantidad:])


def extraer_finales_lote(textos, proporcion=0.25):
    return [
        extraer_final(texto, proporcion)
        for texto in textos
    ]

def extraer_ultimas_oraciones(texto, cantidad=1):
    oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())
    oraciones = [o for o in oraciones if o.strip()]
    return " ".join(oraciones[-cantidad:])


def extraer_oraciones_lote(textos, cantidad=1):
    return [
        extraer_ultimas_oraciones(texto, cantidad)
        for texto in textos
    ]


def marcar_conectores(texto, palabras_conector=2):
    # Cada palabra se marca con las primeras palabras de su frase (el conector)
    # y con la posición de la frase contada desde el final de la reseña
    frases = re.split(r"(?<=[.!?;])\s+", texto.strip())
    frases = [f for f in frases if f.strip()]
    tokens = []
    for i, frase in enumerate(frases):
        palabras = re.findall(r"\w+", frase.lower())
        conector = "_".join(palabras[:palabras_conector])
        posicion = len(frases) - i
        tokens += [f"{conector}__{p}" for p in palabras]
        tokens += [f"p{posicion}__{p}" for p in palabras]
    return " ".join(tokens)


def marcar_conectores_lote(textos, palabras_conector=2):
    return [
        marcar_conectores(texto, palabras_conector)
        for texto in textos
    ]
