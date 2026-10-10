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


def segmentar_clausulas(texto):
    # Divide en oraciones y, dentro de cada oración, antes de conjunciones
    # como "pero", "aunque" o ", y", que suelen separar dos opiniones
    frases = re.split(r"(?<=[.!?;])\s+", texto.strip())
    conjunciones = r"(?:pero|aunque|y|sin embargo|mientras que|en cambio|pues|porque|ya que|así que)"
    clausulas = []
    for frase in frases:
        partes = re.split(
            r",\s*(?=" + conjunciones + r"\b)|\s+(?=(?:pero|aunque|sin embargo)\b)",
            frase,
            flags=re.I
        )
        clausulas += [p for p in partes if p and p.strip()]
    return clausulas


def marcar_clausulas(texto, palabras_conector=1):
    # Igual que marcar_conectores, pero por cláusulas, y agregando también
    # el conector de la cláusula anterior
    clausulas = segmentar_clausulas(texto)
    conectores = [
        "_".join(re.findall(r"\w+", c.lower())[:palabras_conector])
        for c in clausulas
    ]
    tokens = []
    for i, clausula in enumerate(clausulas):
        palabras = re.findall(r"\w+", clausula.lower())
        posicion = len(clausulas) - i
        tokens += [f"{conectores[i]}__{p}" for p in palabras]
        tokens += [f"p{posicion}__{p}" for p in palabras]
        if i > 0:
            tokens += [f"tras_{conectores[i - 1]}__{p}" for p in palabras]
    return " ".join(tokens)


def marcar_clausulas_lote(textos, palabras_conector=1):
    return [
        marcar_clausulas(texto, palabras_conector)
        for texto in textos
    ]
