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
