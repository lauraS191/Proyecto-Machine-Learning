"""Auditoría, detección de idioma y particiones por grupos antes de aumentar."""

import hashlib
import re
import shutil
import unicodedata
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import exclusive, fingerprint, read_csv, read_json, sha256, write_csv, write_json


def normalize(text):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def text_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def language_detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(
        Language.SPANISH, Language.ENGLISH, Language.PORTUGUESE, Language.FRENCH
    ).build()


def detect_language(text, detector):
    scores = detector.compute_language_confidence_values(text)
    if not scores or scores[0].value - scores[1].value < 0.10:
        return "incierto"
    codes = {"SPANISH": "es", "ENGLISH": "en", "PORTUGUESE": "pt", "FRENCH": "fr"}
    # Lingua segmenta también cambios dentro de una misma oración.
    counts = defaultdict(int)
    for segment in detector.detect_multiple_languages_of(text):
        counts[codes.get(segment.language.name, "incierto")] += segment.word_count
    total = sum(counts.values())
    if total and counts["es"] >= 3 and counts["en"] >= 3:
        if min(counts["es"], counts["en"]) / total >= 0.15:
            return "es-en"
    return codes.get(scores[0].language.name, "incierto")


def duplicate_groups(texts, seed=42, threshold=0.85):
    """MinHash LSH (64 hashes, 16 bandas) y Jaccard exacto de trigramas.

    La búsqueda de candidatos es aproximada. Los pares aceptados siempre
    cumplen el umbral Jaccard, y sus componentes conexas comparten partición.
    """
    parent = list(range(len(texts)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    shingles = []
    buckets = defaultdict(list)
    exact = {}
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**32, size=64, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**32, size=64, dtype=np.uint64)
    candidate_count = accepted = 0
    for index, text in enumerate(texts):
        normalized = normalize(text)
        words = normalized.split()
        tokens = {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))}
        shingles.append(tokens)
        values = np.array([int.from_bytes(hashlib.blake2s(t.encode(), digest_size=4).digest(), "little") for t in sorted(tokens)], dtype=np.uint64)
        signature = ((a[:, None] * values + b[:, None]) & np.uint64(2**32 - 1)).min(axis=1)
        candidates = set()
        for band in range(16):
            key = (band, signature[band * 4:band * 4 + 4].tobytes())
            candidates.update(buckets[key])
            buckets[key].append(index)
        if normalized in exact:
            candidates.add(exact[normalized])
        exact[normalized] = index
        for other in sorted(candidates):
            candidate_count += 1
            similarity = len(tokens & shingles[other]) / len(tokens | shingles[other])
            if similarity >= threshold:
                parent[find(index)] = find(other)
                accepted += 1
    return [f"g{find(i)}" for i in range(len(texts))], {
        "method": "MinHash64/16x4 + Jaccard word trigrams",
        "threshold": threshold, "candidate_pairs": candidate_count, "accepted_pairs": accepted,
        "groups": len({find(i) for i in range(len(texts))}),
    }


def assign_splits(frame, seed):
    strata = frame["label"] + "/" + frame["origin"] + "/" + frame["language"]
    group_counts = pd.DataFrame({"stratum": strata, "group": frame["group"]}).drop_duplicates()["stratum"].value_counts()
    rare = strata.map(group_counts).lt(20)
    base = frame["label"] + "/" + frame["origin"]
    # Colapsar toda la categoría evita crear un nuevo estrato de una sola fila.
    fallback = base.isin(set(base[rare]))
    strata = strata.where(~fallback, base)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    assigned = np.empty(len(frame), dtype=object)
    for fold, (_, indices) in enumerate(splitter.split(frame, strata, frame["group"])):
        assigned[indices] = "train" if fold < 14 else "dev" if fold < 17 else "holdout"
    return assigned


def prepare(args):
    output = Path(args.prepared)
    with exclusive(output):
        if (output / "manifest.json").exists():
            raise ValueError("La partición ya está congelada. Use otra carpeta para una nueva versión.")
        train = read_csv(args.train, ["id", "text", "label"])
        evaluation = read_csv(args.eval, ["id", "text"])
        sample = read_csv(args.sample, ["id", "answer"])
        legacy = read_csv(args.legacy, ["id", "text", "label"])
        if len(train) != 20466 or len(evaluation) != 6781:
            raise ValueError("Se esperaban los archivos oficiales de Parte 2 (20466/6781 filas)")
        if "label" in evaluation or not evaluation["id"].equals(sample["id"]):
            raise ValueError("Evaluación etiquetada o plantilla desalineada")
        train["text_hash"] = train["text"].map(text_hash)
        legacy_hashes = dict(zip(legacy["text"].map(text_hash), legacy["label"]))
        if not set(legacy_hashes).issubset(set(train["text_hash"])):
            raise ValueError("Los datos originales no están contenidos en Parte 2")
        old = train["text_hash"].isin(legacy_hashes)
        if not train.loc[old, "label"].equals(train.loc[old, "text_hash"].map(legacy_hashes).rename("label")):
            raise ValueError("Cambió una etiqueta de Parte 1")
        overlap = set(train["text_hash"]) & set(evaluation["text"].map(text_hash))
        if overlap:
            raise ValueError("Hay textos compartidos entre entrenamiento y evaluación")
        train["origin"] = np.where(old, "parte1", "nuevo")
        detector = language_detector()
        print("Detectando idiomas y agrupando reseñas similares...", flush=True)
        train["language"] = [detect_language(text, detector) for text in train["text"]]
        if args.language_overrides:
            overrides = read_csv(args.language_overrides, ["id", "language"])
            if not set(overrides["id"]).issubset(set(train["id"])) or not overrides["language"].isin(["es", "en", "pt", "fr", "es-en", "incierto"]).all():
                raise ValueError("Correcciones de idioma inválidas")
            train["language"] = train["id"].map(overrides.set_index("id")["language"]).fillna(train["language"])
        train["group"], grouping = duplicate_groups(train["text"].tolist(), args.seed)
        train["split"] = assign_splits(train, args.seed)
        if train.groupby("group")["split"].nunique().max() != 1:
            raise AssertionError("Grupo compartido entre particiones")
        sources = {"train": args.train, "eval": args.eval, "sample": args.sample, "legacy": args.legacy}
        hashes = {key: sha256(path) for key, path in sources.items()}
        overrides_hash = sha256(args.language_overrides) if args.language_overrides else None
        manifest = {"seed": args.seed, "source_hashes": hashes, "grouping": grouping,
                    "language_detector": f"lingua-language-detector=={version('lingua-language-detector')}",
                    "language_overrides_sha256": overrides_hash,
                    "rows": len(train), "eval_rows": len(evaluation)}
        write_csv(output / "splits.csv", train.drop(columns="text"))
        manifest["splits_sha256"] = sha256(output / "splits.csv")
        manifest["fingerprint"] = fingerprint(manifest)
        # Sólo train alimenta el entrenamiento. Eval se conserva como entrada de inferencia.
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in [("train", "train.csv"), ("eval", "eval.csv"), ("sample", "sample_submission.csv")]:
            destination = data_dir / filename
            if destination.resolve() != Path(sources[key]).resolve():
                if destination.exists() and sha256(destination) != hashes[key]:
                    raise ValueError(f"Ya existe otro archivo en {destination}")
                shutil.copyfile(sources[key], destination)
        counts = train.groupby(["split", "origin", "language", "label"]).size().reset_index(name="n")
        write_csv(output / "distribution.csv", counts)
        # Muestra de idioma para revisión humana sin exponer el holdout.
        review = train[train["split"].eq("train")].groupby("language", group_keys=False).sample(n=1, random_state=args.seed)
        more = train[train["split"].eq("train")].sample(n=100, random_state=args.seed)
        write_csv(output / "language_review.csv", pd.concat([review, more]).drop_duplicates("id")[["id", "text", "language"]])
        write_json(output / "manifest.json", manifest)
        print(train.groupby(["split", "origin"]).size().to_string(), flush=True)


def load_prepared(prepared, data_dir):
    prepared = Path(prepared)
    manifest = read_json(prepared / "manifest.json")
    if fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("El manifiesto fue modificado")
    if sha256(prepared / "splits.csv") != manifest["splits_sha256"]:
        raise ValueError("Las particiones congeladas fueron modificadas")
    path = Path(data_dir) / "train.csv"
    if sha256(path) != manifest["source_hashes"]["train"]:
        raise ValueError("Cambió el entrenamiento original")
    frame = read_csv(path, ["id", "text", "label"])
    splits = read_csv(prepared / "splits.csv", ["id", "label", "text_hash", "origin", "language", "group", "split"])
    if set(frame["id"]) != set(splits["id"]):
        raise ValueError("IDs de partición inconsistentes")
    frame = frame.merge(splits, on=["id", "label"], validate="one_to_one")
    if len(frame) != manifest["rows"] or not frame["text"].map(text_hash).equals(frame["text_hash"]):
        raise ValueError("Textos o etiquetas diferentes del manifiesto")
    if frame.groupby("group")["split"].nunique().max() != 1:
        raise ValueError("Fuga de grupos entre particiones")
    if not frame["split"].isin(["train", "dev", "holdout"]).all():
        raise ValueError("Particiones desconocidas")
    return frame, manifest
