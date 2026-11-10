"""Variantes con procedencia verificable y revisión de traducciones."""

import itertools
import unicodedata
from pathlib import Path

import pandas as pd

from .common import exclusive, fingerprint, read_csv, read_json, sha256, write_csv, write_json
from .data import load_prepared

TRANSLATOR = "facebook/m2m100_418M"
TRANSLATOR_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
VARIANT_COLUMNS = ["variant_id", "source_id", "source_hash", "text", "label", "language", "strategy"]


def orthographic(text):
    # Se conserva ñ, cuya modificación puede cambiar el significado.
    protected = text.replace("ñ", "\ue000").replace("Ñ", "\ue001")
    unaccented = "".join(char for char in unicodedata.normalize("NFD", protected) if unicodedata.category(char) != "Mn")
    return "  ".join(unaccented.replace("\ue000", "ñ").replace("\ue001", "Ñ").lower().split())


def source_rows(args):
    frame, manifest = load_prepared(args.prepared, args.data_dir)
    if args.scope == "train" and (Path(args.prepared) / "holdout_receipt.json").exists():
        raise ValueError("La búsqueda está cerrada después de abrir el holdout")
    if args.scope == "full":
        if not args.selection:
            raise ValueError("La aumentación full requiere --selection después de congelar candidatos")
        selection = read_json(args.selection)
        if selection["data_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Selección de otros datos")
    else:
        frame = frame[frame["split"].eq("train")]
    return frame, manifest


def augment(args):
    with exclusive(Path(args.output).parent):
        _augment(args)


def _augment(args):
    frame, manifest = source_rows(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix(".json")
    settings = {"strategy": args.strategy, "scope": args.scope, "seed": args.seed,
                "data_fingerprint": manifest["fingerprint"], "limit": args.limit,
                "translator": TRANSLATOR if args.strategy == "translation" else None,
                "revision": TRANSLATOR_REVISION if args.strategy == "translation" else None,
                "num_beams": 4, "max_new_tokens": 512}
    settings["fingerprint"] = fingerprint(settings)
    variants = []
    if output.exists():
        if not args.resume or not meta_path.exists():
            raise FileExistsError("La aumentación existe. Use --resume para continuar una traducción interrumpida")
        previous = read_json(meta_path)
        if previous["fingerprint"] != settings["fingerprint"] or previous.get("complete"):
            raise ValueError("Configuración diferente o aumentación ya completa")
        if sha256(output) != previous["csv_sha256"]:
            raise ValueError("Caché de aumentación modificada")
        variants = read_csv(output, VARIANT_COLUMNS).to_dict("records")
    if args.strategy == "orthographic":
        for row in frame.itertuples():
            text = orthographic(row.text)
            if text != row.text:
                variants.append({"variant_id": f"orth-{row.id}", "source_id": row.id,
                                 "source_hash": row.text_hash, "text": text, "label": row.label,
                                 "language": row.language, "strategy": "orthographic"})
        variants = variants[:args.limit] if args.limit else variants
    else:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        from .modeling import device_for

        device = device_for(args.device)
        tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR, revision=TRANSLATOR_REVISION)
        model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR, revision=TRANSLATOR_REVISION).to(device).eval()
        completed = {row["variant_id"] for row in variants}
        # Round-robin de clases e idiomas fuente, con destinos balanceados.
        groups = [iter(group.sample(frac=1, random_state=args.seed).itertuples()) for _, group in frame[frame["language"].isin(["es", "en", "pt", "fr"])].groupby(["label", "language"], sort=True)]
        ordered = [row for batch in itertools.zip_longest(*groups) for row in batch if row is not None]
        tasks = []
        target_counts = {target: 0 for target in ["en", "pt", "fr"]}
        for row in ordered:
            possible = [target for target in target_counts if target != row.language]
            target = min(possible, key=lambda lang: target_counts[lang])
            target_counts[target] += 1
            tasks.append((row, target))
            if len(tasks) >= (args.limit or 2000):
                break
        for index, (row, target) in enumerate(tasks):
            variant_id = f"mt-{row.id}-{target}"
            if variant_id in completed:
                continue
            tokenizer.src_lang = row.language
            encoded = tokenizer(row.text, return_tensors="pt", truncation=False).to(device)
            if encoded["input_ids"].shape[1] > 1024:
                raise ValueError(f"Fuente demasiado larga para traducir sin truncar: {row.id}")
            with torch.inference_mode():
                tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target), num_beams=4, max_new_tokens=512, do_sample=False)
            if tokens.shape[1] >= 513 and tokens[0, -1].item() != tokenizer.eos_token_id:
                raise ValueError(f"Traducción truncada de {row.id}")
            text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            if not text:
                raise ValueError(f"Traducción vacía de {row.id}")
            variants.append({"variant_id": variant_id, "source_id": row.id, "source_hash": row.text_hash,
                             "text": text, "label": row.label, "language": target, "strategy": "translation"})
            if (index + 1) % 20 == 0:
                write_csv(output, pd.DataFrame(variants, columns=VARIANT_COLUMNS))
                write_json(meta_path, {**settings, "complete": False, "csv_sha256": sha256(output)})
                print(f"Traducciones {index + 1}/{len(tasks)}", flush=True)
    if not variants:
        raise ValueError("No se produjeron variantes")
    result = pd.DataFrame(variants, columns=VARIANT_COLUMNS)
    write_csv(output, result)
    metadata = {**settings, "complete": True, "csv_sha256": sha256(output), "rows": len(result)}
    if args.strategy == "translation":
        sample = result.sample(n=min(100, len(result)), random_state=args.seed)
        sample = sample.merge(frame[["id", "text"]].rename(columns={"id": "source_id", "text": "source_text"}), on="source_id", validate="many_to_one")
        sample["decision"] = ""
        sample["reason"] = ""
        review = output.with_name(output.stem + "_review.csv")
        write_csv(review, sample)
        metadata["review_variant_ids"] = sample["variant_id"].tolist()
        metadata["review_file"] = review.name
    write_json(meta_path, metadata)
    print(f"Variantes guardadas en {output}", flush=True)


def approve_translations(args):
    with exclusive(Path(args.output).parent):
        _approve_translations(args)


def _approve_translations(args):
    path = Path(args.variants)
    meta = read_json(path.with_suffix(".json"))
    if meta["strategy"] != "translation" or not meta["complete"] or meta["csv_sha256"] != sha256(path):
        raise ValueError("Traducciones incompletas o modificadas")
    review = read_csv(args.review, ["variant_id", "decision", "reason"])
    if review["variant_id"].duplicated().any() or set(review["variant_id"]) != set(meta["review_variant_ids"]):
        raise ValueError("La revisión debe cubrir exactamente la muestra de auditoría")
    if not review["decision"].isin(["accept", "reject"]).all():
        raise ValueError("Completar las decisiones accept/reject de toda la muestra")
    rejection_rate = float(review["decision"].eq("reject").mean())
    if rejection_rate > 0.05:
        raise ValueError("Más del 5 % cambia o pierde significado. Corregir la generación y repetir la auditoría")
    variants = read_csv(path, VARIANT_COLUMNS)
    rejected = set(review.loc[review["decision"].eq("reject"), "variant_id"])
    result = variants[~variants["variant_id"].isin(rejected)]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    write_csv(output, result)
    write_json(output.with_suffix(".json"), {**meta, "csv_sha256": sha256(output), "rows": len(result),
               "reviewed": True, "review_sha256": sha256(args.review), "rejection_rate": rejection_rate,
               "audit": "100-row sample (or all rows if fewer); unreviewed rows inherit source label"})


def load_variants(paths, sources, manifest, scope):
    frames = []
    for path in paths:
        meta = read_json(Path(path).with_suffix(".json"))
        if meta["data_fingerprint"] != manifest["fingerprint"] or meta["scope"] != scope:
            raise ValueError("Aumentación de otra partición o alcance")
        if not meta["complete"] or sha256(path) != meta["csv_sha256"]:
            raise ValueError("Aumentación incompleta o modificada")
        if meta["strategy"] == "translation" and not meta.get("reviewed"):
            raise ValueError("La traducción requiere aprobar la auditoría primero")
        variants = read_csv(path, VARIANT_COLUMNS)
        if variants["variant_id"].duplicated().any() or not set(variants["source_id"]).issubset(set(sources["id"])):
            raise ValueError("Variante duplicada o fuente fuera del entrenamiento permitido")
        lookup = sources.set_index("id")
        if not variants["label"].equals(variants["source_id"].map(lookup["label"]).rename("label")):
            raise ValueError("La etiqueta de una variante difiere de su fuente")
        if not variants["source_hash"].equals(variants["source_id"].map(lookup["text_hash"]).rename("source_hash")):
            raise ValueError("Texto fuente diferente del original")
        frames.append(variants)
    if not frames:
        return pd.DataFrame(columns=VARIANT_COLUMNS)
    result = pd.concat(frames, ignore_index=True)
    if result["variant_id"].duplicated().any():
        raise ValueError("Se cargó una variante más de una vez")
    return result
