"""Métricas por subgrupo, ensamble, holdout y registro de envíos."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import LABELS, environment, exclusive, new_run, read_csv, read_json, seal_bundle, sha256, verify_bundle, write_csv, write_json
from .data import load_prepared
from .modeling import prediction_frame, probabilities


def metrics(predictions):
    def summarize(frame):
        return {"n": len(frame), "accuracy": float(accuracy_score(frame["label"], frame["prediction"])),
                "f1_macro": float(f1_score(frame["label"], frame["prediction"], labels=LABELS, average="macro", zero_division=0))}

    result = summarize(predictions)
    result["confusion_matrix"] = confusion_matrix(predictions["label"], predictions["prediction"], labels=LABELS).tolist()
    result["labels"] = LABELS
    result["classification_report"] = classification_report(predictions["label"], predictions["prediction"], labels=LABELS, output_dict=True, zero_division=0)
    result["subgroups"] = {}
    for column in ["language", "origin"]:
        if column in predictions:
            result["subgroups"][column] = {str(name): summarize(frame) for name, frame in predictions.groupby(column)}
    return result


def dev_predictions(directory):
    directory = Path(directory)
    run = read_json(directory / "run.json")
    if run["status"] != "complete" or run["scope"] != "train":
        raise ValueError("Se requiere un experimento completo sobre la partición train")
    if sha256(directory / "dev_predictions.csv") != run["dev_predictions_sha256"]:
        raise ValueError("Predicciones de desarrollo modificadas")
    frame = read_csv(directory / "dev_predictions.csv", ["id", "label", "prediction", "split"] + [f"p_{label}" for label in LABELS])
    if not frame["split"].eq("dev").all():
        raise ValueError("El archivo contiene filas ajenas a desarrollo")
    probs = frame[[f"p_{label}" for label in LABELS]].astype(float).to_numpy()
    recomputed = prediction_frame(frame, probs)
    if not frame["prediction"].equals(recomputed["prediction"]):
        raise ValueError("Predicciones y probabilidades inconsistentes")
    return frame, probs, run


def compare(args):
    first, p_first, first_run = dev_predictions(args.first)
    second, p_second, second_run = dev_predictions(args.second)
    if first_run["data_fingerprint"] != second_run["data_fingerprint"]:
        raise ValueError("Experimentos de particiones diferentes")
    if set(first["id"]) != set(second["id"]):
        raise ValueError("Distintos IDs de desarrollo")
    second = second.set_index("id").loc[first["id"]].reset_index()
    if not first["label"].equals(second["label"]):
        raise ValueError("Etiquetas de desarrollo diferentes")
    p_second = second[[f"p_{label}" for label in LABELS]].astype(float).to_numpy()
    reports = []
    base_metrics = [metrics(first), metrics(second)]
    best_index = int(base_metrics[1]["accuracy"] > base_metrics[0]["accuracy"])
    baseline = base_metrics[best_index]
    best_frame = [first, second][best_index]
    candidates = []
    baseline_correct = best_frame["label"].eq(best_frame["prediction"]).to_numpy()
    for weight in [0.25, 0.5, 0.75]:
        mixed = prediction_frame(first, weight * p_first + (1 - weight) * p_second)
        report = metrics(mixed)
        stable = all(values["accuracy"] >= reference["accuracy"] - 0.02
                     for group, subgroups in baseline["subgroups"].items()
                     for name, reference in subgroups.items() if reference["n"] >= 30
                     for values in [report["subgroups"][group][name]])
        difference = mixed["label"].eq(mixed["prediction"]).to_numpy().astype(float) - baseline_correct
        rng = np.random.default_rng(42)
        bootstrap = [float(rng.choice(difference, size=len(difference), replace=True).mean()) for _ in range(1000)]
        summary = {"weight_first": weight, "accuracy": report["accuracy"], "f1_macro": report["f1_macro"],
                   "stable_subgroups": stable, "gain_vs_best": float(difference.mean()),
                   "paired_bootstrap_gain_95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
        reports.append(summary)
        if stable and report["accuracy"] > baseline["accuracy"]:
            candidates.append((report["accuracy"], report["subgroups"].get("origin", {}).get("nuevo", {}).get("accuracy", 0), weight, mixed))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "comparison.json", {"first": str(args.first), "second": str(args.second),
               "baseline_accuracy": baseline["accuracy"], "candidates": reports,
               "error_overlap": int((~first["label"].eq(first["prediction"]) & ~second["label"].eq(second["prediction"])).sum())})
    if not candidates:
        print("Ningún ensamble mejoró la accuracy con estabilidad por subgrupos", flush=True)
        return
    _, _, weight, mixed = max(candidates, key=lambda candidate: candidate[:2])
    bundle = output / "bundle"
    members = []
    for index, (source, member_weight) in enumerate([(args.first, weight), (args.second, 1 - weight)]):
        relative = f"members/{index}"
        shutil.copytree(Path(source) / "bundle", bundle / relative)
        members.append({"path": relative, "weight": member_weight})
    seal_bundle(bundle, {"kind": "ensemble", "labels": LABELS, "members": members,
               "data_fingerprint": first_run["data_fingerprint"], "scope": "train"})
    write_csv(output / "dev_predictions.csv", mixed)
    write_json(output / "dev_metrics.json", metrics(mixed))
    write_json(output / "run.json", {"status": "complete", "kind": "ensemble", "scope": "train",
               "data_fingerprint": first_run["data_fingerprint"], "environment": environment(),
               "dev_predictions_sha256": sha256(output / "dev_predictions.csv"),
               "components": [{"run_name": Path(path).name, "run_path": os.path.relpath(Path(path).resolve(), output.resolve()), "weight": value}
                              for path, value in [(args.first, weight), (args.second, 1 - weight)]]})
    print(f"Ensamble seleccionado, peso del primer modelo {weight}", flush=True)


def error_review(args):
    predictions, _, _ = dev_predictions(args.run)
    frame, _ = load_prepared(args.prepared, args.data_dir)
    errors = predictions[predictions["label"].ne(predictions["prediction"])].merge(frame[["id", "text"]], on="id", validate="one_to_one")
    samples = [group.sample(frac=1, random_state=42).to_dict("records") for _, group in errors.groupby(["language", "label"])]
    chosen = []
    while samples and len(chosen) < args.limit:
        for sample in samples:
            if sample and len(chosen) < args.limit:
                chosen.append(sample.pop())
        samples = [sample for sample in samples if sample]
    result = pd.DataFrame(chosen, columns=errors.columns)
    result["error_type"] = ""
    result["explanation"] = ""
    write_csv(args.output, result)
    print(f"{len(result)} errores para revisar", flush=True)


def freeze(args):
    output = Path(args.output)
    with exclusive(output.parent):
        if output.exists():
            raise FileExistsError("Ya existe la selección congelada")
        fingerprints = set()
        candidates = []
        names = set()
        for path in args.candidates:
            path = Path(path).resolve()
            _, _, run = dev_predictions(path)
            verify_bundle(path / "bundle")
            if path.name in names:
                raise ValueError("Nombres repetidos de candidatos")
            names.add(path.name)
            fingerprints.add(run["data_fingerprint"])
            candidates.append({"run_name": path.name, "run_path": os.path.relpath(path, output.parent.resolve()),
                               "run_sha256": sha256(path / "run.json"), "bundle_manifest_sha256": sha256(path / "bundle" / "bundle.json")})
        if len(fingerprints) != 1:
            raise ValueError("Candidatos de diferentes particiones")
        write_json(output, {"data_fingerprint": fingerprints.pop(), "candidates": candidates, "environment": environment()})


def holdout(args):
    selection_path = Path(args.selection)
    selection = read_json(selection_path)
    frame, manifest = load_prepared(args.prepared, args.data_dir)
    if selection["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Selección de otra versión de los datos")
    prepared = Path(args.prepared)
    with exclusive(prepared):
        receipt = prepared / "holdout_receipt.json"
        if receipt.exists():
            previous = read_json(receipt)
            if previous["selection_sha256"] != sha256(selection_path) or previous["status"] == "complete":
                raise ValueError("Holdout ya consultado o intento de cambiar candidatos. Ver holdout_receipt.json")
        else:
            write_json(receipt, {"status": "running", "selection_sha256": sha256(selection_path), "environment": environment()})
        test = frame[frame["split"].eq("holdout")].reset_index(drop=True)
        results = []
        for candidate in selection["candidates"]:
            path = selection_path.parent / candidate["run_path"]
            if sha256(path / "run.json") != candidate["run_sha256"] or sha256(path / "bundle" / "bundle.json") != candidate["bundle_manifest_sha256"]:
                raise ValueError("Candidato modificado después de congelar")
            destination = path / "holdout_metrics.json"
            if not destination.exists():
                probs = probabilities(path / "bundle", test["text"].tolist(), args.batch_size, args.device)
                predictions = prediction_frame(test, probs)
                write_csv(path / "holdout_predictions.csv", predictions)
                write_json(destination, metrics(predictions))
            results.append({"run": candidate["run_name"], **read_json(destination)})
        write_json(prepared / "holdout_summary.json", results)
        write_json(receipt, {"status": "complete", "selection_sha256": sha256(selection_path),
                            "candidates": [item["run"] for item in results], "environment": environment()})


def submission_check(prediction, evaluation, sample, expected_rows):
    if list(prediction.columns) != ["id", "answer"]:
        raise ValueError("Se requieren exactamente las columnas id,answer")
    if len(prediction) != expected_rows or prediction["id"].duplicated().any():
        raise ValueError("Número de filas o IDs incorrectos")
    if not prediction["id"].equals(evaluation["id"]) or not prediction["id"].equals(sample["id"]):
        raise ValueError("IDs u orden distintos de evaluación/plantilla")
    if not prediction["answer"].isin(LABELS).all():
        raise ValueError("Etiquetas de envío inválidas")


def predict(args):
    with exclusive(Path(args.output).parent):
        _predict(args)


def _predict(args):
    evaluation = read_csv(args.input, ["id", "text"])
    sample = read_csv(args.sample, ["id", "answer"])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    metadata = read_json(Path(args.bundle) / "bundle.json")
    if metadata["scope"] != "full":
        raise ValueError("Para los envíos finales se requiere un refit full")
    probs = probabilities(args.bundle, evaluation["text"].tolist(), args.batch_size, args.device)
    result = pd.DataFrame({"id": evaluation["id"], "answer": np.asarray(LABELS)[probs.argmax(axis=1)]})
    submission_check(result, evaluation, sample, 6781)
    write_csv(output, result)
    write_json(output.with_suffix(".json"), {"bundle_path": os.path.relpath(Path(args.bundle).resolve(), output.parent.resolve()),
               "bundle_manifest_sha256": sha256(Path(args.bundle) / "bundle.json"), "input_sha256": sha256(args.input),
               "sample_sha256": sha256(args.sample), "csv_sha256": sha256(output), "rows": len(result), "environment": environment()})


def verify_submission(args):
    result = read_csv(args.submission, ["id", "answer"])
    evaluation = read_csv(args.input, ["id", "text"])
    sample = read_csv(args.sample, ["id", "answer"])
    submission_check(result, evaluation, sample, 6781)
    receipt = read_json(Path(args.submission).with_suffix(".json"))
    if receipt["input_sha256"] != sha256(args.input) or receipt["sample_sha256"] != sha256(args.sample) or receipt["csv_sha256"] != sha256(args.submission):
        raise ValueError("Cambió un archivo respecto al recibo de inferencia")
    if receipt["bundle_manifest_sha256"] != sha256(Path(args.bundle) / "bundle.json"):
        raise ValueError("Cambió el bundle respecto al envío")
    probs = probabilities(args.bundle, evaluation["text"].tolist(), args.batch_size, args.device)
    if not np.array_equal(result["answer"], np.asarray(LABELS)[probs.argmax(axis=1)]):
        raise ValueError("El modelo recargado no reproduce todas las predicciones")
    print(f"Verificadas {len(result)} predicciones desde el bundle recargado", flush=True)


def register_submission(args):
    path = Path(args.submission)
    receipt = read_json(path.with_suffix(".json"))
    if sha256(path) != receipt["csv_sha256"] or not 0 <= args.score <= 1:
        raise ValueError("CSV modificado o score inválido")
    registry = Path(args.registry)
    with exclusive(registry.parent):
        entries = read_json(registry) if registry.exists() else []
        if any(entry["csv_sha256"] == receipt["csv_sha256"] for entry in entries):
            raise ValueError("Este conjunto de predicciones ya está registrado")
        entries.append({**receipt, "submission": str(path.resolve()), "public_score": args.score,
                        "kaggle_reference": args.reference, "notes": args.notes})
        write_json(registry, entries)
        best = max(entries, key=lambda entry: entry["public_score"])
        write_json(registry.parent / "best_public.json", best)
        print(f"Envíos diferentes registrados: {len(entries)}. Mejor accuracy pública: {best['public_score']}", flush=True)


def full_ensemble(args):
    original = read_json(Path(args.source_run) / "run.json")
    selection = read_json(args.selection)
    selected = next((item for item in selection["candidates"] if item["run_name"] == Path(args.source_run).name), None)
    if not selected or selected["run_sha256"] != sha256(Path(args.source_run) / "run.json") or original["kind"] != "ensemble":
        raise ValueError("Ensamble no congelado")
    if len(args.members) != len(original["components"]):
        raise ValueError("Faltan componentes full del ensamble")
    for source, member in zip(original["components"], args.members):
        member_meta = read_json(Path(member) / "run.json")
        if member_meta["scope"] != "full" or member_meta["status"] != "complete" or Path(member_meta["source_run"]).name != source["run_name"]:
            raise ValueError("El componente full no corresponde al candidato original")
        frozen_component = next((item for item in selection["candidates"] if item["run_name"] == source["run_name"]), None)
        if not frozen_component or member_meta["source_run_sha256"] != frozen_component["run_sha256"] or member_meta["data_fingerprint"] != selection["data_fingerprint"]:
            raise ValueError("Procedencia del componente full diferente de la selección")
        verify_bundle(Path(member) / "bundle")
    directory = new_run(args.runs, args.name)
    bundle = directory / "bundle"
    members = []
    for index, (component, member) in enumerate(zip(original["components"], args.members)):
        relative = f"members/{index}"
        shutil.copytree(Path(member) / "bundle", bundle / relative)
        members.append({"path": relative, "weight": component["weight"]})
    seal_bundle(bundle, {"kind": "ensemble", "scope": "full", "labels": LABELS,
               "data_fingerprint": selection["data_fingerprint"], "members": members})
    write_json(directory / "run.json", {"status": "complete", "kind": "ensemble", "scope": "full",
               "source_run": str(Path(args.source_run).resolve()), "environment": environment(),
               "data_fingerprint": selection["data_fingerprint"]})
