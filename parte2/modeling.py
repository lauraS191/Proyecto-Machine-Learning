"""Fine-tuning con presupuesto comparable, bundles e inferencia portable."""

import gc
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import LABEL2ID, LABELS, environment, new_run, read_json, seal_bundle, sha256, verify_bundle, write_csv, write_json
from .data import load_prepared


def device_for(requested="auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def probabilities(bundle, texts, batch_size=8, device="auto"):
    """Las columnas siempre siguen LABELS, incluso si cambia el orden del modelo."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bundle = Path(bundle)
    metadata = verify_bundle(bundle)
    if metadata["labels"] != LABELS:
        raise ValueError("Correspondencia de etiquetas incompatible")
    if metadata["kind"] == "ensemble":
        parts = metadata["members"]
        if not parts or not np.isclose(sum(part["weight"] for part in parts), 1):
            raise ValueError("Pesos del ensamble inválidos")
        result = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
        for part in parts:
            if part["weight"] <= 0:
                raise ValueError("Peso de ensamble no positivo")
            result += part["weight"] * probabilities(bundle / part["path"], texts, batch_size, device)
        return result
    if metadata["kind"] != "transformer":
        raise ValueError("Tipo de bundle desconocido")
    target = device_for(device)
    tokenizer = AutoTokenizer.from_pretrained(bundle, local_files_only=True, use_fast=metadata.get("tokenizer_use_fast", True))
    model = AutoModelForSequenceClassification.from_pretrained(bundle, local_files_only=True).to(target).eval()
    mapping = {int(index): label for index, label in model.config.id2label.items()}
    if set(mapping.values()) != set(LABELS):
        raise ValueError("id2label del modelo no corresponde al bundle")
    columns = [next(index for index, label in mapping.items() if label == wanted) for wanted in LABELS]
    outputs = []
    effective_batch = batch_size
    start = 0
    while start < len(texts):
        try:
            encoded = tokenizer(texts[start:start + effective_batch], padding=True, truncation=True,
                                max_length=metadata["max_length"], return_tensors="pt").to(target)
            with torch.inference_mode():
                logits = model(**encoded).logits.float()
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy()[:, columns])
            start += effective_batch
        except torch.OutOfMemoryError:
            if effective_batch == 1:
                raise
            effective_batch //= 2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outputs) if outputs else np.empty((0, len(LABELS)))


def prediction_frame(frame, probs):
    if probs.shape != (len(frame), len(LABELS)) or not np.isfinite(probs).all():
        raise ValueError("Probabilidades ausentes o no finitas")
    if (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilidades inválidas")
    result = frame.drop(columns="text", errors="ignore").copy().reset_index(drop=True)
    result["prediction"] = np.asarray(LABELS)[probs.argmax(axis=1)]
    for index, label in enumerate(LABELS):
        result[f"p_{label}"] = probs[:, index]
    return result


def train(args):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    from .augment import load_variants
    from .evaluation import metrics

    frame, manifest = load_prepared(args.prepared, args.data_dir)
    if not args.full and (Path(args.prepared) / "holdout_receipt.json").exists():
        raise ValueError("El holdout ya se abrió. La fase de búsqueda está cerrada para estas particiones")
    catalog = read_json(args.config)
    config = {**catalog["defaults"], **catalog["experiments"][args.experiment]}
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        if not 1 <= args.epochs <= 5:
            raise ValueError("El plan admite de 1 a 5 épocas")
        config["epochs"] = args.epochs
    source_metadata = None
    if args.full:
        if not args.selection or not args.source_run:
            raise ValueError("El refit requiere --selection y --source-run")
        selection = read_json(args.selection)
        source = Path(args.source_run).resolve()
        source_metadata = read_json(source / "run.json")
        selected = next((item for item in selection["candidates"] if item["run_name"] == source.name), None)
        if selection["data_fingerprint"] != manifest["fingerprint"] or not selected:
            raise ValueError("Candidato no congelado para estos datos")
        if sha256(source / "run.json") != selected["run_sha256"]:
            raise ValueError("El candidato cambió después de congelarlo")
        if source_metadata["scope"] != "train" or source_metadata["kind"] != "transformer":
            raise ValueError("Refit sólo de un transformer de desarrollo")
        config = {**source_metadata["config"], "epochs": source_metadata["best_epoch"]}
        if bool(args.variants) != bool(source_metadata["augmentations"]):
            raise ValueError("El refit debe conservar las estrategias de aumentación del candidato")
        source_strategies = sorted(item["strategy"] for item in source_metadata["augmentations"])
        full_strategies = sorted(read_json(Path(path).with_suffix(".json"))["strategy"] for path in args.variants)
        if source_strategies != full_strategies:
            raise ValueError("Las estrategias de aumentación del refit difieren del candidato")
    scope = "full" if args.full else "train"
    sources = frame if args.full else frame[frame["split"].eq("train")]
    variants = load_variants(args.variants, sources, manifest, scope)
    training = pd.concat([sources[["text", "label"]], variants[["text", "label"]]], ignore_index=True)
    development = frame[frame["split"].eq("dev")].reset_index(drop=True)
    target = device_for(args.device)
    if target.type != "cuda" and not args.allow_cpu_training:
        raise ValueError("Este entrenamiento requiere CUDA. Use Kaggle/Colab con GPU, o --allow-cpu-training para pruebas locales")
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    device_count = torch.cuda.device_count() if target.type == "cuda" else 1
    if config["effective_batch_size"] % (config["batch_size"] * device_count):
        raise ValueError("batch físico × GPUs debe dividir el batch efectivo")
    directory = new_run(args.runs, args.name)
    metadata = {"status": "running", "kind": "transformer", "scope": scope, "config": config,
                "data_fingerprint": manifest["fingerprint"], "environment": environment(),
                "original_rows": len(sources), "augmented_rows": len(variants), "device": str(target), "device_count": device_count,
                "source_run": str(Path(args.source_run).resolve()) if args.source_run else None,
                "source_run_sha256": sha256(Path(args.source_run) / "run.json") if args.source_run else None,
                "augmentations": [{"path": os.path.relpath(Path(path).resolve(), directory.resolve()),
                                   "strategy": read_json(Path(path).with_suffix(".json"))["strategy"],
                                   "sha256": sha256(path)} for path in args.variants]}
    write_json(directory / "run.json", metadata)
    started = time.monotonic()
    try:
        set_seed(config["seed"])
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], use_fast=config.get("tokenizer_use_fast", True))
        lengths = [len(tokens) for tokens in tokenizer(sources["text"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]]
        max_length = config["max_length"]
        initial_truncation = float(np.mean(np.asarray(lengths) > max_length))
        if args.full:
            max_length = source_metadata["token_lengths"]["max_length"]
        elif initial_truncation > 0.01:
            max_length = 384
        metadata["token_lengths"] = {"initial_limit": config["max_length"], "initial_truncated_fraction": initial_truncation,
                                     "max_length": max_length, "truncated_fraction": float(np.mean(np.asarray(lengths) > max_length)),
                                     "max_observed": max(lengths)}
        # Cada época virtual consume los mismos updates sobre el tamaño original.
        steps_per_epoch = math.ceil(len(sources) / config["effective_batch_size"])
        max_steps = math.ceil(config["epochs"] * steps_per_epoch)
        write_json(directory / "run.json", metadata)

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.tokens = tokenizer(rows["text"].tolist(), truncation=True, max_length=max_length)
                self.labels = rows["label"].map(LABEL2ID).tolist()

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                item = {key: value[index] for key, value in self.tokens.items()}
                item["labels"] = self.labels[index]
                return item

        train_dataset = TextDataset(training)
        dev_dataset = None if args.full else TextDataset(development)

        def compute_metrics(output):
            prediction = output.predictions.argmax(axis=-1)
            return {"accuracy": accuracy_score(output.label_ids, prediction),
                    "f1_macro": f1_score(output.label_ids, prediction, labels=list(range(3)), average="macro", zero_division=0)}

        batch_size = config["batch_size"]
        while True:
            model = AutoModelForSequenceClassification.from_pretrained(
                config["model"], revision=config["revision"], num_labels=3,
                id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
            )
            training_args = TrainingArguments(
                output_dir=str(directory / "checkpoints"), max_steps=max_steps,
                per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=config["effective_batch_size"] // (batch_size * device_count),
                learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
                warmup_ratio=config["warmup_ratio"], lr_scheduler_type="linear", optim="adamw_torch",
                eval_strategy="no" if args.full else "steps", eval_steps=steps_per_epoch,
                save_strategy="steps", save_steps=steps_per_epoch, save_total_limit=2,
                load_best_model_at_end=not args.full, metric_for_best_model="accuracy", greater_is_better=True,
                fp16=target.type == "cuda" and not torch.cuda.is_bf16_supported(),
                bf16=target.type == "cuda" and torch.cuda.is_bf16_supported(),
                use_cpu=target.type == "cpu", seed=config["seed"], data_seed=config["seed"],
                report_to="none", logging_steps=max(1, steps_per_epoch // 10),
                dataloader_num_workers=0, dataloader_pin_memory=target.type == "cuda",
                gradient_checkpointing=True,
            )
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                              eval_dataset=dev_dataset, processing_class=tokenizer,
                              data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
            try:
                trainer.train()
                break
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                # Reinicia desde la misma semilla, conservando batch efectivo y presupuesto.
                batch_size //= 2
                metadata.setdefault("oom_retries", []).append({"new_batch_size": batch_size})
                write_json(directory / "run.json", metadata)
                del trainer, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                shutil.rmtree(directory / "checkpoints", ignore_errors=True)
                set_seed(config["seed"])
        bundle = directory / "bundle"
        trainer.save_model(str(bundle))
        tokenizer.save_pretrained(bundle)
        metadata["actual_batch_size"] = batch_size
        metadata["max_steps"] = max_steps
        metadata["steps_per_virtual_epoch"] = steps_per_epoch
        metadata["training_log"] = trainer.state.log_history
        if args.full:
            best_epoch = config["epochs"]
        else:
            best_step = int(Path(trainer.state.best_model_checkpoint).name.rsplit("-", 1)[1])
            best_epoch = best_step / steps_per_epoch
        metadata["best_epoch"] = best_epoch
        seal_bundle(bundle, {"kind": "transformer", "labels": LABELS,
                   "model": config["model"], "revision": config["revision"], "max_length": max_length,
                   "tokenizer_use_fast": bool(tokenizer.is_fast),
                   "data_fingerprint": manifest["fingerprint"], "scope": scope})
        if not args.full:
            predictions = trainer.predict(dev_dataset)
            logits = predictions.predictions
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            prediction = prediction_frame(development, probs)
            write_csv(directory / "dev_predictions.csv", prediction)
            write_json(directory / "dev_metrics.json", metrics(prediction))
            metadata["dev_predictions_sha256"] = sha256(directory / "dev_predictions.csv")
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated() if target.type == "cuda" else None
        metadata["status"] = "complete"
        write_json(directory / "run.json", metadata)
        print(f"Experimento completo en {directory}", flush=True)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        write_json(directory / "run.json", metadata)
        raise
