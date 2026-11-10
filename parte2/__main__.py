"""Ejecutar desde la raíz con python -m parte2 --help."""

import argparse
from pathlib import Path

from .common import DEFAULT_DATA, DEFAULT_PREPARED, DEFAULT_RUNS, ROOT


def main():
    parser = argparse.ArgumentParser(description="Parte 2: sentimiento multilingüe reproducible")
    commands = parser.add_subparsers(dest="command", required=True)

    def data_arguments(command):
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
        command.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)

    def inference_arguments(command):
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")

    prepare = commands.add_parser("prepare", help="Auditar archivos oficiales y congelar particiones")
    data_arguments(prepare)
    prepare.add_argument("--train", type=Path, required=True)
    prepare.add_argument("--eval", type=Path, required=True)
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--legacy", type=Path, default=ROOT / "data" / "train.csv")
    prepare.add_argument("--language-overrides", type=Path)
    prepare.add_argument("--seed", type=int, default=42)

    augment = commands.add_parser("augment", help="Generar variantes de train, o full tras congelar")
    data_arguments(augment)
    augment.add_argument("--strategy", choices=["orthographic", "translation"], required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--scope", choices=["train", "full"], default="train")
    augment.add_argument("--selection", type=Path)
    augment.add_argument("--limit", type=int, default=0)
    augment.add_argument("--seed", type=int, default=42)
    augment.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    augment.add_argument("--resume", action="store_true")

    approve = commands.add_parser("approve-translations", help="Validar decisiones de la auditoría humana")
    approve.add_argument("--variants", type=Path, required=True)
    approve.add_argument("--review", type=Path, required=True)
    approve.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train", help="Fine-tuning de un candidato o refit full")
    data_arguments(train)
    train.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    train.add_argument("--name", required=True)
    train.add_argument("--config", type=Path, default=ROOT / "parte2" / "experiments.json")
    train.add_argument("--experiment", default="xlmr_2e5")
    train.add_argument("--variants", nargs="*", type=Path, default=[])
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    train.add_argument("--allow-cpu-training", action="store_true")
    train.add_argument("--full", action="store_true")
    train.add_argument("--source-run", type=Path)
    train.add_argument("--selection", type=Path)

    compare = commands.add_parser("compare", help="Comparar tres pesos de ensamble en desarrollo")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    errors = commands.add_parser("errors", help="Muestra estratificada de errores para sustentación")
    data_arguments(errors)
    errors.add_argument("--run", type=Path, required=True)
    errors.add_argument("--limit", type=int, default=100)
    errors.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser("freeze", help="Congelar candidatos antes de consultar holdout")
    freeze.add_argument("--candidates", nargs="+", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    holdout = commands.add_parser("holdout", help="Evaluar una única selección congelada")
    data_arguments(holdout)
    inference_arguments(holdout)
    holdout.add_argument("--selection", type=Path, required=True)

    full_ensemble = commands.add_parser("full-ensemble", help="Empaquetar refits con los pesos congelados")
    full_ensemble.add_argument("--source-run", type=Path, required=True)
    full_ensemble.add_argument("--selection", type=Path, required=True)
    full_ensemble.add_argument("--members", nargs="+", type=Path, required=True)
    full_ensemble.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    full_ensemble.add_argument("--name", required=True)

    for name in ["predict", "verify"]:
        command = commands.add_parser(name, help="Generar envío" if name == "predict" else "Recargar modelo y reproducir CSV")
        inference_arguments(command)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--input", type=Path, default=DEFAULT_DATA / "eval.csv")
        command.add_argument("--sample", type=Path, default=DEFAULT_DATA / "sample_submission.csv")
        command.add_argument("--output" if name == "predict" else "--submission", type=Path, required=True)

    register = commands.add_parser("register", help="Registrar un envío realizado y su accuracy pública")
    register.add_argument("--submission", type=Path, required=True)
    register.add_argument("--score", type=float, required=True)
    register.add_argument("--reference", required=True, help="ID o enlace del envío en Kaggle")
    register.add_argument("--notes", default="")
    register.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "parte2" / "submissions.json")

    args = parser.parse_args()
    if getattr(args, "batch_size", 1) is not None and getattr(args, "batch_size", 1) < 1:
        parser.error("batch-size debe ser positivo")
    if getattr(args, "limit", 0) < 0:
        parser.error("limit no puede ser negativo")
    if args.command == "prepare":
        from .data import prepare as operation
    elif args.command == "augment":
        from .augment import augment as operation
    elif args.command == "approve-translations":
        from .augment import approve_translations as operation
    elif args.command == "train":
        from .modeling import train as operation
    else:
        from . import evaluation
        operation = {"compare": evaluation.compare, "errors": evaluation.error_review,
                     "freeze": evaluation.freeze, "holdout": evaluation.holdout,
                     "full-ensemble": evaluation.full_ensemble, "predict": evaluation.predict,
                     "verify": evaluation.verify_submission, "register": evaluation.register_submission}[args.command]
    operation(args)


if __name__ == "__main__":
    main()
