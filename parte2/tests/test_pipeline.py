"""Regresiones de fuga de datos, artefactos e inferencia por la CLI real."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from parte2.augment import VARIANT_COLUMNS, approve_translations, load_variants, orthographic
from parte2.common import LABELS, ROOT, fingerprint, read_json, seal_bundle, sha256, verify_bundle, write_csv, write_json
from parte2.data import assign_splits, duplicate_groups, load_prepared, text_hash
from parte2.evaluation import compare, metrics, submission_check
from parte2.modeling import prediction_frame


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def cli(self, *arguments, success=True):
        process = subprocess.run([sys.executable, "-m", "parte2", *map(str, arguments)], cwd=ROOT,
                                 env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"},
                                 capture_output=True, text=True, timeout=120)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        else:
            self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        return process

    def fixture(self):
        data = self.root / "data"
        prepared = self.root / "prepared"
        rows = pd.DataFrame({"id": list(map(str, range(60))),
                             "text": [f"producto numero {i} funciona muy bien" for i in range(60)],
                             "label": [LABELS[i % 3] for i in range(60)]})
        write_csv(data / "train.csv", rows)
        splits = rows.drop(columns="text").copy()
        splits["text_hash"] = rows["text"].map(text_hash)
        splits["origin"] = ["parte1" if i % 2 else "nuevo" for i in range(60)]
        splits["language"] = ["es" if i % 2 else "en" for i in range(60)]
        splits["group"] = splits["id"].map(lambda value: "g" + value)
        splits["split"] = ["train"] * 42 + ["dev"] * 9 + ["holdout"] * 9
        write_csv(prepared / "splits.csv", splits)
        manifest = {"rows": 60, "source_hashes": {"train": sha256(data / "train.csv")},
                    "splits_sha256": sha256(prepared / "splits.csv")}
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(prepared / "manifest.json", manifest)
        return data, prepared, manifest

    def test_similar_reviews_share_group_and_split(self):
        texts = ["Este producto funciona perfectamente y lo uso todos los días."] * 2
        texts += ["No sirve, lo devolví ayer.", "No sirve, lo devolvi ayer."]
        groups, _ = duplicate_groups(texts)
        self.assertEqual(groups[0], groups[1])
        frame = pd.DataFrame({"label": [LABELS[i % 3] for i in range(240)], "origin": "nuevo", "language": "es", "group": [f"g{i // 2}" for i in range(240)]})
        frame["split"] = assign_splits(frame, 42)
        self.assertEqual(frame.groupby("group")["split"].nunique().max(), 1)
        self.assertEqual(set(frame["split"]), {"train", "dev", "holdout"})

    def test_augmentation_rejects_dev_sources_and_changed_labels(self):
        data, prepared, manifest = self.fixture()
        rows, _ = load_prepared(prepared, data)
        path = self.root / "variants.csv"
        variant = {"variant_id": "x", "source_id": "42", "source_hash": rows.iloc[42]["text_hash"],
                   "text": "otra reseña", "label": rows.iloc[42]["label"], "language": "es", "strategy": "orthographic"}
        write_csv(path, pd.DataFrame([variant], columns=VARIANT_COLUMNS))
        write_json(path.with_suffix(".json"), {"data_fingerprint": manifest["fingerprint"], "scope": "train",
                   "complete": True, "strategy": "orthographic", "csv_sha256": sha256(path)})
        with self.assertRaisesRegex(ValueError, "fuente fuera"):
            load_variants([path], rows[rows["split"].eq("train")], manifest, "train")
        self.assertEqual(orthographic("AÑO útil"), "año  util")

    def test_modified_files_and_misordered_submissions_fail(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "weights").write_text("original")
        seal_bundle(bundle, {"kind": "test"})
        verify_bundle(bundle)
        (bundle / "weights").write_text("different")
        with self.assertRaisesRegex(ValueError, "modificado"):
            verify_bundle(bundle)
        expected = pd.DataFrame({"id": ["1", "2"], "answer": ["neutral", "positivo"]})
        submission_check(expected, expected, expected, 2)
        with self.assertRaisesRegex(ValueError, "orden"):
            submission_check(expected.iloc[::-1].reset_index(drop=True), expected, expected, 2)
        data, prepared, _ = self.fixture()
        with (data / "train.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Cambió"):
            load_prepared(prepared, data)

    def test_translation_audit_requires_complete_review_and_excludes_rejections(self):
        variants = pd.DataFrame([{"variant_id": f"v{i}", "source_id": str(i), "source_hash": "hash",
                                 "text": "translated text", "label": "neutral", "language": "en", "strategy": "translation"}
                                for i in range(100)], columns=VARIANT_COLUMNS)
        path = self.root / "mt.csv"
        review_path = self.root / "review.csv"
        output = self.root / "approved.csv"
        write_csv(path, variants)
        write_json(path.with_suffix(".json"), {"strategy": "translation", "complete": True,
                   "csv_sha256": sha256(path), "review_variant_ids": variants["variant_id"].tolist()})
        review = variants[["variant_id"]].copy()
        review["decision"] = ""
        review["reason"] = ""
        write_csv(review_path, review)
        args = SimpleNamespace(variants=path, review=review_path, output=output)
        with self.assertRaisesRegex(ValueError, "Completar"):
            approve_translations(args)
        review["decision"] = "accept"
        review.loc[0, "decision"] = "reject"
        review.loc[0, "reason"] = "Cambió la negación"
        write_csv(review_path, review)
        approve_translations(args)
        self.assertEqual(len(pd.read_csv(output)), 99)
        self.assertTrue(read_json(output.with_suffix(".json"))["reviewed"])

    def test_ensemble_aligns_ids_before_combining(self):
        rows = pd.DataFrame({"id": list(map(str, range(9))), "label": [LABELS[i % 3] for i in range(9)],
                             "split": "dev", "language": "es", "origin": "nuevo"})
        directories = []
        for model_index in range(2):
            probs = np.full((9, 3), 0.01)
            for i in range(9):
                correct = i < 6 if model_index == 0 else i >= 3
                probs[i] = 0.01 if correct else 0.3
                probs[i, i % 3 if correct else (i + 1) % 3] = 0.98 if correct else 0.4
            predictions = prediction_frame(rows, probs)
            if model_index:
                predictions = predictions.iloc[::-1].reset_index(drop=True)
            directory = self.root / f"model{model_index}"
            write_csv(directory / "dev_predictions.csv", predictions)
            bundle = directory / "bundle"
            bundle.mkdir()
            (bundle / "placeholder").write_text("test")
            seal_bundle(bundle, {"kind": "test"})
            write_json(directory / "run.json", {"status": "complete", "scope": "train", "data_fingerprint": "same",
                       "dev_predictions_sha256": sha256(directory / "dev_predictions.csv")})
            directories.append(directory)
        destination = self.root / "ensemble"
        compare(SimpleNamespace(first=directories[0], second=directories[1], output=destination))
        self.assertEqual(read_json(destination / "dev_metrics.json")["accuracy"], 1.0)

    def test_cli_train_freeze_holdout_refit_predict_and_reload(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        torch.set_num_threads(1)
        data, prepared, _ = self.fixture()
        base = self.root / "tiny_base"
        base.mkdir()
        (base / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "producto", "numero", "funciona", "muy", "bien"]))
        tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
        tokenizer.save_pretrained(base)
        model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=12,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, num_labels=3))
        model.save_pretrained(base)
        config = self.root / "experiments.json"
        write_json(config, {"defaults": {"seed": 42, "epochs": 1, "batch_size": 4, "effective_batch_size": 8,
                    "max_length": 32, "weight_decay": 0.01, "warmup_ratio": 0.1},
                    "experiments": {"tiny": {"model": str(base), "revision": "local", "learning_rate": 0.00002}}})
        runs = self.root / "runs"
        common = ["--prepared", prepared, "--data-dir", data]
        training = ["--runs", runs, "--device", "cpu", "--allow-cpu-training", "--config", config, "--experiment", "tiny"]
        self.cli("train", "--name", "tiny", *common, *training)
        run = runs / "tiny"
        self.assertEqual(read_json(run / "run.json")["status"], "complete")
        variants = self.root / "orth.csv"
        self.cli("augment", "--strategy", "orthographic", "--output", variants, *common)
        self.cli("train", "--name", "tiny_aug", "--variants", variants, *common, *training)
        self.assertEqual(read_json(run / "run.json")["max_steps"], read_json(runs / "tiny_aug/run.json")["max_steps"])
        selection = self.root / "selection.json"
        self.cli("freeze", "--candidates", run, "--output", selection)
        self.cli("holdout", "--selection", selection, "--device", "cpu", *common)
        self.cli("holdout", "--selection", selection, "--device", "cpu", *common, success=False)
        self.cli("train", "--name", "late", *common, *training, success=False)
        self.cli("train", "--name", "tiny_full", "--full", "--source-run", run, "--selection", selection, *common, *training)
        bundle = runs / "tiny_full" / "bundle"
        evaluation = pd.DataFrame({"id": list(map(str, range(6781))), "text": "producto funciona bien"})
        sample = pd.DataFrame({"id": evaluation["id"], "answer": "neutral"})
        write_csv(data / "eval.csv", evaluation)
        write_csv(data / "sample_submission.csv", sample)
        output = self.root / "submission.csv"
        inference = ["--bundle", bundle, "--input", data / "eval.csv", "--sample", data / "sample_submission.csv", "--device", "cpu", "--batch-size", "128"]
        self.cli("predict", "--output", output, *inference)
        self.cli("verify", "--submission", output, *inference)
        registry = self.root / "registry" / "submissions.json"
        self.cli("register", "--submission", output, "--score", "0.5", "--reference", "synthetic-test", "--registry", registry)
        self.cli("register", "--submission", output, "--score", "0.5", "--reference", "synthetic-test", "--registry", registry, success=False)
        self.assertEqual(len(read_json(registry)), 1)


if __name__ == "__main__":
    unittest.main()
