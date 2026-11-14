"""Run built student TODOs and their real checks before downstream consumers.

Use the pinned lab environment, after rebuilding the notebooks:
    build/qa_pinned_env/Scripts/python.exe tools/test_student_guards.py

Each case executes the student cell with its unchanged stub on tiny fixtures.
It must print corrective feedback and stop before plotting, training or reporting
unsupported results. The corresponding built solution must pass the same check
and reach the intercepted consumer; an unconditional stop cannot pass this test.
Only external inputs and expensive downstream consumers are replaced. No model
download, dataset download, training run, widget or hosted request is performed.
This covers four W03 TODOs, W08 TODOs 3-4 and all five W09 TODOs, not every student lab or input.
"""
import ast
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def code_cells(week, solution=False):
    suffix = "_Solutions" if solution else ""
    paths = list((ROOT / "deliverables").glob(
        f"Week_{week:02d}_*/Week_{week:02d}_Lab{suffix}.ipynb"))
    if len(paths) != 1:
        raise AssertionError(f"Rebuild Week {week:02d}: expected one built notebook, found {len(paths)}")
    notebook = json.loads(paths[0].read_text(encoding="utf-8"))
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in notebook["cells"] if c["cell_type"] == "code"]


def function_cell(week, function, solution=False, start_at=None):
    cells = [src for src in code_cells(week, solution) if f"def {function}(" in src]
    if len(cells) != 1:
        raise AssertionError(f"Expected one built cell defining {function}, found {len(cells)}")
    tree = ast.parse(cells[0])
    if start_at:
        # Dataset/model preparation before these definitions belongs to an
        # external input boundary. Keep the actual functions, checks, guards
        # and every subsequent statement; supply small local inputs instead.
        first = next(i for i, n in enumerate(tree.body)
                     if isinstance(n, ast.FunctionDef) and n.name == start_at)
        tree.body = tree.body[first:]
    return compile(tree, f"built_W{week:02d}_{function}", "exec")


def real_check(week):
    cells = [src for src in code_cells(week) if "def check(condition, ok_message, hint):" in src]
    if len(cells) != 1:
        raise AssertionError("Expected exactly one injected setup check()")
    node = next(n for n in ast.parse(cells[0]).body
                if isinstance(n, ast.FunctionDef) and n.name == "check")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "built_shared_check", "exec"), namespace)
    return namespace["check"]


class ConsumerReached(Exception):
    """A passing check reached the real downstream call, now safely intercepted."""


def stop_at_consumer(name):
    def intercepted(*args, **kwargs):
        raise ConsumerReached(name)
    return intercepted


class PlotBoundary:
    def __getattr__(self, name):
        return stop_at_consumer("plot." + name)


class FeatureInputs(dict):
    def to(self, device):
        return self


def embedding_inputs(**kwargs):
    if "images" in kwargs:
        return FeatureInputs(pixel_values=torch.tensor([[3., 4.]]).repeat(len(kwargs["images"]), 1))
    rows = torch.tensor([[6., 8.]]).repeat(len(kwargs["text"]), 1)
    return FeatureInputs(input_ids=rows, attention_mask=torch.ones_like(rows))


def tiny_class_texts(texts):
    return torch.tensor([[1., 0.] if "cat" in text else [0., 1.] for text in texts])


class StudentGuardTests(unittest.TestCase):
    def namespace(self, week):
        return {"torch": torch, "nn": nn, "F": F, "np": np,
                "DEVICE": "cpu", "SEED": 42, "SMOKE": True,
                "check": real_check(week), "plt": PlotBoundary()}

    def exercise(self, week, function, fixture, todo, *, start_at=None,
                 next_cell=None, modules=None, forbid_report=None):
        for solution in (False, True):
            with self.subTest(week=week, function=function, notebook="solution" if solution else "student"):
                namespace = self.namespace(week)
                namespace.update(fixture())
                code = function_cell(week, function, solution, start_at)
                output = io.StringIO()
                error = None
                try:
                    module_boundary = patch.dict("sys.modules", modules) if modules else nullcontext()
                    with redirect_stdout(output), module_boundary:
                        exec(code, namespace)
                        if next_cell:
                            consumers = [src for src in code_cells(week, solution) if next_cell in src]
                            self.assertEqual(len(consumers), 1, "Expected one actual downstream consumer cell")
                            exec(compile(consumers[0], "built_following_consumer", "exec"), namespace)
                except (AssertionError, ValueError, ConsumerReached) as exc:
                    error = exc
                if solution:
                    self.assertIsInstance(error, ConsumerReached,
                                          "Correct built solution must reach the intercepted consumer")
                    self.assertNotIn("Try again:", output.getvalue())
                else:
                    self.assertIsInstance(error, (AssertionError, ValueError),
                                          "Student stub must stop helpfully before the dependent consumer")
                    self.assertIn(f"TODO {todo}", str(error))
                    self.assertIn("rerun", str(error).lower())
                    self.assertIn("Try again:", output.getvalue())
                    if forbid_report:
                        self.assertNotIn(forbid_report, output.getvalue(),
                                         "Failed TODO must not print an unsupported measurement")

    def test_w03_slerp_stops_before_generator_and_plot(self):
        self.exercise(3, "slerp", lambda: {
            "G": SimpleNamespace(eval=stop_at_consumer("generator.eval")), "Z_DIM": 2,
        }, 3)

    def test_w03_predictions_stop_before_coverage_histogram(self):
        self.exercise(3, "check_predictions", lambda: {
            "EVAL_N": 4, "Z_DIM": 2, "G": lambda x: x,
            "clf": lambda x: F.one_hot(torch.arange(4), num_classes=10).float(),
        }, 4, forbid_report="classes with at least")

    def test_w03_feature_distance_stops_before_evaluation_inputs(self):
        self.exercise(3, "frechet_distance", lambda: {
            "DataLoader": stop_at_consumer("evaluation.DataLoader"), "full_train": object(), "EVAL_N": 4,
        }, 5, forbid_report="Fréchet distance =")

    def test_w03_gradient_penalty_stops_before_toy_training_plot(self):
        self.exercise(3, "gradient_penalty", lambda: {"TOY_STEPS": 1}, 6,
                      forbid_report="toy training steps:")

    def test_w09_embeddings_stop_before_cosine_plot(self):
        def fixture():
            return {
                "images": [object() for _ in range(6)], "labels": np.arange(6),
                "CLASSES": ["automobile", "bird", "cat", "ship", "horse", "airplane"],
                "proc": embedding_inputs,
                "clip": SimpleNamespace(
                    visual_projection=lambda x: x, text_projection=lambda x: x,
                    vision_model=lambda **x: SimpleNamespace(pooler_output=x["pixel_values"]),
                    text_model=lambda **x: SimpleNamespace(pooler_output=x["input_ids"])),
                "time": SimpleNamespace(perf_counter=lambda: 0.0),
            }
        self.exercise(9, "embed_images", fixture, 1, start_at="embed_images")

    def test_w09_zero_shot_stops_before_accuracy_report_and_confusion_matrix(self):
        metrics = SimpleNamespace(confusion_matrix=stop_at_consumer("confusion_matrix"))
        self.exercise(9, "zero_shot_accuracy", lambda: {
            "CLASSES": ["cat", "dog"], "labels": np.array([0, 1]),
            "img_emb": torch.eye(2), "embed_texts": tiny_class_texts,
        }, 2, next_cell="cm = confusion_matrix(labels, last_pred",
                      modules={"sklearn.metrics": metrics}, forbid_report="accuracy =")

    def test_w09_precision_stops_before_gallery_retrieval(self):
        self.exercise(9, "precision_at_k", lambda: {
            "CLASSES": ["cat", "dog"], "embed_texts": stop_at_consumer("gallery.embed_texts"),
        }, 3, forbid_report="precision@1 =")

    def test_w09_contrastive_loss_stops_before_gallery_measurement(self):
        self.exercise(9, "clip_loss", lambda: {
            "images": [object()], "show_idx": [0],
            "embed_images": stop_at_consumer("gallery.embed_images"),
        }, 4, forbid_report="loss, correct pairs:")

    def test_w09_wer_stops_before_audio_transcription(self):
        self.exercise(9, "wer", lambda: {
            "time": SimpleNamespace(perf_counter=lambda: 0.0),
            "speech": SimpleNamespace(select=stop_at_consumer("audio.select")),
        }, 5, start_at="load_audio", forbid_report="mean WER:")


def built_cell(week, marker, solution):
    found = [src for src in code_cells(week, solution) if marker in src]
    if len(found) != 1:
        raise AssertionError(f"Expected one built W{week:02d} cell containing {marker!r}, found {len(found)}")
    return found[0]


def w08_fixture(week=8):
    """Invented tiny stand-ins for the trained adapter, data splits and evaluator."""
    import hashlib
    import pandas as pd

    model = nn.Linear(10, 10, bias=False)

    def save_pretrained(directory):
        path = Path(directory)
        path.mkdir(exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(b"0" * 400000)  # 4 bytes x 100,000 trainable

    model.save_pretrained = save_pretrained
    gold, test = ["intent_a", "intent_b"], ["INVENTED TEST A", "INVENTED TEST B"]
    dev = pd.DataFrame({"text": ["INVENTED VALIDATION"], "gold": ["intent_a"], "raw": ["intent_b"],
                        "pred": ["intent_b"], "correct": [False]})

    def evaluate(*args, **kwargs):
        return pd.DataFrame({"text": test, "gold": gold, "raw": gold, "pred": gold, "correct": [True, True]})

    def template(example):
        return [{"role": "user", "content": example["text"]}]

    return {"torch": torch, "pd": pd, "json": json, "hashlib": hashlib, "Path": Path, "check": real_check(week),
            "trainer": SimpleNamespace(model=model, state=SimpleNamespace(global_step=1)),
            "args": SimpleNamespace(to_dict=lambda: {"seed": 42}, learning_rate=0.01, seed=42),
            "train_ds": [{"text": "INVENTED TRAINING", "label_text": "intent_a"}], "test_ds": {"text": test},
            "MODEL_ID": "invented-tiny-fixture", "INTENTS": gold, "RANK": 1, "zero": template, "few": template,
            "LABEL_RE": {}, "evaluate": evaluate, "display": lambda x: None,
            "base": SimpleNamespace(parameters=lambda: [torch.zeros(1000000)]),
            "df_zero_dev": dev, "df_few_dev": dev.copy(), "trainable": 100000, "total": 1100000}


class Week08GuardTests(unittest.TestCase):
    """W08 TODO 3/4 checks must stop unsupported comparisons and sizes (Q3-01, Q3-02)."""

    def run_cell(self, code, namespace):
        output, error = io.StringIO(), None
        try:
            with redirect_stdout(output):
                exec(compile(code, "built_w08_cell", "exec"), namespace)
        except AssertionError as exc:
            error = exc
        return output.getvalue(), error

    def test_w08_comparison_stops_before_error_analysis(self):
        for solution in (False, True):
            with self.subTest(notebook="solution" if solution else "student"):
                namespace = w08_fixture()
                out, error = self.run_cell(built_cell(8, "def check_comparison(", solution), namespace)
                if solution:
                    self.assertIsNone(error)
                    self.assertNotIn("Try again:", out)
                    self.assertEqual(list(namespace["comparison"]["model"]),
                                     ["base, zero-shot", "base, few-shot", "LoRA fine-tuned"])
                else:
                    self.assertIsInstance(error, AssertionError, "Unfinished TODO 3 must stop this cell")
                    self.assertIn("TODO 3", str(error))
                    self.assertIn("rerun", str(error).lower())
                    self.assertIn("Try again:", out)

    def test_w08_sizes_stop_before_printing_or_reporting(self):
        import tempfile
        import os
        for solution in (False, True):
            with self.subTest(notebook="solution" if solution else "student"), \
                    tempfile.TemporaryDirectory() as folder:
                namespace = w08_fixture()
                # A completed TODO 3 isolates TODO 4 from the earlier guard.
                with redirect_stdout(io.StringIO()):
                    exec(compile(built_cell(8, "def check_comparison(", True), "completed_todo3", "exec"), namespace)
                namespace["tuned"] = namespace["trainer"].model
                old = Path.cwd()
                os.chdir(folder)
                try:
                    out, error = self.run_cell(built_cell(8, 'tuned.save_pretrained("banking-lora-adapter")', solution),
                                               namespace)
                finally:
                    os.chdir(old)
                if solution:
                    self.assertIsNone(error)
                    self.assertIn("adapter: 0.4 MB", out)
                else:
                    self.assertIsInstance(error, AssertionError, "Unfinished TODO 4 must stop before reporting sizes")
                    self.assertIn("TODO 4", str(error))
                    self.assertIn("rerun", str(error).lower())
                    self.assertIn("Try again:", out)
                    self.assertNotIn("adapter:", out, "Zero placeholder sizes must not be printed as measurements")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
