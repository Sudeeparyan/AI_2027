"""Regression checks for full-run provenance and safe figure harvesting.

Run: .venv/Scripts/python.exe tools/test_lab_evidence.py
Uses isolated temporary notebooks; no downloads or curriculum asset changes.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nbformat
from PIL import Image

import compare_lab_code
import harvest_figures


class LabEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "build/week_09"
        self.run_dir.mkdir(parents=True)
        self.built = self.root / "deliverables/Week_09_Test/Week_09_Lab_Solutions.ipynb"
        self.built.parent.mkdir(parents=True)
        image = BytesIO()
        Image.new("RGB", (1, 1), "white").save(image, format="PNG")
        self.png = image.getvalue()
        self.nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("x = 1 # FIGURE_A", execution_count=1, outputs=[
                nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(self.png).decode()})]),
            nbformat.v4.new_code_cell("y = x + 1 # FIGURE_B", execution_count=2, outputs=[
                nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(self.png).decode()})]),
        ])
        self.full = self.run_dir / "executed_solutions_full.ipynb"
        nbformat.write(self.nb, self.full)
        nbformat.write(self.nb, self.built)
        self.record = {"week": 9, "mode": "full", "ok": True}
        self.save_record()
        self.assets = self.root / "curriculum/assets/week_09"
        self.assets.mkdir(parents=True)
        (self.assets / "a.png").write_bytes(b"existing asset")
        self.addCleanup(patch.stopall)
        patch.object(harvest_figures, "MAPS", {9: {"a": ("FIGURE_A", 0), "b": ("FIGURE_B", 0)}}).start()

    def save_record(self):
        (self.run_dir / "lab_test_full.json").write_text(json.dumps(self.record), encoding="utf-8")

    def harvest(self, dry_run=False):
        with redirect_stdout(StringIO()):
            return harvest_figures.harvest(9, root=self.root, dry_run=dry_run)

    def assert_assets_unchanged(self):
        self.assertEqual((self.assets / "a.png").read_bytes(), b"existing asset")
        self.assertFalse((self.assets / "b.png").exists())

    def test_later_smoke_is_never_the_harvest_source(self):
        smoke = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise RuntimeError('smoke')")])
        nbformat.write(smoke, self.run_dir / "executed_solutions.ipynb")
        self.assertEqual(self.harvest(), [])
        self.assertEqual((self.assets / "a.png").read_bytes(), self.png)
        self.assertEqual((self.assets / "b.png").read_bytes(), self.png)

    def test_missing_figure_leaves_all_existing_assets_unchanged(self):
        self.nb.cells[1].outputs = []
        nbformat.write(self.nb, self.full)
        self.assertEqual(self.harvest(), ["b"])
        self.assert_assets_unchanged()

    def test_failed_full_record_is_rejected(self):
        self.record["ok"] = False
        self.save_record()
        with self.assertRaisesRegex(ValueError, "successful full-run"):
            self.harvest()
        self.assert_assets_unchanged()

    def test_missing_full_record_is_rejected(self):
        (self.run_dir / "lab_test_full.json").unlink()
        with self.assertRaisesRegex(ValueError, "--full first"):
            self.harvest()
        self.assert_assets_unchanged()

    def test_incomplete_full_notebook_is_rejected(self):
        self.nb.cells[1].execution_count = None
        nbformat.write(self.nb, self.full)
        with self.assertRaisesRegex(ValueError, "unexecuted or failed"):
            self.harvest()
        self.assert_assets_unchanged()

    def test_error_output_is_rejected_even_with_success_record(self):
        self.nb.cells[1].outputs = [nbformat.v4.new_output("error", ename="ValueError", evalue="bad", traceback=[])]
        nbformat.write(self.nb, self.full)
        with self.assertRaisesRegex(ValueError, "unexecuted or failed"):
            self.harvest()
        self.assert_assets_unchanged()

    def test_executed_notebook_hash_detects_tampering(self):
        self.record["executed_notebook_sha256"] = hashlib.sha256(self.full.read_bytes()).hexdigest()
        self.save_record()
        self.nb.cells[0].outputs = []
        nbformat.write(self.nb, self.full)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.harvest()
        self.assert_assets_unchanged()

    def test_cell_reordering_and_duplicates_are_detected(self):
        original = compare_lab_code.code_cells(self.built)
        self.nb.cells.reverse()
        nbformat.write(self.nb, self.built)
        self.assertNotEqual(original, compare_lab_code.code_cells(self.built))
        with self.assertRaisesRegex(ValueError, "executed code differs"):
            self.harvest()
        self.nb.cells.reverse()
        self.nb.cells.append(nbformat.v4.new_code_cell(self.nb.cells[0].source))
        nbformat.write(self.nb, self.built)
        self.assertNotEqual(original, compare_lab_code.code_cells(self.built))

    def test_comment_edits_are_allowed_but_string_edits_are_detected(self):
        self.assertEqual(compare_lab_code.normalise_code("x=1 # comment"), compare_lab_code.normalise_code("# new comment\nx = 1"))
        a = "message = '''first\n# original literal text\nlast'''"
        b = "message = '''first\n# changed literal text\nlast'''"
        self.assertNotEqual(compare_lab_code.normalise_code(a), compare_lab_code.normalise_code(b))

    def test_runner_tags_are_skipped_consistently(self):
        original = compare_lab_code.code_cells(self.built)
        self.nb.cells.append(nbformat.v4.new_code_cell("optional_setup()", metadata={"tags": ["skip-test"]}))
        nbformat.write(self.nb, self.built)
        self.assertEqual(original, compare_lab_code.code_cells(self.built))

    def test_dry_run_never_changes_assets(self):
        self.assertEqual(self.harvest(dry_run=True), [])
        self.assert_assets_unchanged()

    def compare(self):
        with patch.object(compare_lab_code, "ROOT", self.root), patch("sys.argv", ["compare_lab_code.py", "--weeks", "9"]), redirect_stdout(StringIO()):
            return compare_lab_code.main()

    def test_code_audit_fails_when_full_evidence_is_missing(self):
        (self.run_dir / "lab_test_full.json").unlink()
        self.assertEqual(self.compare(), 1)

    def test_code_audit_fails_when_full_execution_failed(self):
        self.record["ok"] = False
        self.save_record()
        self.assertEqual(self.compare(), 1)

    def test_code_audit_detects_execution_order_change(self):
        self.nb.cells.reverse()
        nbformat.write(self.nb, self.built)
        self.assertEqual(self.compare(), 1)

    def test_code_audit_checks_the_recorded_executed_hash(self):
        self.record["executed_notebook_sha256"] = "incorrect"
        self.save_record()
        self.assertEqual(self.compare(), 1)


class OptionalCloudTests(unittest.TestCase):
    """Exercise the actual W01/W06/W11 shared-provider comparison cells.

    Execute only the isolated provider dispatcher/comparison cells with stubs.
    Do not execute setup imports, load a real .env file or instantiate an SDK.
    """

    def run_shared_cell(self, week, *, gemini=False, openai=False, smoke=False,
                        provider_reply="stubbed hosted reply"):
        root = Path(__file__).resolve().parents[1]
        setup = ast.parse((root / "curriculum/labs/_setup.py").read_text(encoding="utf-8"))
        for switch in ("USE_GEMINI", "USE_OPENAI"):
            assignment = next(n for n in setup.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == switch for t in n.targets))
            self.assertIs(assignment.value.value, False, f"{switch} must default to False")
        dispatcher = next(n for n in setup.body if isinstance(n, ast.FunctionDef) and n.name == "ask_model")
        source = (root / f"curriculum/labs/week_{week:02d}_lab.py").read_text(encoding="utf-8")
        candidates = []
        for cell in source.split("# %%"):
            if cell.startswith(" [markdown]") or "ask_model(" not in cell:
                continue
            tree = ast.parse('\n'.join(line for line in cell.splitlines() if not line.startswith('%')))
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ask_model"
                   for n in ast.walk(tree)):
                candidates.append(tree)
        self.assertEqual(len(candidates), 1, f"W{week:02d} must use one shared optional-comparison cell")
        # The available saved keys are deliberately fake. They must never
        # select a provider, and the dispatcher must not inspect a .env file.
        gemini_stub = Mock(return_value=provider_reply)
        openai_stub = Mock(return_value=provider_reply)
        namespace = {"SMOKE": smoke, "USE_GEMINI": gemini, "USE_OPENAI": openai,
                     "ask_gemini": gemini_stub, "ask_openai": openai_stub,
                     "os": SimpleNamespace(environ={"GEMINI_API_KEY": "test-only-key",
                                                    "OPENAI_API_KEY": "test-only-key"}),
                     "fictional_prompt": "Explain an invented paper; do not invent a citation.",
                     "honest_answer": "local honesty-instruction reply", "MODEL_ID": "local-checkpoint",
                     "GEMINI_MODEL": "stub-gemini-model", "OPENAI_MODEL": "stub-openai-model",
                     "api_answer": "local endpoint reply", "DECODER_ID": "local-decoder",
                     "CORE_NEW_TOKENS": 24, "dec": object(), "dec_tok": object(),
                     "chat": Mock(return_value=("local decoder reply", 2, 0.1)),
                     "time": SimpleNamespace(perf_counter=lambda: 1.0),
                     "pd": SimpleNamespace(DataFrame=lambda rows: rows)}
        output = StringIO()
        with redirect_stdout(output):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[dispatcher], type_ignores=[])),
                         "shared_ask_model", "exec"), namespace)
            exec(compile(ast.fix_missing_locations(candidates[0]), f"week_{week:02d}_shared_comparison", "exec"), namespace)
        self.assertNotIn("test-only-key", output.getvalue())
        namespace['_stdout'] = output.getvalue()
        return gemini_stub, openai_stub, namespace

    def run_shared_week1_cell(self, **options):
        return self.run_shared_cell(1, **options)

    def assert_local_comparison(self, namespace):
        self.assertIsNone(namespace["hosted_answer"])
        self.assertEqual(namespace["comparison_answer"], namespace["honest_answer"])
        self.assertEqual(namespace["comparison_model"], namespace["MODEL_ID"])

    def assert_other_comparison(self, week, namespace, hosted_reply=None):
        if week == 6:
            namespace['chat'].assert_called_once_with(namespace['dec'], namespace['dec_tok'],
                                                     namespace['question'], max_new_tokens=24)
            rows = namespace['log']
            self.assertEqual(rows[0]['route'], 'local Transformers')
            self.assertEqual(rows[0]['answer'], 'local decoder reply')
            self.assertIn(namespace['local_rubric'], namespace['_stdout'])
            self.assertEqual(len(rows), 1 if hosted_reply is None else 2)
            self.assertEqual(namespace['hosted'], hosted_reply)
            if hosted_reply is not None:
                self.assertEqual(rows[1]['route'], 'selected shared provider')
                self.assertEqual(rows[1]['answer'], hosted_reply)
                self.assertIsNone(rows[1]['prompt_tokens'])
                self.assertIsNone(rows[1]['completion_tokens'])
        else:
            self.assertEqual(week, 11)
            self.assertEqual(namespace['hosted_answer'], hosted_reply)
            self.assertIn(hosted_reply if hosted_reply is not None else namespace['api_answer'], namespace['_stdout'])
            self.assertIn('Rubric: endpoint handling; schema validation', namespace['_stdout'])

    def test_saved_credentials_do_not_enable_cloud_requests(self):
        gemini, openai, namespace = self.run_shared_cell(11)
        gemini.assert_not_called()
        openai.assert_not_called()
        self.assert_other_comparison(11, namespace)

    def test_smoke_never_calls_cloud_even_when_opted_in(self):
        for week in (6, 11):
            for flags in ({"gemini": True}, {"openai": True}, {"gemini": True, "openai": True}):
                with self.subTest(week=week, flags=flags):
                    gemini, openai, namespace = self.run_shared_cell(week, smoke=True, **flags)
                    gemini.assert_not_called()
                    openai.assert_not_called()
                    self.assert_other_comparison(week, namespace)

    def test_explicit_opt_in_reaches_only_the_stubbed_provider(self):
        for week in (6, 11):
            for name in ('gemini', 'openai'):
                with self.subTest(week=week, provider=name):
                    gemini, openai, namespace = self.run_shared_cell(week, **{name: True})
                    selected, disabled = (gemini, openai) if name == 'gemini' else (openai, gemini)
                    prompt = (namespace['question'] if week == 6
                              else 'Name two uses of FastAPI for an invented classroom app.')
                    selected.assert_called_once_with(prompt, temperature=0.2, max_output_tokens=128 if week == 6 else 96)
                    disabled.assert_not_called()
                    self.assert_other_comparison(week, namespace, 'stubbed hosted reply')

    def check_other_week(self, week, smoke=False):
        gemini, openai, namespace = self.run_shared_cell(week, smoke=smoke, gemini=smoke)
        gemini.assert_not_called()
        openai.assert_not_called()
        self.assert_other_comparison(week, namespace)

    def test_upgraded_weeks_two_selected_providers_keep_local_fallback(self):
        for week in (6, 11):
            with self.subTest(week=week):
                gemini, openai, namespace = self.run_shared_cell(week, gemini=True, openai=True)
                gemini.assert_not_called()
                openai.assert_not_called()
                self.assert_other_comparison(week, namespace)

    def test_upgraded_weeks_unavailable_provider_keeps_local_fallback(self):
        for week in (6, 11):
            for name in ('gemini', 'openai'):
                with self.subTest(week=week, provider=name):
                    gemini, openai, namespace = self.run_shared_cell(week, provider_reply=None, **{name: True})
                    selected, disabled = (gemini, openai) if name == 'gemini' else (openai, gemini)
                    selected.assert_called_once()
                    disabled.assert_not_called()
                    self.assert_other_comparison(week, namespace)

    def test_week1_credentials_alone_do_not_enable_requests(self):
        gemini, openai, namespace = self.run_shared_week1_cell()
        gemini.assert_not_called()
        openai.assert_not_called()
        self.assert_local_comparison(namespace)

    def test_week1_smoke_ignores_explicit_opt_in(self):
        for flags in ({"gemini": True}, {"openai": True}, {"gemini": True, "openai": True}):
            with self.subTest(flags=flags):
                gemini, openai, namespace = self.run_shared_week1_cell(smoke=True, **flags)
                gemini.assert_not_called()
                openai.assert_not_called()
                self.assert_local_comparison(namespace)

    def test_week1_each_opt_in_routes_settings_to_only_its_stubbed_provider(self):
        for name in ("gemini", "openai"):
            with self.subTest(provider=name):
                gemini, openai, namespace = self.run_shared_week1_cell(**{name: True})
                selected, disabled = (gemini, openai) if name == "gemini" else (openai, gemini)
                selected.assert_called_once_with(namespace["fictional_prompt"], temperature=0.7, max_output_tokens=128)
                disabled.assert_not_called()
                self.assertEqual(namespace["comparison_answer"], "stubbed hosted reply")
                self.assertEqual(namespace["comparison_model"], namespace[f"{name.upper()}_MODEL"])

    def test_week1_two_selected_providers_fall_back_without_requests(self):
        gemini, openai, namespace = self.run_shared_week1_cell(gemini=True, openai=True)
        gemini.assert_not_called()
        openai.assert_not_called()
        self.assert_local_comparison(namespace)

    def test_week1_unavailable_provider_keeps_the_complete_local_response(self):
        for name in ("gemini", "openai"):
            with self.subTest(provider=name):
                gemini, openai, namespace = self.run_shared_week1_cell(provider_reply=None, **{name: True})
                selected, disabled = (gemini, openai) if name == "gemini" else (openai, gemini)
                selected.assert_called_once()
                disabled.assert_not_called()
                self.assert_local_comparison(namespace)

    def test_week6_credentials_alone_do_not_enable_requests(self):
        self.check_other_week(6)

    def test_week6_smoke_keeps_the_local_comparison(self):
        self.check_other_week(6, smoke=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
