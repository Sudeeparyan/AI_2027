"""Shared setup contract tests without network requests or model downloads."""
import ast
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class SharedSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        tree = ast.parse((ROOT / "curriculum/labs/_setup.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.secret = Mock(return_value="test-secret-never-print")
        self.generate = Mock(return_value=SimpleNamespace(text="Synthetic answer"))
        self.client = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate), close=Mock())
        self.sdk = Mock(return_value=self.client)
        self.openai_generate = Mock(return_value=SimpleNamespace(output_text="OpenAI synthetic answer"))
        self.openai_client = SimpleNamespace(responses=SimpleNamespace(create=self.openai_generate), close=Mock())
        self.openai_sdk = Mock(return_value=self.openai_client)
        self.dotenv = Mock(return_value={})
        self.sleep = Mock()
        self.ns = {"Path": Path, "hashlib": hashlib, "json": json,
                   "os": SimpleNamespace(environ={"GEMINI_API_KEY": "test-secret-never-print"}),
                   "time": SimpleNamespace(sleep=self.sleep), "USE_GEMINI": False,
                   "SMOKE": False, "IN_COLAB": False, "GEMINI_MODEL": "gemini-3.5-flash-lite",
                   "USE_OPENAI": False, "OPENAI_MODEL": "gpt-4.1-mini", "_openai_calls": 0,
                   "LOCAL_ENV_FILE": str(Path(self.temp.name) / ".env"),
                   "_gemini_calls": 0, "_gemini_cache": Path(self.temp.name)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "shared-functions", "exec"), self.ns)
        modules = {
            "google": SimpleNamespace(genai=SimpleNamespace(Client=self.sdk)),
            "google.genai": SimpleNamespace(types=SimpleNamespace(
                GenerateContentConfig=lambda **kw: kw, HttpOptions=lambda **kw: kw,
                HttpRetryOptions=lambda **kw: kw)),
            "google.colab": SimpleNamespace(userdata=SimpleNamespace(get=self.secret)),
            "openai": SimpleNamespace(OpenAI=self.openai_sdk),
            "dotenv": SimpleNamespace(dotenv_values=self.dotenv),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def ask(self):
        output = StringIO()
        with redirect_stdout(output):
            result = self.ns["ask_gemini"]("Invent a fictional two-word example.")
        self.assertNotIn("test-secret-never-print", output.getvalue())
        return result

    def test_off_never_reads_secret_or_constructs_client(self):
        self.ns["IN_COLAB"] = True
        self.assertIsNone(self.ask())
        self.secret.assert_not_called()
        self.sdk.assert_not_called()
        self.dotenv.assert_not_called()

    def test_smoke_disables_opted_in_api(self):
        self.ns.update(USE_GEMINI=True, SMOKE=True, IN_COLAB=True)
        self.assertIsNone(self.ask())
        self.secret.assert_not_called()
        self.sdk.assert_not_called()

    def test_missing_key_falls_back(self):
        self.ns.update(USE_GEMINI=True, os=SimpleNamespace(environ={}))
        self.assertIsNone(self.ask())
        self.sdk.assert_not_called()

    def test_429_retries_then_falls_back(self):
        error = Exception("private request details")
        error.code = 429
        self.ns["USE_GEMINI"] = True
        self.generate.side_effect = error
        self.assertIsNone(self.ask())
        self.assertEqual(self.generate.call_count, 4)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2, 4])
        self.client.close.assert_called_once()

    def test_429_recovery_and_disk_cache(self):
        error = Exception()
        error.code = 429
        self.ns["USE_GEMINI"] = True
        self.generate.side_effect = [error, SimpleNamespace(text="Recovered synthetic answer")]
        self.assertEqual(self.ask(), "Recovered synthetic answer")
        self.assertEqual(self.ask(), "Recovered synthetic answer")
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.sdk.call_count, 1)
        saved = next(Path(self.temp.name).glob("*.json")).read_text(encoding="utf-8")
        self.assertNotIn("test-secret-never-print", saved)

    def test_other_error_does_not_retry(self):
        self.ns["USE_GEMINI"] = True
        self.generate.side_effect = Exception("private request details")
        self.assertIsNone(self.ask())
        self.sleep.assert_not_called()

    def test_request_ceiling_includes_retries(self):
        self.ns.update(USE_GEMINI=True, _gemini_calls=40)
        self.assertIsNone(self.ask())
        self.generate.assert_not_called()

    def test_colab_secret_precedes_environment_without_exposure(self):
        self.ns.update(USE_GEMINI=True, IN_COLAB=True)
        self.assertEqual(self.ask(), "Synthetic answer")
        self.secret.assert_called_once_with("GEMINI_API_KEY")

    def test_check_feedback(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertTrue(self.ns["check"](True, "Rows sum to one", "Normalise rows"))
            self.assertFalse(self.ns["check"](False, "Rows sum to one", "Normalise rows"))
        self.assertIn("✅ Rows sum to one", output.getvalue())
        self.assertIn("Try again: Normalise rows", output.getvalue())

    def test_install_pins_preserve_names_and_extras(self):
        from build_week import install_cell
        text = install_cell("%pip install -q torch torchvision transformers>=4.5 peft[extra]", first=True)
        self.assertIn("torch==2.11.0 torchvision==0.26.0 transformers==5.18.0 peft[extra]==0.21.1", text)
        self.assertIn("google-genai==2.29.0 ipywidgets==7.7.1", text)

    def test_install_pins_preserve_flags_urls_and_comments(self):
        from build_week import install_cell
        original = '%pip install -q transformers>=4.5,<5 --index-url https://example.org/torch/simple --no-binary torch # peft'
        text = install_cell(original, first=False)
        self.assertEqual(text, original.replace("transformers>=4.5,<5", "transformers==5.18.0"))
        self.assertEqual(install_cell('url = "https://example.org/torch/simple"', first=False),
                         'url = "https://example.org/torch/simple"')

    def test_sdk_has_one_attempt_and_bounded_timeout(self):
        self.ns["USE_GEMINI"] = True
        self.assertEqual(self.ask(), "Synthetic answer")
        self.assertEqual(self.sdk.call_args.kwargs["http_options"],
                         {"timeout": 30000, "retry_options": {"attempts": 1}})

    def ask_selected(self):
        output = StringIO()
        with redirect_stdout(output):
            result = self.ns["ask_model"]("Invent a fictional two-word example.")
        self.assertNotIn("test-secret-never-print", output.getvalue())
        self.assertNotIn("private request details", output.getvalue())
        return result

    def enable_openai(self):
        self.ns.update(USE_OPENAI=True, os=SimpleNamespace(environ={"OPENAI_API_KEY": "test-secret-never-print"}))

    def test_dispatch_off_and_conflicting_switches_read_no_credentials(self):
        self.assertIsNone(self.ask_selected())
        self.ns.update(USE_GEMINI=True, USE_OPENAI=True, IN_COLAB=True)
        self.assertIsNone(self.ask_selected())
        self.secret.assert_not_called()
        self.dotenv.assert_not_called()
        self.sdk.assert_not_called()
        self.openai_sdk.assert_not_called()

    def test_dispatch_gemini_selects_only_gemini(self):
        self.ns["USE_GEMINI"] = True
        self.assertEqual(self.ask_selected(), "Synthetic answer")
        self.openai_sdk.assert_not_called()

    def test_openai_responses_contract_and_cached_answer(self):
        self.enable_openai()
        self.assertEqual(self.ask_selected(), "OpenAI synthetic answer")
        self.assertEqual(self.ask_selected(), "OpenAI synthetic answer")
        self.openai_generate.assert_called_once_with(model="gpt-4.1-mini",
            input="Invent a fictional two-word example.", temperature=0.2,
            max_output_tokens=256, store=False)
        self.assertEqual(self.openai_sdk.call_args.kwargs,
                         {"api_key": "test-secret-never-print", "timeout": 30.0, "max_retries": 0})
        self.openai_client.close.assert_called_once()
        self.sdk.assert_not_called()
        self.dotenv.assert_not_called()
        for file in Path(self.temp.name).glob("*.json"):
            self.assertNotIn("test-secret-never-print", file.read_text(encoding="utf-8"))

    def test_openai_429_retries_are_bounded_and_private(self):
        self.enable_openai()
        error = Exception("private request details")
        error.status_code = 429
        self.openai_generate.side_effect = error
        self.assertIsNone(self.ask_selected())
        self.assertEqual(self.openai_generate.call_count, 4)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2, 4])

    def test_both_providers_share_attempt_ceiling(self):
        self.enable_openai()
        self.ns.update(_gemini_calls=39, _openai_calls=1)
        self.assertIsNone(self.ask_selected())
        self.openai_generate.assert_not_called()
        self.ns.update(USE_GEMINI=True, USE_OPENAI=False)
        self.ns["os"] = SimpleNamespace(environ={"GEMINI_API_KEY": "test-secret-never-print"})
        self.assertIsNone(self.ask_selected())
        self.generate.assert_not_called()

    def test_openai_missing_key_blank_and_other_errors_fall_back(self):
        self.ns.update(USE_OPENAI=True, os=SimpleNamespace(environ={}))
        self.assertIsNone(self.ask_selected())
        self.openai_sdk.assert_not_called()
        self.enable_openai()
        self.openai_generate.return_value = SimpleNamespace(output_text="  ")
        self.assertIsNone(self.ask_selected())
        self.openai_generate.side_effect = Exception("private request details")
        self.assertIsNone(self.ask_selected())
        self.sleep.assert_not_called()

    def test_local_dotenv_is_read_only_after_opt_in(self):
        self.dotenv.return_value = {"OPENAI_API_KEY": "test-secret-never-print"}
        self.ns["os"] = SimpleNamespace(environ={})
        self.assertIsNone(self.ask_selected())
        self.dotenv.assert_not_called()
        self.ns["USE_OPENAI"] = True
        self.assertEqual(self.ask_selected(), "OpenAI synthetic answer")
        self.dotenv.assert_called_once_with(self.ns["LOCAL_ENV_FILE"])
        self.assertEqual(self.ns["os"].environ, {})

    def test_openai_colab_secret_and_smoke_do_not_read_local_dotenv(self):
        self.enable_openai()
        self.ns.update(IN_COLAB=True, SMOKE=True)
        self.assertIsNone(self.ask_selected())
        self.secret.assert_not_called()
        self.openai_sdk.assert_not_called()
        self.ns["SMOKE"] = False
        self.assertEqual(self.ask_selected(), "OpenAI synthetic answer")
        self.secret.assert_called_once_with("OPENAI_API_KEY")
        self.dotenv.assert_not_called()

    def test_runner_restores_cpu_environment_before_gpu_run(self):
        from test_labs import runtime_environment
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "GENAI_LAB_SMOKE": "old",
                                     "MPLBACKEND": "Agg"}, clear=True):
            before = dict(os.environ)
            with runtime_environment(False, "cpu"):
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(os.environ["GENAI_FORCE_CPU"], "1")
            self.assertEqual(dict(os.environ), before)
            with runtime_environment(True, "cuda"):
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(os.environ["GENAI_RUNNER_GPU"], "1")
                self.assertNotIn("GENAI_LAB_SMOKE", os.environ)
            self.assertEqual(dict(os.environ), before)

    def test_runner_restores_environment_on_failure(self):
        from test_labs import runtime_environment
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                with runtime_environment(False, "cpu"):
                    raise RuntimeError("test failure")
            self.assertEqual(dict(os.environ), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
