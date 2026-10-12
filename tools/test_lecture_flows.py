"""Small isolated E3-flow regressions; no weekly build, Office or lab run."""
import copy
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import nbformat
from PIL import Image

import beginner
import build_week
import lecture_flows
import verify_beginner


def fixture():
    return {"type": "flow", "title": "Read and report", "steps": [
        {"label": "Read", "role": "data"}, {"label": "Report", "role": "output"}], "arrows": ["values"]}


class LectureFlowTests(unittest.TestCase):
    def test_partial_annotation_and_invalid_connections_fail(self):
        original = fixture()
        self.assertEqual(lecture_flows.errors(original), [])
        for modify in (lambda s: s.pop("arrows"), lambda s: s["steps"][1].pop("role"),
                       lambda s: s["arrows"].append("extra"), lambda s: s["arrows"].__setitem__(0, ""),
                       lambda s: s["arrows"].__setitem__(0, None),
                       lambda s: s["steps"][1].__setitem__("role", []),
                       lambda s: s.update(feedback="repeat", feedback_to=20)):
            value = copy.deepcopy(original)
            modify(value)
            self.assertTrue(lecture_flows.errors(value), value)
        legacy = {"type": "flow", "steps": [{"label": "First"}, {"label": "Next"}]}
        self.assertFalse(lecture_flows.annotated(legacy))
        self.assertEqual(lecture_flows.errors(legacy), [])

    def test_null_arrow_requires_an_explicit_input_branch(self):
        value = {"type": "flow", "steps": [{"label": "Read", "role": "data"},
                 {"label": "Draw", "role": "data"}, {"label": "Combine", "role": "model"}],
                 "arrows": [None, "noise"], "branches": [{"from": 0, "to": 2, "label": "input"}]}
        self.assertEqual(lecture_flows.errors(value), [])
        value["branches"][0]["from"] = True
        self.assertTrue(lecture_flows.errors(value))

    def test_word_image_joins_exact_section_and_is_idempotent(self):
        value = fixture()
        lecture_flows.collect({"slides": [{"type": "title"}, value]})
        original = "# Lecture notes\n\n## Slide 2: Read and report\n\nOriginal explanation.\n\n## Slide 3: Next\n\nNext content.\n"
        result = lecture_flows.notes_with_flows(original, [{"type": "title"}, value])
        self.assertEqual(result.count("fig:lecture_flow_02"), 1)
        self.assertLess(result.index("fig:lecture_flow_02"), result.index("Original explanation."))
        self.assertIn("Original explanation.", result)
        self.assertEqual(lecture_flows.notes_with_flows(result, [{"type": "title"}, value]), result)
        fallback = lecture_flows.notes_with_flows("# Notes\nText.\n# Class recap\nReview.\n", [value])
        self.assertLess(fallback.index("# Lecture flow diagrams"), fallback.index("# Class recap"))

    def test_cli_exports_single_scene_and_pilot_flows_fit(self):
        # An isolated renderer unit fixture; no deliverables or machine jobs.
        pilot = build_week.load_week(1)
        authored = lecture_flows.collect(pilot)
        payload = {"flows": authored or [{"id": "lecture_flow_01", "slide": fixture()}]}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            source = out / "input.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            subprocess.run(["node", str(build_week.TOOLS / "test_lecture_flows.js"), str(source)], check=True)
            subprocess.run(["node", str(build_week.TOOLS / "technical_diagrams.js"), "--flows", str(source), str(out)], check=True)
            for entry in payload["flows"]:
                identifier = entry["id"]
                scene = json.loads((out / f"{identifier}.scene.json").read_text(encoding="utf-8"))
                with Image.open(out / f"{identifier}.png") as image:
                    self.assertAlmostEqual(image.width / image.height, scene["width"] / scene["height"], delta=.005)
                self.assertIn("<polyline", (out / f"{identifier}.svg").read_text(encoding="utf-8"))
                self.assertEqual(build_week.asset(out / f"{identifier}.png")["scene"], scene)

    def test_optional_notebook_flow_keeps_exact_bytes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cur = root / "curriculum"
            (cur / "labs").mkdir(parents=True)
            (cur / "labs/_setup.py").write_text("# Shared setup unit fixture\n", encoding="utf-8")
            source = ("# %% [markdown]\n# # Lab\n# ![Overview](fig:beginner_overview)\n"
                      "# ![Optional lecture flow](fig:lecture_flow_01)\n"
                      "# ![Stored worked example](fig:archived_example)\n# Run to print one value.\n"
                      "# %%\nvalue = 1\nprint(value)\n")
            (cur / "labs/week_01_lab.py").write_text(source, encoding="utf-8")
            figures = {}
            for identifier, colour in (("beginner_overview", "blue"), ("lecture_flow_01", "green"),
                                       ("lecture_flow_02", "red"), ("archived_example", "orange")):
                path = root / f"{identifier}.png"
                Image.new("RGB", (30, 20), colour).save(path)
                path.with_suffix(".svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
                figures[identifier] = {"path": str(path)}
            week = {"beginner": {"diagrams": []}, "assets": {"figures": figures}}
            out = root / "deliverables/week_01"
            out.mkdir(parents=True)
            with patch.object(build_week, "CUR", cur), patch.object(build_week, "BUILD", root / "build"), patch.object(verify_beginner, "ROOT", root):
                paths = build_week.build_notebooks(1, week, out)
                for path in paths:
                    verify_beginner.verify_notebook(path, out / "Diagrams", path == paths[0], source)
                    notebook = nbformat.read(path, as_version=4)
                    self.assertTrue(any("lecture_flow_01.png" in c.get("attachments", {}) for c in notebook.cells))
                    self.assertFalse(any("lecture_flow_02.png" in c.get("attachments", {}) for c in notebook.cells))
                    self.assertTrue(any("archived_example.png" in c.get("attachments", {}) for c in notebook.cells))
                    exported = root / "build/week_01" / path.with_suffix(".py").name
                    self.assertIn("Diagrams/archived_example.png", exported.read_text(encoding="utf-8"))
                Image.new("RGB", (30, 20), "black").save(out / "Diagrams/archived_example.png")
                with self.assertRaisesRegex(AssertionError, "attachment differs"):
                    verify_beginner.verify_notebook(paths[0], out / "Diagrams", True, source)
                build_week.build_notebooks(1, week, out)
                Image.new("RGB", (30, 20), "black").save(out / "Diagrams/lecture_flow_01.png")
                with self.assertRaisesRegex(AssertionError, "attachment differs"):
                    verify_beginner.verify_notebook(paths[0], out / "Diagrams", True, source)

    def test_labs_only_resolves_named_lecture_figure_and_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cur = root / "curriculum"
            (cur / "labs").mkdir(parents=True)
            (cur / "labs/_setup.py").write_text("# Shared setup unit fixture\n", encoding="utf-8")
            source = "# %% [markdown]\n# # Lab\n# ![Stored example](fig:archived_example)\n# Run to print one value.\n# %%\nprint(1)\n"
            (cur / "labs/week_01_lab.py").write_text(source, encoding="utf-8")
            example = root / "archived_example.png"
            Image.new("RGB", (30, 20), "orange").save(example)
            out = root / "deliverables/week_01"
            out.mkdir(parents=True)
            with patch.object(build_week, "CUR", cur), patch.object(build_week, "BUILD", root / "build"), \
                 patch.object(build_week.beginner, "load", return_value=None), \
                 patch.object(build_week, "load_week", return_value={"slides": []}), \
                 patch.object(build_week, "lecture_flow_assets", return_value={}), \
                 patch.object(build_week, "run_figures", return_value={"archived_example": example}) as generate:
                paths = build_week.build_notebooks(1, {"beginner": {"diagrams": []}}, out)
                generate.assert_called_once_with(1)
                notebook = nbformat.read(paths[0], as_version=4)
                attachment = next(c.attachments["archived_example.png"]["image/png"] for c in notebook.cells if c.get("attachments"))
                import base64
                self.assertEqual(base64.b64decode(attachment), example.read_bytes())
                import xml.etree.ElementTree as ET
                svg = ET.parse(out / "Diagrams/archived_example.svg").getroot()
                embedded = svg.find("{http://www.w3.org/2000/svg}image").attrib["href"]
                self.assertEqual(base64.b64decode(embedded.partition(",")[2]), example.read_bytes())
                (cur / "labs/week_01_lab.py").write_text(source.replace("archived_example", "missing_example"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unknown figure missing_example"):
                    build_week.build_notebooks(1, {"beginner": {"diagrams": []}}, out)

    def test_word_verifier_requires_the_exact_shared_flow_png(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            support = root / "beginner_overview.png"
            lecture = root / "lecture_flow_01.png"
            Image.new("RGB", (30, 20), "blue").save(support)
            Image.new("RGB", (30, 20), "green").save(lecture)
            document = root / "notes.docx"
            spec = {"slides": [{"flow_asset": "lecture_flow_01"}]}
            with zipfile.ZipFile(document, "w") as archive:
                archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Beginner walkthrough</w:t></w:r></w:p></w:body></w:document>')
                archive.writestr("word/media/support.png", support.read_bytes())
                archive.writestr("word/media/lecture.png", lecture.read_bytes())
            result = verify_beginner.verify_docx(document, {"diagrams": []}, root, spec)
            self.assertEqual(result["embedded_lecture_flows"], 1)
            Image.new("RGB", (30, 20), "black").save(lecture)
            with self.assertRaisesRegex(AssertionError, "exact shared lecture flow"):
                verify_beginner.verify_docx(document, {"diagrams": []}, root, spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
