"""Regressions for authored notes, compact vocabulary and contact-sheet grouping."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import yaml

import beginner
import notes_sheets


def authored_notes(subject):
    return (f"In short: {subject} identifies the operation.\n"
            "Explain: Read inputs before outputs. Describe the operation in plain words.\n"
            "Example: Three inputs produce three rows.\n"
            "Ask: What do the rows count?\nAnswer: They count inputs.\n"
            "Watch out: Do not confuse rows with vector width.")


class VisualSupportTests(unittest.TestCase):
    def fixture(self):
        data = copy.deepcopy(beginner.load(1))
        week = yaml.safe_load((beginner.ROOT / "curriculum/weeks/week_01.yaml").read_text(encoding="utf-8"))
        week["week"] = 1
        week["notes_md"] = "# Lecture notes\nRead the worked example.\n"
        return week, data

    def test_modern_support_notes_pass_through_without_legacy_suffixes(self):
        week, data = self.fixture()
        agenda = next(s for s in week["slides"] if s["type"] == "agenda")
        agenda["notes"] = authored_notes("Course map")
        data["vocabulary_notes"] = authored_notes("Vocabulary")
        data["recap"]["notes"] = authored_notes("Recap")
        for diagram in data["diagrams"]:
            diagram["notes"] = authored_notes(diagram["id"])
        with patch.object(beginner, "load", return_value=data):
            result = beginner.enrich(week)
        self.assertEqual(next(s for s in result["slides"] if s["type"] == "course_map")["notes"], agenda["notes"])
        self.assertEqual(next(s for s in result["slides"] if s.get("title") == "Starting vocabulary")["notes"], data["vocabulary_notes"])
        self.assertEqual(result["slides"][-1]["notes"], data["recap"]["notes"])
        for slide in result["slides"]:
            if slide["type"] == "technical":
                self.assertEqual(slide["notes"], slide["diagram"]["notes"])
                self.assertNotIn("Worked example:", slide["notes"])

    def test_compact_vocabulary_keeps_full_reference_and_checks(self):
        week, data = self.fixture()
        data["vocabulary_count"] = 2
        canonical = [p["meaning"] for p in data["prerequisites"]]
        for i, item in enumerate(data["prerequisites"][:2]):
            item["short_meaning"] = f"Compact meaning {i}"
        with patch.object(beginner, "load", return_value=data):
            result = beginner.enrich(week)
        slide = next(s for s in result["slides"] if s.get("title") == "Starting vocabulary")
        self.assertEqual(slide["header"], ["Term", "Meaning"])
        self.assertEqual(slide["rows"], [[p["term"], p["short_meaning"]] for p in data["prerequisites"][:2]])
        self.assertEqual([p["meaning"] for p in data["prerequisites"]], canonical)
        for p in data["prerequisites"]:
            self.assertIn(p["meaning"].replace("|", "/").replace("\n", " "), result["notes_md"])
            self.assertIn(p["answer"], result["notes_md"])

    def test_canonical_short_meaning_wins_without_replacing_full_meaning(self):
        _, data = self.fixture()
        first = data["prerequisites"][0]
        first["short_meaning"] = "Authored short"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "curriculum/beginner/week_01.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(data), encoding="utf-8")
            (root / "curriculum/glossary.json").write_text(json.dumps({"terms":[{
                "term":first["term"],"meaning":"Canonical full meaning","short_meaning":"Canonical short"}]}), encoding="utf-8")
            with patch.object(beginner, "ROOT", root):
                loaded = beginner.load(1)
            self.assertEqual(loaded["prerequisites"][0]["meaning"], "Canonical full meaning")
            self.assertEqual(loaded["prerequisites"][0]["short_meaning"], "Canonical short")
            data["vocabulary_count"] = 0
            source.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(beginner, "ROOT", root), self.assertRaisesRegex(ValueError, "vocabulary_count"):
                beginner.load(1)

    def test_six_slide_tiles_preserve_numeric_order_and_source_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "build/render/week_01"
            source.mkdir(parents=True)
            colours = [(number * 20, 0, 0) for number in range(1, 8)]
            files=[]
            for number, colour in zip([1, 2, 3, 4, 5, 6, 10], colours):
                path = source / f"slides-{number}.png"
                Image.new("RGB", (120, 60), colour).save(path)
                files.append(path)
            before = [p.read_bytes() for p in files]
            Image.new("RGB", (120, 60), "blue").save(source / "slides-sheet-1.png")
            with patch.object(notes_sheets, "ROOT", root):
                count, out = notes_sheets.slide_sheets(1, cell_w=120)
            self.assertEqual(count, 7)
            self.assertEqual(len(list(out.glob("*.png"))), 2)
            with Image.open(out / "sheet_01.png") as sheet:
                self.assertEqual(sheet.size, (270, 280))
                for i, colour in enumerate(colours[:6]):
                    self.assertEqual(sheet.getpixel((70 + (i % 2) * 130, 58 + (i // 2) * 90)), colour)
            with Image.open(out / "sheet_02.png") as sheet:
                self.assertEqual(sheet.getpixel((70, 58)), colours[-1])
            self.assertEqual([p.read_bytes() for p in files], before)

    def test_empty_or_invalid_sheet_request_has_clear_behaviour(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(notes_sheets, "ROOT", Path(folder)):
                count, out = notes_sheets.slide_sheets(1)
                self.assertEqual(count, 0)
                self.assertEqual(list(out.glob("*.png")), [])
                with self.assertRaisesRegex(ValueError, "positive"):
                    notes_sheets.slide_sheets(1, per=0)
                with self.assertRaisesRegex(FileNotFoundError, "No rendered images"):
                    notes_sheets.slide_sheet([1], 1)


if __name__ == "__main__":
    unittest.main()
