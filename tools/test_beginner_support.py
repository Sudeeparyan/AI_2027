"""Regression checks for course vocabulary and moved recap slide references."""
import json
import unittest

import beginner
import build_week


class BeginnerSupportTests(unittest.TestCase):
    def test_unquoted_yaml_mapping_cannot_render_as_bullet(self):
        week = build_week.load_week(8)
        slide = next(s for s in week["slides"] if s.get("bullets"))
        slide["bullets"] = [{"BloombergGPT": "Mixed data"}]
        self.assertTrue(any("bullets must be a list of strings" in error for error in build_week.validate(week)))

    def test_range_includes_recap_moved_past_resources(self):
        mapping = {38: 44, 39: 46, 40: 45}
        source = "# Lecture plan\n\n| Minutes | Slides | Task |\n|---|---|---|\n| 110–120 | 38–40 | Review |\n\n# Lecture notes\nRead slides 38–40. Compare slide 40.\n"
        result = beginner.remap_slide_references(source, mapping)
        self.assertIn("| 110–120 | 44–46 | Review |", result)
        self.assertIn("Read slides 44–46. Compare slide 45.", result)
        self.assertNotIn("44–45", result)

    def test_every_week_uses_canonical_meanings(self):
        source = json.loads((beginner.ROOT / "curriculum/glossary.json").read_text(encoding="utf-8"))
        meanings = {p["term"].casefold(): p["meaning"] for p in source["terms"]}
        seen = set()
        for week in range(1, 13):
            data = beginner.load(week)
            for item in data["prerequisites"] + data["glossary"]:
                key = item["term"].casefold()
                self.assertIn(key, meanings, (week, key))
                self.assertEqual(item["meaning"], meanings[key], (week, key))
                seen.add(key)
        self.assertEqual(seen, set(meanings))


if __name__ == "__main__":
    unittest.main()
