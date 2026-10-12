"""Regression tests for note migration, section checks and existing prose audits."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import audit_prose
import build_week
from speaker_notes import note_sentences, quiz_answer_letter, validate_speaker_notes


BASE_NOTE = (
    "In short: A model compares numerical representations before it predicts a label. "
    "Explain: An embedding is a list of numbers representing one input. "
    "Similar inputs can receive nearby representations during training. "
    "A score measures how strongly two representations match. "
    "We check predictions against labels that were excluded from training. "
    "Example: Consider three synthetic messages and three candidate labels. "
    "Each message becomes one row in a matrix before we calculate its scores. "
    "The analogy is a map, but nearby points need not share every property. "
    "Ask: Which option checks whether the model can handle a new message? "
    "Answer: B. Use a separate held-out example with a known label. "
    "Watch out: A high score does not guarantee that the predicted label is correct."
)
LEGACY_NOTE = (
    "Say: Explain how one model represents an input before predicting a label. "
    "Analogy: A map records locations but leaves out many properties of the places. "
    "Ask: How can we check the prediction on a new message? "
    "Answer: B. Compare it with a label excluded from training. "
    "Common mistake: Treating one successful example as evidence of general reliability."
)


class SpeakerNoteTests(unittest.TestCase):
    def test_valid_modern_note_and_bold_labels(self):
        self.assertEqual([], validate_speaker_notes(BASE_NOTE, quiz_answer=1).errors)
        bold = BASE_NOTE
        for label in ("In short", "Explain", "Example", "Ask", "Answer", "Watch out"):
            bold = bold.replace(label + ":", "**" + label + ":**")
        self.assertEqual([], validate_speaker_notes(bold, quiz_answer=1).errors)

    def test_legacy_warns_and_strict_mode_fails(self):
        result = validate_speaker_notes(LEGACY_NOTE, strict_legacy=False, quiz_answer=1)
        self.assertEqual([], result.errors)
        self.assertEqual(1, len(result.warnings))
        strict = validate_speaker_notes(LEGACY_NOTE, strict_legacy=True)
        self.assertEqual([], strict.warnings)
        self.assertTrue(any("legacy" in error for error in strict.errors))

    def test_unlabelled_legacy_with_ask_answer_cues_still_warns(self):
        result = validate_speaker_notes(LEGACY_NOTE.replace("Say:", "Discuss").replace("Analogy:", "Imagine").replace("Common mistake:", "Avoid"), strict_legacy=False)
        self.assertFalse(result.modern)
        self.assertEqual([], result.errors)
        self.assertEqual(1, len(result.warnings))

    def test_final_gate_default_rejects_legacy_without_affecting_modern_notes(self):
        with patch("speaker_notes.STRICT_LEGACY_NOTES", True):
            legacy = validate_speaker_notes(LEGACY_NOTE)
            self.assertTrue(legacy.errors)
            self.assertEqual([], legacy.warnings)
            self.assertEqual([], validate_speaker_notes(BASE_NOTE).errors)

    def test_short_legacy_notes_still_fail(self):
        self.assertTrue(any("minimum 45" in error for error in
                            validate_speaker_notes("Say: Too little detail.").errors))

    def test_missing_reordered_duplicate_and_empty_sections_fail(self):
        for malformed, message in (
            (BASE_NOTE.replace("Example:", "Illustration:"), "'Example:' occurs 0"),
            (BASE_NOTE.replace("Ask:", "TEMP:").replace("Answer:", "Ask:").replace("TEMP:", "Answer:"), "out of order"),
            (BASE_NOTE + " Watch out: Another warning.", "'Watch out:' occurs 2"),
            (BASE_NOTE.replace("Example: Consider", "Example: Ask: Consider"), "'Example:' is empty"),
        ):
            with self.subTest(message=message):
                self.assertTrue(any(message in error for error in validate_speaker_notes(malformed).errors))

    def test_mixed_formats_fail_and_missing_first_label_is_modern(self):
        self.assertTrue(validate_speaker_notes(BASE_NOTE.replace("In short:", "Say:")).errors)
        self.assertTrue(validate_speaker_notes(BASE_NOTE.replace("In short:", "Summary:")).modern)

    def test_total_word_bounds_include_labels(self):
        start, rest = BASE_NOTE.split("Example:")
        short_base = start + "Example: One message becomes a vector. Ask:" + rest.split("Ask:", 1)[1]
        for words in (110, 180):
            note = short_base.replace("Ask:", " ".join(["detail"] * (words - len(short_base.split()))) + " Ask:")
            self.assertEqual(words, len(note.split()))
            self.assertEqual([], validate_speaker_notes(note).errors)
        for words in (109, 181):
            note = short_base.replace("Ask:", " ".join(["detail"] * (words - len(short_base.split()))) + " Ask:")
            self.assertTrue(any("including labels" in error for error in validate_speaker_notes(note).errors))

    def test_explain_sentence_count_and_length(self):
        too_few = BASE_NOTE.replace(
            "Similar inputs can receive nearby representations during training. ", "").replace(
            "A score measures how strongly two representations match. ", "")
        too_long = BASE_NOTE.replace("An embedding is", " ".join(["extra"] * 21) + " An embedding is")
        self.assertTrue(any("expected 3–6" in error for error in validate_speaker_notes(too_few).errors))
        self.assertTrue(any("maximum is 20" in error for error in validate_speaker_notes(too_long).errors))

    def test_decimal_abbreviations_and_code_are_not_sentence_endings(self):
        text = "A score is 0.6, e.g. on this input. Inspect `items. count` first. Check the final prediction."
        self.assertEqual(3, len(note_sentences(text)))

    def test_quiz_letter_rule_for_modern_legacy_and_bold_notes(self):
        self.assertEqual("B", quiz_answer_letter(BASE_NOTE))
        self.assertEqual("B", quiz_answer_letter(LEGACY_NOTE))
        self.assertEqual("D", quiz_answer_letter("Say: Answer D. Ask: Why is A wrong? Answer: A is a hard maximum."))
        self.assertEqual("B", quiz_answer_letter(BASE_NOTE.replace("Answer:", "**Answer:**")))
        mismatch = validate_speaker_notes(BASE_NOTE, quiz_answer=0)
        self.assertTrue(any("notes say B" in error for error in mismatch.errors))

    def test_build_validation_summarises_legacy_warnings(self):
        week = {
            "expected_slides": 3,
            "slides": [{"type": "bullets", "title": "Example", "bullets": ["One idea"], "notes": LEGACY_NOTE}] * 3,
            "subtitle": "Example", "mimlos": ["LO1"], "objectives": ["Explain"], "coverage": ["Example"],
            "notes_md": "\n".join("# " + title for title in build_week.REQUIRED_NOTE_SECTIONS),
        }
        warnings = []
        self.assertEqual([], build_week.validate(week, strict_notes=False, note_warnings=warnings))
        self.assertEqual(1, len(warnings))
        self.assertIn("3 slide(s)", warnings[0])
        self.assertEqual(3, len(build_week.validate(week, strict_notes=True)))

    def test_watch_out_labels_repetition_and_answers_stay_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weeks").mkdir()
            (root / "notes").mkdir()
            slides = [{"notes": f"In short: Slide {index}. Ask: What happens? Watch out: No. "
                                  "Watch out: Never treat one successful example as proof of reliable general performance."}
                      for index in range(3)]
            (root / "weeks" / "week_01.yaml").write_text(yaml.safe_dump({"slides": slides}), encoding="utf-8")
            with patch.object(audit_prose, "ROOT", root):
                findings = audit_prose.repeated_notes()
            self.assertTrue(any(finding.startswith("REPEAT") for finding in findings))
            self.assertEqual(3, sum(finding.startswith("LABEL") for finding in findings))
            self.assertEqual(3, sum(finding.startswith("ANSWER") for finding in findings))

    def test_prose_quiz_mismatch_remains_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weeks").mkdir()
            (root / "notes").mkdir()
            slide = {"type": "quiz", "options": ["First", "Second", "Third", "Fourth"],
                     "answer": 0, "notes": BASE_NOTE}
            (root / "weeks" / "week_01.yaml").write_text(yaml.safe_dump({"slides": [slide]}), encoding="utf-8")
            with patch.object(audit_prose, "ROOT", root):
                findings = audit_prose.quiz_answers()
            self.assertTrue(any("answer index is A, notes say B" in finding for finding in findings))


if __name__ == "__main__":
    unittest.main()
