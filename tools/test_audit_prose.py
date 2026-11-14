"""Keep Markdown activity boxes distinct from genuinely damaged notation."""
import unittest
import audit_prose


class ProseAuditTests(unittest.TestCase):
    def test_question_followed_by_activity_quote_is_valid_prose(self):
        source = "> **Predict:** Does doubling rank double the adapter count?\n> **Core:** Complete the adapter.\n"
        cleaned = audit_prose.clean(source)
        self.assertIsNone(audit_prose.LOST.search(cleaned))
        self.assertEqual(len(cleaned), len(source))
        self.assertEqual(cleaned.count("\n"), source.count("\n"))

    def test_damaged_symbol_in_sentence_still_fails(self):
        self.assertIsNotNone(audit_prose.LOST.search(audit_prose.clean("Compare x ? > y.")))


if __name__ == "__main__":
    unittest.main()
