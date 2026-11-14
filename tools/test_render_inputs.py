"""A stale image or explanation must fail even when Office IDs change."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from verify_render_inputs import differences


def fixture(path, rid="rId1", text="Current explanation", image=b"same PNG", target="media/a.png"):
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("docProps/core.xml", "volatile " + rid)
        package.writestr("word/document.xml", f'''<w:document xmlns:w="urn:word"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <w:p>{text}</w:p><w:drawing r:embed="{rid}"/></w:document>''')
        package.writestr("word/_rels/document.xml.rels", f'''<Relationships
            xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="{rid}" Target="{target}" Type="image"/></Relationships>''')
        package.writestr("word/media/a.png", image)


class RenderInputTests(unittest.TestCase):
    def test_new_timestamp_and_relationship_id_keep_the_same_content(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.docx", Path(folder) / "b.docx"
            fixture(first)
            fixture(second, rid="random-new-id")
            self.assertEqual(differences(first, second), [])

    def test_changed_explanation_image_or_relationship_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.docx", Path(folder) / "b.docx"
            fixture(first)
            for changed in ({"text": "Old explanation"}, {"image": b"old PNG"},
                            {"target": "media/old.png"}):
                with self.subTest(changed=changed):
                    fixture(second, **changed)
                    self.assertTrue(differences(first, second))

    def test_missing_preview_cannot_certify_the_current_document(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "current.docx"
            fixture(first)
            self.assertEqual(differences(first, Path(folder) / "missing.docx"),
                             ["render input missing"])


if __name__ == "__main__":
    unittest.main()
