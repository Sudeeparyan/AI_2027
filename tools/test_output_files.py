"""Exercise complete replacement and failure preservation in generated text writes."""
import mmap
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import nbformat
import output_files


class OutputFileTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "lab.ipynb"

    def test_replaces_file_after_mapped_reader_closes(self):
        self.path.write_bytes(b"previous complete output")
        with self.path.open("rb") as stream:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.assertEqual(mapping[:], b"previous complete output")
        closer = threading.Timer(0.05, mapping.close)
        closer.start()
        try:
            output_files.atomic_text(self.path, "new complete output")
            self.assertEqual(self.path.read_text(encoding="utf-8"), "new complete output")
        finally:
            closer.join()
            if not mapping.closed:
                mapping.close()
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_failed_replace_keeps_destination_and_cleans_temporary(self):
        self.path.write_bytes(b"previous complete output")
        with patch.object(output_files.os, "replace", side_effect=PermissionError(13, "locked")), patch.object(output_files.time, "sleep"):
            with self.assertRaises(PermissionError):
                output_files.atomic_text(self.path, "replacement")
        self.assertEqual(self.path.read_bytes(), b"previous complete output")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_transient_lock_retries_without_truncating_target(self):
        self.path.write_bytes(b"previous complete output")
        replace = output_files.os.replace
        observed = []
        def locked_once(source, destination):
            observed.append(self.path.read_bytes())
            if len(observed) == 1:
                raise PermissionError(13, "temporary lock")
            replace(source, destination)
        with patch.object(output_files.os, "replace", side_effect=locked_once), patch.object(output_files.time, "sleep") as sleep:
            output_files.atomic_text(self.path, "replacement")
        self.assertEqual(observed, [b"previous complete output"] * 2)
        sleep.assert_called_once_with(0.1)
        self.assertEqual(self.path.read_bytes(), b"replacement")

    def test_notebook_serialization_matches_existing_writer(self):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("μ → σ")])
        original = self.path.with_name("original.ipynb")
        nbformat.write(notebook, original)
        output_files.atomic_text(self.path, nbformat.writes(notebook) + "\n")
        self.assertEqual(self.path.read_bytes(), original.read_bytes())

    def test_binary_copy_preserves_exact_asset_bytes(self):
        original = self.path.with_name("diagram.png")
        original.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")
        self.path.write_bytes(b"old")
        output_files.atomic_copy(original, self.path)
        self.assertEqual(self.path.read_bytes(), original.read_bytes())

    def test_identical_copy_does_not_replace_mapped_asset(self):
        original = self.path.with_name("diagram.png")
        original.write_bytes(b"identical asset")
        self.path.write_bytes(original.read_bytes())
        with self.path.open("rb") as stream:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with patch.object(output_files.os, "replace", side_effect=AssertionError("must not replace")):
                output_files.atomic_copy(original, self.path)
            self.assertEqual(mapping[:], b"identical asset")
        finally:
            mapping.close()


if __name__ == "__main__":
    unittest.main()
