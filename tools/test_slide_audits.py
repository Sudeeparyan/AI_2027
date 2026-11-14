"""Built-table text must obey the same limits as ordinary text boxes."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from audit_slide_text import audit
from verify_pilot_artifacts import readability


def deck(path, words=41, size=1400, y=1828800, table=True):
    text = ' '.join(['evidence'] * words)
    frame = (f'<p:graphicFrame><p:xfrm><a:off x="0" y="{y}"/></p:xfrm>'
             f'<a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody>'
             f'<a:p><a:r><a:rPr sz="{size}"/><a:t>{text}</a:t></a:r></a:p>'
             '</a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>')
    if not table:
        frame = (f'<p:sp><p:spPr><a:xfrm><a:off x="0" y="{y}"/></a:xfrm></p:spPr>'
                 f'<p:txBody><a:p><a:r><a:rPr sz="{size}"/><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>')
    xml = ('<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
           '<p:cSld><p:spTree>' + frame + '</p:spTree></p:cSld></p:sld>')
    with zipfile.ZipFile(path, 'w') as package:
        package.writestr('ppt/slides/slide1.xml', xml)


class SlideAuditTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'fixture.pptx'
        self.spec = {'slides': [{'type': 'table', 'title': 'Evidence'}]}

    def test_table_and_text_body_both_enforce_forty_words(self):
        for table in (True, False):
            with self.subTest(table=table):
                deck(self.path, words=41, table=table)
                self.assertEqual(readability(self.path, self.spec)['above_40_word_slides'], [1])
                deck(self.path, words=40, table=table)
                self.assertEqual(readability(self.path, self.spec)['above_40_word_slides'], [])

    def test_table_font_below_fourteen_points_is_detected(self):
        deck(self.path, words=2, size=1300)
        self.assertEqual(audit(self.path)[0]['font_pt'], 13)
        deck(self.path, words=2, size=1400)
        self.assertEqual(audit(self.path), [])

    def test_running_header_is_excluded_from_body_and_font_limits(self):
        deck(self.path, words=50, size=1100, y=91440)
        self.assertEqual(readability(self.path, self.spec)['above_40_word_slides'], [])
        self.assertEqual(audit(self.path), [])


if __name__ == '__main__':
    unittest.main()
