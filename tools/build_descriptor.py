"""Build the revised module descriptor.

Only the Section 7.3 weekly table (Detail and Tutorials columns) changes. The table is
spliced into word/document.xml as a string, so every other byte of the document and every
other ZIP part stays identical to the head professor's template.

Outputs (deliverables/00_Course/):
  Descriptor - Generative AI (7.3 revised).docx          clean copy
  Descriptor - Generative AI (7.3 tracked changes).docx  same edit as Word tracked changes

Run: .venv/Scripts/python.exe tools/build_descriptor.py
"""
from __future__ import annotations

import copy
import re
import sys
import zipfile
from pathlib import Path

import yaml
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "context" / "Descriptor - Generative AI-old.docx"
SPEC = ROOT / "curriculum" / "section_7_3.yaml"
OUT_DIR = ROOT / "deliverables" / "00_Course"
CLEAN = OUT_DIR / "Descriptor - Generative AI (7.3 revised).docx"
TRACKED = OUT_DIR / "Descriptor - Generative AI (7.3 tracked changes).docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
AUTHOR = "Proposed 7.3 revision"
DATE = "2026-09-28T00:00:00Z"
DETAIL_COL, TUTORIAL_COL = 2, 3


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def cell_text(tc) -> str:
    return " ".join(
        "".join(t.text or "" for t in p.iter(w("t"))).strip() for p in tc.iter(w("p"))
    ).strip()


def locate_table(xml: str) -> tuple[int, int]:
    """Return the [start, end) span of the 7.3 weekly table inside document.xml."""
    header = xml.index(">Tutorials (Examples)<")
    start = xml.rindex("<w:tbl>", 0, header)
    end = xml.index("</w:tbl>", header) + len("</w:tbl>")
    if "<w:tbl>" in xml[start + 7:end]:
        raise RuntimeError("Unexpected nested table inside Section 7.3")
    return start, end


def parse_fragment(fragment: str, root_tag_open: str):
    """Parse a table fragment by borrowing the namespace declarations of the document root."""
    wrapped = root_tag_open + fragment + "</w:document>"
    return etree.fromstring(wrapped.encode("utf-8"))[0]


def make_run(text: str) -> etree._Element:
    r = etree.Element(w("r"))
    rpr = etree.SubElement(r, w("rPr"))
    etree.SubElement(rpr, w("sz")).set(w("val"), "20")
    t = etree.SubElement(r, w("t"))
    t.text = text
    t.set(f"{{{XML_NS}}}space", "preserve")
    return r


def paragraph_like(template_p, text: str) -> etree._Element:
    """New cell paragraph: the template's TableParagraph style with one uniform spacing.

    The original cells mix exact line heights and zero left indents (rows 4 and 7), so new
    text uses a single consistent layout instead of copying each cell's quirks.
    """
    p = etree.Element(w("p"))
    ppr = etree.SubElement(p, w("pPr"))
    style = template_p.find(f"{w('pPr')}/{w('pStyle')}")
    if style is not None:
        ppr.append(copy.deepcopy(style))
    spacing = etree.SubElement(ppr, w("spacing"))
    spacing.set(w("before"), "20")
    spacing.set(w("after"), "40")
    spacing.set(w("line"), "252")
    spacing.set(w("lineRule"), "auto")
    etree.SubElement(ppr, w("ind")).set(w("right"), "85")
    rpr = etree.SubElement(ppr, w("rPr"))
    etree.SubElement(rpr, w("sz")).set(w("val"), "20")
    p.append(make_run(text))
    return p


def mark_paragraph(p, kind: str, rid: int) -> None:
    """Add <w:ins>/<w:del> as the first child of the paragraph-mark run properties."""
    ppr = p.find(w("pPr"))
    if ppr is None:
        ppr = etree.Element(w("pPr"))
        p.insert(0, ppr)
    rpr = ppr.find(w("rPr"))
    if rpr is None:
        rpr = etree.SubElement(ppr, w("rPr"))
    mark = etree.Element(w(kind))
    mark.set(w("id"), str(rid))
    mark.set(w("author"), AUTHOR)
    mark.set(w("date"), DATE)
    rpr.insert(0, mark)


class Ids:
    def __init__(self, start: int):
        self.n = start

    def __call__(self) -> int:
        self.n += 1
        return self.n


def rebuild_cell(tc, paragraphs: list[str], tracked: bool, ids: Ids) -> None:
    old_ps = tc.findall(w("p"))
    template = old_ps[0]
    new_ps = [paragraph_like(template, text) for text in paragraphs]
    if not tracked:
        for p in old_ps:
            tc.remove(p)
        for p in new_ps:
            tc.append(p)
        return
    # Tracked: old content deleted (runs + paragraph marks), then new paragraphs inserted.
    # Accepting all changes yields exactly the clean cell.
    for p in old_ps:
        for r in list(p.iter(w("r"))):
            for t in r.findall(w("t")):
                t.tag = w("delText")
            parent = r.getparent()
            wrapper = etree.Element(w("del"))
            wrapper.set(w("id"), str(ids()))
            wrapper.set(w("author"), AUTHOR)
            wrapper.set(w("date"), DATE)
            parent.replace(r, wrapper)
            wrapper.append(r)
        mark_paragraph(p, "del", ids())
    for i, p in enumerate(new_ps):
        run = p.find(w("r"))
        wrapper = etree.Element(w("ins"))
        wrapper.set(w("id"), str(ids()))
        wrapper.set(w("author"), AUTHOR)
        wrapper.set(w("date"), DATE)
        p.replace(run, wrapper)
        wrapper.append(run)
        if i < len(new_ps) - 1:
            mark_paragraph(p, "ins", ids())
        tc.append(p)


def build(tracked: bool) -> bytes:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))["weeks"]
    assert [s["week"] for s in spec] == list(range(1, 13)), "Spec must list weeks 1..12 in order"

    with zipfile.ZipFile(SOURCE) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    start, end = locate_table(xml)
    root_open = re.search(r"<w:document\b[^>]*>", xml).group(0)
    tbl = parse_fragment(xml[start:end], root_open)
    rows = tbl.findall(w("tr"))
    assert len(rows) == 13, f"Expected 13 rows in 7.3 table, found {len(rows)}"
    assert cell_text(rows[0].findall(w("tc"))[3]) == "Tutorials (Examples)"

    max_id = max([int(m) for m in re.findall(r'w:id="(\d+)"', xml)] or [0])
    ids = Ids(max_id + 1000)

    for row, week in zip(rows[1:], spec):
        cells = row.findall(w("tc"))
        assert len(cells) == 4
        topic, number = cell_text(cells[0]), cell_text(cells[1])
        assert number == str(week["week"]), f"Week cell mismatch: {number} vs {week['week']}"
        # Lecture topics are the professor's and must stay unchanged.
        assert " ".join(topic.split()) == " ".join(week["topic"].split()), (
            f"Topic changed for week {week['week']}: {topic!r} vs {week['topic']!r}"
        )
        detail = " ".join(week["detail"].split())
        rebuild_cell(cells[DETAIL_COL], [detail], tracked, ids)
        rebuild_cell(cells[TUTORIAL_COL], [" ".join(t.split()) for t in week["tutorials"]], tracked, ids)

    new_tbl = etree.tostring(tbl, encoding="unicode")
    # lxml re-declares namespaces on the detached fragment; strip them (root already declares).
    new_tbl = re.sub(r'^<w:tbl\b[^>]*>', "<w:tbl>", new_tbl)
    new_xml = xml[:start] + new_tbl + xml[end:]
    etree.fromstring(new_xml.encode("utf-8"))  # must still be well-formed
    return new_xml.encode("utf-8")


def write_docx(target: Path, document_xml: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with zipfile.ZipFile(SOURCE) as src, zipfile.ZipFile(tmp, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "word/document.xml":
                data = document_xml
            dst.writestr(info, data, compress_type=info.compress_type)
    tmp.replace(target)


def verify(target: Path) -> list[str]:
    """Everything except the 7.3 table must be byte-identical to the template."""
    problems = []
    with zipfile.ZipFile(SOURCE) as a, zipfile.ZipFile(target) as b:
        if a.namelist() != b.namelist():
            problems.append("ZIP part list differs")
        for name in a.namelist():
            if name == "word/document.xml":
                continue
            if a.read(name) != b.read(name):
                problems.append(f"Part changed: {name}")
        old = a.read("word/document.xml").decode("utf-8")
        new = b.read("word/document.xml").decode("utf-8")
    s1, e1 = locate_table(old)
    s2, e2 = locate_table(new)
    if old[:s1] != new[:s2]:
        problems.append("document.xml differs BEFORE the 7.3 table")
    if old[e1:] != new[e2:]:
        problems.append("document.xml differs AFTER the 7.3 table")
    return problems


def main() -> int:
    for target, tracked in ((CLEAN, False), (TRACKED, True)):
        write_docx(target, build(tracked))
        problems = verify(target)
        status = "OK" if not problems else "FAILED: " + "; ".join(problems)
        print(f"{target.name}: {status}")
        if problems:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
