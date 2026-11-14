"""Reject Office previews made from an older version of a distributed file.

Run after rendering: python tools/verify_render_inputs.py
Ignore package timestamps and random relationship IDs, but compare instructional
XML, relationships, embedded images and every other package part. This checks
preview provenance; it does not replace looking at the rendered pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import sys
import zipfile

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def package_content(path: Path) -> dict[str, bytes]:
    """Normalise volatile Office metadata without discarding teaching content."""
    with zipfile.ZipFile(path) as package:
        parts = {name: package.read(name) for name in package.namelist()
                 if name != "docProps/core.xml" and not name.endswith("/")}
    owner_maps = {}
    for name, data in list(parts.items()):
        if not name.endswith(".rels"):
            continue
        root = etree.fromstring(data)
        replacements = {}
        for relation in root:
            attributes = sorted((k, v) for k, v in relation.attrib.items() if k != "Id")
            identity = hashlib.sha256(repr(attributes).encode("utf-8")).hexdigest()
            replacements[relation.get("Id")] = "rel_" + identity
            relation.set("Id", "rel_" + identity)
        root[:] = sorted(root, key=lambda relation: relation.get("Id"))
        parts[name] = etree.tostring(root, method="c14n2")
        parent, filename = posixpath.split(name)
        owner = posixpath.join(posixpath.dirname(parent), filename[:-5])
        owner_maps[owner] = replacements
    for name, data in list(parts.items()):
        if not name.endswith(".xml"):
            continue
        root = etree.fromstring(data)
        replacements = owner_maps.get(name, {})
        for node in root.iter():
            for key, value in list(node.attrib.items()):
                if key.startswith("{" + DOC_REL + "}") and value in replacements:
                    node.set(key, replacements[value])
        parts[name] = etree.tostring(root, method="c14n2")
    return parts


def differences(distributed: Path, rendered: Path) -> list[str]:
    if not distributed.is_file():
        return ["distributed file missing"]
    if not rendered.is_file():
        return ["render input missing"]
    current, preview = package_content(distributed), package_content(rendered)
    return [name for name in sorted(current.keys() | preview.keys())
            if current.get(name) != preview.get(name)]


def pairs(weeks: list[int], include_course: bool):
    for week in weeks:
        directory = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*"))
        preview = ROOT / f"build/render/week_{week:02d}"
        for suffix, render_name in (("Lecture_Slides.pptx", "slides.pptx"),
                                    ("Teaching_Notes.docx", "notes.docx")):
            yield directory / f"Week_{week:02d}_{suffix}", preview / render_name
    if include_course:
        for source in sorted((ROOT / "deliverables/00_Course").glob("*.docx")):
            yield source, ROOT / "build/render/course" / source.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", default="1-12")
    parser.add_argument("--no-course", action="store_true")
    args = parser.parse_args()
    weeks = sorted({week for group in args.weeks.split(",") if group
                    for week in range(int(group.split("-")[0]), int(group.split("-")[-1]) + 1)})
    report = []
    for source, preview in pairs(weeks, not args.no_course):
        changed = differences(source, preview)
        report.append({"distributed": str(source.relative_to(ROOT)),
                       "render_input": str(preview.relative_to(ROOT)),
                       "changed_parts": changed, "ok": not changed})
        print(f"{'FAIL' if changed else 'PASS'} {source.name}" +
              (f": {', '.join(changed[:4])}" if changed else ""))
    target = ROOT / "build/render_input_verification.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(not entry["ok"] for entry in report))


if __name__ == "__main__":
    sys.exit(main())
