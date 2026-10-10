"""Audit instructional PowerPoint text sizes; exclude the small page chrome."""
from __future__ import annotations
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def audit(path):
    issues = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
            if not match:
                continue
            root = ET.fromstring(archive.read(name))
            for shape in root.findall(".//p:sp", NS):
                offset = shape.find("p:spPr/a:xfrm/a:off", NS)
                y = int(offset.get("y")) / 914400 if offset is not None else 0
                if y < .5 or y >= 7:
                    continue  # running header, course footer and page number
                for paragraph in shape.findall(".//a:p", NS):
                    default = paragraph.find("a:pPr/a:defRPr", NS)
                    for run in paragraph.findall("a:r", NS):
                        text = run.find("a:t", NS)
                        props = run.find("a:rPr", NS)
                        size = props.get("sz") if props is not None else None
                        size = size or (default.get("sz") if default is not None else None)
                        if text is not None and text.text and size and int(size) < 1400:
                            issues.append({"slide": int(match[1]), "font_pt": int(size) / 100, "text": text.text})
    return issues


def main():
    report = []
    for week in range(1, 13):
        path = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*")) / f"Week_{week:02d}_Lecture_Slides.pptx"
        issues = audit(path)
        report.append({"week": week, "issues": issues})
        print(f"week {week:02d}: {len(issues)} instructional text runs below 14 pt")
    (ROOT / "build/slide_text_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return int(any(r["issues"] for r in report))


if __name__ == "__main__":
    sys.exit(main())
