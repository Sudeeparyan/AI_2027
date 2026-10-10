"""Audit the projected text size of pictures placed on lecture slides.

Run after build_week.py: .venv/Scripts/python.exe tools/audit_figure_text.py [--weeks 1-12]

audit_slide_text.py checks editable PowerPoint text. Pictures need a separate
check: a 15 pt label in a 13-inch figure is smaller than 15 pt once the figure
is shrunk onto a slide. For each picture this script finds the source figure
(by exact PNG bytes), redraws it to read the smallest font, and multiplies by
the slide scale (displayed width / natural width at the saved 200 dpi).

Copied lab-result images have no redrawable source; they are listed with their
scale so they can be reviewed by eye. Equation images use their known 22 pt
render size. Exit status 1 means some drawn text projects below 14 pt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lint_figures  # noqa: E402 - also configures matplotlib and the figure path
import matplotlib.pyplot as plt  # noqa: E402

EMU = 914400
MIN_PT = 14.0
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"


def png_width(data: bytes) -> int:
    return int.from_bytes(data[16:20], "big")


def smallest_fonts(week: int, folder: Path) -> dict[str, float]:
    """Smallest visible font (pt) in each drawn figure of a week."""
    module = importlib.import_module(f"week_{week:02d}")
    result = {}
    for name in module.FIGURES:
        sizes = []
        for fig in lint_figures.capture(module, name, folder):
            fig.canvas.draw()
            sizes += [t.get_fontsize() for t in lint_figures.visible_texts(fig)]
            plt.close(fig)
        if sizes:
            result[name] = min(sizes)
    return result


def audit_week(week: int, folder: Path) -> list[dict]:
    deck = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*")) / f"Week_{week:02d}_Lecture_Slides.pptx"
    figures = ROOT / "build" / "figures" / f"week_{week:02d}"
    by_hash = {hashlib.sha256(p.read_bytes()).hexdigest(): p.stem for p in figures.glob("*.png")}
    equations = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "build" / "eq").glob("*.png")}
    fonts = smallest_fonts(week, folder)
    rows = []
    with zipfile.ZipFile(deck) as archive:
        for name in sorted(archive.namelist(), key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)]):
            match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
            if not match:
                continue
            rels = ET.fromstring(archive.read(f"ppt/slides/_rels/slide{match[1]}.xml.rels"))
            targets = {r.get("Id"): r.get("Target") for r in rels.iter(REL)}
            for pic in ET.fromstring(archive.read(name)).iter(f"{{{NS['p']}}}pic"):
                blip = pic.find(".//a:blip", NS)
                extent = pic.find(".//a:xfrm/a:ext", NS)
                if blip is None or extent is None:
                    continue
                media = "ppt/" + targets[blip.get(f"{{{NS['r']}}}embed")].replace("../", "")
                data = archive.read(media)
                digest = hashlib.sha256(data).hexdigest()
                shown = int(extent.get("cx")) / EMU
                if digest in equations:
                    natural = png_width(data) / 300
                    rows.append({"slide": int(match[1]), "figure": "equation", "scale": shown / natural,
                                 "smallest_pt": 22 * shown / natural})
                    continue
                figure = by_hash.get(digest)
                if figure is None:
                    continue  # icons and other decoration
                natural = png_width(data) / 200
                row = {"slide": int(match[1]), "figure": figure, "scale": round(shown / natural, 3)}
                if figure in fonts:
                    row["smallest_pt"] = round(fonts[figure] * shown / natural, 1)
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", default="1-12")
    args = parser.parse_args()
    report, small = [], 0
    with tempfile.TemporaryDirectory(prefix="figure-text-") as temp:
        for week in lint_figures.weeks_arg(args.weeks):
            rows = audit_week(week, Path(temp))
            report.append({"week": week, "pictures": rows})
            for row in rows:
                if row.get("smallest_pt") is not None and row["smallest_pt"] < MIN_PT:
                    small += 1
                    print(f"week {week:02d} slide {row['slide']:2d} {row['figure']}: smallest text {row['smallest_pt']:.1f} pt")
                elif "smallest_pt" not in row and row["figure"] != "cover":
                    print(f"week {week:02d} slide {row['slide']:2d} {row['figure']}: result image at scale {row['scale']:.2f} (check by eye)")
    (ROOT / "build" / "figure_text_audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{small} drawn pictures project text below {MIN_PT:.0f} pt")
    return 1 if small else 0


if __name__ == "__main__":
    sys.exit(main())
