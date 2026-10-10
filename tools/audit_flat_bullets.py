"""Find slide content that was flattened into loose bullets in the Word notes sources.

Two patterns were found in October 2026 (weeks 9-12):

1. A lab section whose bullets repeat the slide's `runtime` or `deliverable` text with no label
   ("- Colab T4 GPU or CPU. About 7 min ..."). Write "- **Runtime:** ..." and "- **Hand in:** ...".
2. Three or more short bullets without end punctuation in one section: the step labels of a flow or stat
   slide printed as their own bullets, each followed by its explanation ("- Input", "- Image, document, ...").
   Write one bullet per step: "- **Input:** image, document, ...".

Quiz options ("**A.** ...") and bold-labelled bullets are not reported. This audit must report nothing before a release.

Usage: .venv/Scripts/python.exe tools/audit_flat_bullets.py
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "curriculum"


def problems():
    found = []
    for week in range(1, 13):
        md = (ROOT / f"notes/week_{week:02d}.md").read_text(encoding="utf-8")
        slides = yaml.safe_load((ROOT / f"weeks/week_{week:02d}.yaml").read_text(encoding="utf-8"))["slides"]
        lines = md.splitlines()
        for s in slides:
            if s.get("type") != "lab":
                continue
            for key in ("runtime", "deliverable"):
                value = " ".join(str(s.get(key, "")).split())
                for i, line in enumerate(lines, 1):
                    if value and line.strip().lstrip("- ").strip() == value:
                        found.append(f"week {week:02d} notes line {i}: lab {key} without a label: {value[:70]}")
        section, bullets = None, []

        def flush():
            short = [b for _, b in bullets
                     if len(b.split()) <= 4 and not re.search(r"[.:;!?)]$", b) and not b.startswith("**")]
            if section and len(short) >= 3:
                found.append(f"week {week:02d} {section}: {len(short)} short bullets ({' | '.join(short[:5])})")

        for i, line in enumerate(lines, 1):
            if line.startswith("#"):
                flush()
                section, bullets = line.strip("# ").strip(), []
            elif line.startswith("- "):
                bullets.append((i, line[2:].strip()))
        flush()
    return found


if __name__ == "__main__":
    found = problems()
    for f in found:
        print(f)
    print(f"{len(found)} flattened bullet problem(s)")
    sys.exit(1 if found else 0)
