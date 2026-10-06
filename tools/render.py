"""Render a week's slides and teaching notes to PNG images for visual review (uses Microsoft Office).

Run: .venv/Scripts/python.exe tools/render.py --weeks 1 [--what slides,notes] [--dpi 60]
Images: build/render/week_XX/{slides,notes}-NN.png, plus contact sheets slides-sheet-N.png.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(week: int, what: list[str], dpi: int) -> None:
    src_dir = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*"))
    out = ROOT / "build" / "render" / f"week_{week:02d}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if "slides" in what:
        jobs.append(("slides", src_dir / f"Week_{week:02d}_Lecture_Slides.pptx", out / "slides.pptx"))
    if "notes" in what:
        jobs.append(("notes", src_dir / f"Week_{week:02d}_Teaching_Notes.docx", out / "notes.docx"))
    for _, src, dst in jobs:
        shutil.copy2(src, dst)
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools" / "render_office.ps1"), *[str(d) for _, _, d in jobs]], check=True, timeout=900)
    for name, _, dst in jobs:
        for old in out.glob(f"{name}-*.png"):
            old.unlink()
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(dst.with_suffix(".pdf")), str(out / name)], check=True)
        pages = sorted(out.glob(f"{name}-*.png"))
        print(f"week {week:02d} {name}: {len(pages)} pages -> {out}")
        contact_sheets(pages, out, prefix=name, per=12 if name == "slides" else 6, cols=3)


def contact_sheets(pages: list[Path], out: Path, prefix: str = "slides", per: int = 12, cols: int = 3) -> None:
    from PIL import Image, ImageDraw

    for k in range(0, len(pages), per):
        group = pages[k:k + per]
        ims = [Image.open(p) for p in group]
        w, h = ims[0].size
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w + (cols + 1) * 10, rows * (h + 30) + 10), "white")
        d = ImageDraw.Draw(sheet)
        for i, im in enumerate(ims):
            x, y = 10 + (i % cols) * (w + 10), 10 + (i // cols) * (h + 30)
            sheet.paste(im, (x, y + 20))
            d.text((x, y + 2), f"{prefix} {k + i + 1}", fill="black")
        sheet.save(out / f"{prefix}-sheet-{k // per + 1}.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="1")
    ap.add_argument("--what", default="slides,notes")
    ap.add_argument("--dpi", type=int, default=60)
    a = ap.parse_args()
    for w in sorted({w for part in a.weeks.split(",") for w in (range(1, 13) if part == "all" else range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))}):
        render(w, a.what.split(","), a.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
