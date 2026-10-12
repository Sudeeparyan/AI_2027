"""Tile the rendered Word pages of a week into readable contact sheets for review.

Run tools/render.py first; it writes build/render/week_XX/notes-NN.png (one image per Word page).
This script puts four pages on each sheet (2 x 2, each page 1000 px wide) so a reviewer can read
the text, and writes build/render/week_XX/notes_sheets/sheet_NN.png.

For a closer look at single pages, render the PDF at a higher resolution instead:
    pdftoppm -r 170 -f 12 -l 13 -png build/render/week_04/notes.pdf page

With --slides it tiles six slides per sheet in week_XX/slides_sheets/.
With --slide N it instead tiles slide N of every chosen week into one sheet, so a change made to the
same slide in all weeks (for example the lab slide, 44) can be checked at once:
build/render/slide_NN_weeks.png.

Usage: .venv/Scripts/python.exe tools/notes_sheets.py --weeks 6 [--per-sheet 4]
       .venv/Scripts/python.exe tools/notes_sheets.py --slides --weeks 6
       .venv/Scripts/python.exe tools/notes_sheets.py --weeks 1-12 --slide 44
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def parse_weeks(spec):
    weeks = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        weeks += list(range(int(a), int(b or a) + 1))
    return weeks


def tile_pages(pages, out, per=4, cols=2, cell_w=1000):
    """Tile already-rendered pages without changing any input image."""
    if per < 1 or cols < 1 or cell_w < 1:
        raise ValueError("per-sheet, columns and cell width must be positive")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    for s in range(0, len(pages), per):
        group = pages[s:s + per]
        ims = []
        for p in group:
            with Image.open(p) as source:
                ims.append(source.convert("RGB"))
        cell_h = max(round(im.height * cell_w / im.width) for im in ims)
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * 10, rows * (cell_h + 30) + 10), "#9CA3AF")
        draw = ImageDraw.Draw(sheet)
        for i, (p, im) in enumerate(zip(group, ims)):
            im = im.resize((cell_w, round(im.height * cell_w / im.width)), Image.LANCZOS)
            x, y = 10 + (i % cols) * (cell_w + 10), 10 + (i // cols) * (cell_h + 30)
            draw.text((x, y), p.stem, fill="black")
            sheet.paste(im, (x, y + 18))
        sheet.save(out / f"sheet_{s // per + 1:02d}.png")
    return len(pages), out


def sheets(week, per=4, cols=2, cell_w=1000):
    src = ROOT / "build" / "render" / f"week_{week:02d}"
    pages = sorted(p for p in src.glob("notes-*.png") if "sheet" not in p.stem)
    return tile_pages(pages, src / "notes_sheets", per, cols, cell_w)


def slide_sheets(week, per=6, cols=2, cell_w=1000):
    """Six slides per sheet by default, preserving numeric slide order."""
    src = ROOT / "build" / "render" / f"week_{week:02d}"
    pages = sorted((p for p in src.glob("slides-*.png") if p.stem[7:].isdigit()),
                   key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    return tile_pages(pages, src / "slides_sheets", per, cols, cell_w)


def slide_sheet(weeks, number, cols=3, cell_w=1000):
    """Slide `number` of each week side by side, labelled by week."""
    found = [(w, ROOT / "build" / "render" / f"week_{w:02d}" / f"slides-{number:02d}.png") for w in weeks]
    found = [(w, p) for w, p in found if p.exists()]
    if not found:
        raise FileNotFoundError(f"No rendered images found for slide {number}")
    ims = [Image.open(p).convert("RGB") for _, p in found]
    cell_h = max(round(im.height * cell_w / im.width) for im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * 10, rows * (cell_h + 30) + 10), "#9CA3AF")
    draw = ImageDraw.Draw(sheet)
    for i, ((w, _), im) in enumerate(zip(found, ims)):
        im = im.resize((cell_w, round(im.height * cell_w / im.width)), Image.LANCZOS)
        x, y = 10 + (i % cols) * (cell_w + 10), 10 + (i // cols) * (cell_h + 30)
        draw.text((x, y), f"week {w:02d} slide {number}", fill="black")
        sheet.paste(im, (x, y + 18))
    out = ROOT / "build" / "render" / f"slide_{number:02d}_weeks.png"
    sheet.save(out)
    return len(found), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="1")
    ap.add_argument("--per-sheet", type=int, help="images per sheet (Word: 4, slides: 6)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--slide", type=int, help="tile this slide of every chosen week instead of Word pages")
    mode.add_argument("--slides", action="store_true", help="tile six slides of each week per sheet")
    args = ap.parse_args()
    if args.slide:
        n, out = slide_sheet(parse_weeks(args.weeks), args.slide)
        print(f"slide {args.slide}: {n} weeks -> {out}")
        return
    for week in parse_weeks(args.weeks):
        if args.slides:
            n, out = slide_sheets(week, args.per_sheet if args.per_sheet is not None else 6)
        else:
            n, out = sheets(week, args.per_sheet if args.per_sheet is not None else 4)
        print(f"week {week:02d}: {n} {'slides' if args.slides else 'pages'} -> {out}")


if __name__ == "__main__":
    main()
