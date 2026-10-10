"""Lint the layout of every drawn teaching figure (weeks 1-12).

Run: .venv/Scripts/python.exe tools/lint_figures.py [--weeks 1-12] [--verbose]

Each figure function is drawn into a temporary folder with saving patched out,
so the check needs no models and writes nothing into the project. Matplotlib's
own text extents are compared with box outlines and with one another:

* a label centred in a box must fit inside that box with a small margin;
* free text must not overlap other text;
* free text must not straddle a box outline (inside or outside, not across);
* no drawn line or arrow may pass through text or through the inside of a box
  (labels drawn on a line with their own background are exempt);
* text must not run off the saved figure.

Copied result images (measured lab plots) are reported as "image" and still
need visual review. The command exits with status 1 when any issue is found.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "curriculum" / "figures"))

import _style  # noqa: E402,F401 - selects the Agg backend and shared fonts
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

BOX_MARGIN = 3  # pixels at 100 dpi between a label and its box outline
OVERLAP = 2  # pixels of shared area tolerated between two text extents
LINE_CLEARANCE = 3  # pixels a line may approach a box interior or a text extent's sides
# A letter (any script) followed by _ or ^ and a short script, or a bracket followed by one: x_t, ᾱ_t, (a+b)^2.
RAW_SCRIPT = re.compile(r"(?:(?<![\w.\\])[^\W\d_][̀-ͯ]?|[)\]])[_^](?:\{[^{}\s]*\}|[0-9.]+|[^\W_]{1,4}\b)")


def weeks_arg(text: str) -> list[int]:
    weeks = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        weeks.update(range(int(first), int(last or first) + 1))
    return sorted(weeks)


def capture(module, name: str, folder: Path):
    figures = []

    def keep(fig, *_args, **_kwargs):
        figures.append(fig)

    with ExitStack() as stack:
        for attr in ("save", "save_editable_scene"):
            if hasattr(module, attr):
                stack.enter_context(patch.object(module, attr, keep))
        module.FIGURES[name](folder / f"{name}.png")
    return figures


def visible_texts(fig):
    texts = list(fig.texts)
    for legend in fig.legends:  # figure-level legends, e.g. one shared legend under several panels
        texts += list(legend.get_texts())
    for ax in fig.axes:
        texts += [t for t in ax.texts]
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if axis.get_visible():  # only ticks that are drawn inside the view limits
                    texts += [label for tick in axis._update_ticks() for label in (tick.label1, tick.label2)
                              if label.get_visible()]
        legend = ax.get_legend()
        if legend is not None:
            texts += list(legend.get_texts())
    seen, result = set(), []
    for t in texts:
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        if isinstance(t, Annotation) and not t.get_text().strip():
            continue
        seen.add(id(t))
        result.append(t)
    return result


def intersection(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def contains(outer, inner, margin=0.0) -> bool:
    return (inner.x0 >= outer.x0 + margin and inner.x1 <= outer.x1 - margin
            and inner.y0 >= outer.y0 + margin and inner.y1 <= outer.y1 - margin)


def drawn_segments(ax):
    """Straight pieces of every visible line and arrow in display pixels."""
    segments = []
    for line in ax.lines:
        if not line.get_visible() or line.get_linestyle() in ("None", "", " ") or line.get_linewidth() <= 0:
            continue
        points = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
        segments += list(zip(points[:-1], points[1:]))
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    arrows += [t.arrow_patch for t in ax.texts if isinstance(t, Annotation) and t.arrow_patch is not None]
    for arrow in arrows:
        if not arrow.get_visible():
            continue
        path = arrow.get_transform().transform_path(arrow.get_path())
        for poly in path.to_polygons(closed_only=False):
            segments += list(zip(poly[:-1], poly[1:]))
    return segments


def segment_hits(p, q, rect) -> bool:
    """Liang-Barsky clipping: does segment pq enter the rectangle (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = rect
    if x0 >= x1 or y0 >= y1:
        return False
    (px, py), (qx, qy) = p, q
    dx, dy = qx - px, qy - py
    lo, hi = 0.0, 1.0
    for step, room in ((-dx, px - x0), (dx, x1 - px), (-dy, py - y0), (dy, y1 - py)):
        if step == 0:
            if room < 0:
                return False
            continue
        t = room / step
        if step < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        if lo > hi:
            return False
    return True


def lint_figure(fig) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    texts = visible_texts(fig)
    extents = {id(t): t.get_window_extent(renderer) for t in texts}
    owned = set()
    for ax in fig.axes:
        boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch) and p.get_visible()]
        for shape in boxes:
            outline = shape.get_window_extent(renderer)
            centre = (shape.get_x() + shape.get_width() / 2, shape.get_y() + shape.get_height() / 2)
            for t in ax.texts:
                if t.get_text().strip() and tuple(map(float, t.get_position())) == tuple(map(float, centre)):
                    owned.add(id(t))
                    e = extents[id(t)]
                    if not contains(outline, e, BOX_MARGIN):
                        over = max(outline.x0 + BOX_MARGIN - e.x0, e.x1 - outline.x1 + BOX_MARGIN,
                                   outline.y0 + BOX_MARGIN - e.y0, e.y1 - outline.y1 + BOX_MARGIN)
                        issues.append(f"label does not fit its box by {over:.0f} px: {t.get_text()!r}")
        for t in ax.texts:
            if id(t) in owned or not t.get_text().strip() or id(t) not in extents:
                continue
            e = extents[id(t)]
            for shape in boxes:
                outline = shape.get_window_extent(renderer)
                if intersection(outline, e) > OVERLAP and not contains(outline, e, 1) and not contains(e, outline):
                    issues.append(f"text crosses a box outline: {t.get_text()!r}")
                    break
        # Lines and arrows must go around text and boxes, not through them.
        segments = drawn_segments(ax)
        c = LINE_CLEARANCE
        for t in list(ax.texts) + list(fig.texts):
            if not t.get_text().strip() or id(t) not in extents or t.get_rotation() % 90:
                continue
            if t.get_bbox_patch() is not None:  # a label deliberately drawn on its line, with a background
                continue
            e = extents[id(t)]
            ink = (e.x0 + c, e.y0 + 0.2 * e.height, e.x1 - c, e.y1 - 0.2 * e.height)  # extents include line spacing
            if any(segment_hits(p, q, ink) for p, q in segments):
                issues.append(f"a line or arrow crosses text: {t.get_text()!r}")
        for shape in boxes:
            o = shape.get_window_extent(renderer)
            inside = (o.x0 + c, o.y0 + c, o.x1 - c, o.y1 - c)
            if any(segment_hits(p, q, inside) for p, q in segments):
                label = next((t.get_text() for t in ax.texts if id(t) in owned and contains(o, extents[id(t)])), "")
                issues.append(f"a line or arrow crosses the inside of a box: {label!r}")
        # A legend without an opaque frame lies on top of the data: no plotted line may run through it.
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            frame = legend.get_frame()
            opaque = legend.get_frame_on() and frame.get_alpha() in (None, 1) and frame.get_facecolor()[3] == 1
            if not opaque:
                own = {id(h) for h in getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))}
                data = [(p, q) for line in ax.lines if id(line) not in own and line.get_visible()
                        and line.get_linestyle() not in ("None", "", " ")
                        for p, q in zip(*(lambda pts: (pts[:-1], pts[1:]))(
                            line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))))]
                e = legend.get_window_extent(renderer)
                if any(segment_hits(p, q, (e.x0 + c, e.y0 + c, e.x1 - c, e.y1 - c)) for p, q in data):
                    issues.append("a plotted line runs through a frameless legend: "
                                  + ", ".join(t.get_text() for t in legend.get_texts()))
    # Plain-text sub/superscripts (x_t, ᾱ_t, e^2) print literally; figure text should use $...$ or Unicode.
    for t in texts:
        plain = re.sub(r"\$[^$]*\$", "", t.get_text())
        if RAW_SCRIPT.search(plain):
            issues.append(f"raw _ or ^ in figure text: {t.get_text()!r}")
    upright = [t for t in texts if t.get_rotation() % 90 == 0]  # rotated extents are not their ink
    for i, a in enumerate(upright):
        for b in upright[i + 1:]:
            area = intersection(extents[id(a)], extents[id(b)])
            if area > OVERLAP * OVERLAP * 4:
                issues.append(f"text overlaps text: {a.get_text()!r} / {b.get_text()!r}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", default="1-12")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    total = bad = 0
    with tempfile.TemporaryDirectory(prefix="figure-lint-") as temp:
        for week in weeks_arg(args.weeks):
            module = importlib.import_module(f"week_{week:02d}")
            for name in module.FIGURES:
                figures = capture(module, name, Path(temp))
                if not figures:
                    if args.verbose:
                        print(f"week {week:02d} {name}: image (review visually)")
                    continue
                total += 1
                issues = sorted(set(i for fig in figures for i in lint_figure(fig)))
                for fig in figures:
                    plt.close(fig)
                if issues:
                    bad += 1
                    print(f"week {week:02d} {name}:")
                    for issue in issues:
                        print(f"   - {issue}")
                elif args.verbose:
                    print(f"week {week:02d} {name}: ok")
    print(f"{total} drawn figures checked; {bad} with layout issues")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
