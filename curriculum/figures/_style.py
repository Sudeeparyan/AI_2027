"""Shared matplotlib style for every week's figures (matches the slide palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

INK = "#1E1B4B"
PRIMARY = "#4F46E5"
TINT = "#EEF0FF"
ACCENT = "#EA580C"
TEAL = "#0F766E"
TEXT = "#1F2937"
MUTED = "#6B7280"
LAV = "#C7D2FE"
GRID = "#E5E7EB"
SERIES = [PRIMARY, ACCENT, TEAL, "#A21CAF", "#CA8A04", "#0369A1"]

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Calibri" if "Calibri" in _fonts else "DejaVu Sans"

plt.rcParams.update({
    # per-glyph fallback: symbols such as ⊙ ✓ ✗ are missing from Calibri
    "font.family": [FONT] + [f for f in ("Segoe UI Symbol", "DejaVu Sans") if f in _fonts or f == "DejaVu Sans"],
    "font.size": 13,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": INK,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "mathtext.fontset": "cm",
})


def save(fig, path, transparent: bool = False) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, transparent=transparent, facecolor="none" if transparent else "white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=TINT, ec=PRIMARY, color=INK, size=13, weight="bold", radius=0.02, lw=1.4):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
