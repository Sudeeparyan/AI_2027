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
# Okabe–Ito colours, paired with readable dark text and pale backgrounds.
# Match tools/technical_diagrams.js. Role labels supplement colour.
ROLE_COLORS = {"data": "#0072B2", "model": "#009E73", "loss": "#D55E00", "output": "#CC79A7", "tool": "#626B73"}
ROLE_FILLS = {"data": "#E4F2FB", "model": "#E5F7EF", "loss": "#FFF0E6", "output": "#F9EAF4", "tool": "#F2F4F5"}

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
    from pathlib import Path
    fig.savefig(Path(path).with_suffix(".svg"), bbox_inches="tight", pad_inches=0.15, transparent=transparent, facecolor="none" if transparent else "white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=TINT, ec=PRIMARY, color=INK, size=13, weight="bold", radius=0.02, lw=1.4):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight, wrap=True)
    # Remember which roles this diagram draws so role_legend lists only those.
    role = next((r for r, c in ROLE_COLORS.items() if c == ec), None)
    used = ax.__dict__.setdefault("_roles_used", [])
    if role and role not in used:
        used.append(role)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))


def role_box(ax, x, y, w, h, text, role="model", size=18, weight="bold"):
    box(ax, x, y, w, h, text, fc=ROLE_FILLS[role], ec=ROLE_COLORS[role], size=size, weight=weight,
        radius=0.08 if role in ("data", "loss", "output") else 0)


def role_legend(ax, y, width=13, size=16):
    labels = {"data": "Input data", "model": "Model", "loss": "Loss / update", "output": "Output", "tool": "Human / tool"}
    # List only the roles drawn so far; a key entry with no matching box would only confuse readers.
    # A legend may sit on its own axes under several panels, so collect roles from the whole figure.
    used = {r for a in ax.figure.axes for r in getattr(a, "_roles_used", [])}
    if len(used) == 1:
        return  # every box has the same role (a timeline or a list), so a one-entry key explains nothing
    if used:
        labels = {role: label for role, label in labels.items() if role in used}
    for index, (role, label) in enumerate(labels.items()):
        x = 0.1 + index * width / 5
        role_box(ax, x, y, 0.22, 0.22, "", role, size=size)
        ax.text(x + 0.34, y + 0.11, label, fontsize=size, color=TEXT, va="center")


def routed_arrow(ax, points, color=PRIMARY, dashed=False, label=None, label_xy=None, size=17):
    """Orthogonal route with an arrow only on the final segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.8, linestyle="--" if dashed else "-")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0,
                                linestyle="--" if dashed else "-"))
    if label is not None:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=size, color=TEXT,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def save_editable_scene(fig, path):
    """Serialize block schematics, then draw PNG/SVG from their native scene.

    This deliberately excludes charts and images. Portable diagrams and their
    editable PowerPoint counterparts are generated from the same JSON objects.
    """
    import json
    import subprocess
    from pathlib import Path
    from matplotlib.colors import to_hex
    from matplotlib.patches import Circle, FancyBboxPatch
    from matplotlib.text import Annotation

    # Arial is available to both Office and the SVG rasterizer. Measure the
    # same face before serializing so a fallback font cannot widen a label.
    axis_labels=lambda ax: [*ax.texts,ax.title,ax._left_title,ax._right_title]
    for label in [*fig.texts, *(t for ax in fig.axes for t in axis_labels(ax))]:
        label.set_fontfamily("Arial")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.get_size_inches() * 120
    items = []
    colour = lambda value: to_hex(value, keep_alpha=False)[1:].upper()

    def xy(ax, p):
        u, v = fig.transFigure.inverted().transform(ax.transData.transform(p))
        return [float(u * width), float((1 - v) * height)]

    for ax in fig.axes:
        for shape in (s for s in ax.patches if isinstance(s,FancyBboxPatch)):
            centre=(shape.get_x()+shape.get_width()/2,shape.get_y()+shape.get_height()/2)
            labels=[t for t in ax.texts if t.get_position()==centre and t.get_text()]
            for label in labels:
                bounds=label.get_window_extent(renderer)
                lo,hi=ax.transData.transform([(shape.get_x(),shape.get_y()),
                                              (shape.get_x()+shape.get_width(),shape.get_y()+shape.get_height())])
                if bounds.width>hi[0]-lo[0]-3 or bounds.height>hi[1]-lo[1]-3:
                    raise ValueError(f"{Path(path).stem}: label exceeds block: {label.get_text()!r}")
        for line in ax.lines:
            xs, ys = line.get_data()
            if len(xs) > 1:
                items.append(dict(kind="route", points=[xy(ax,p) for p in zip(xs,ys)],
                                  color=colour(line.get_color()), dashed=line.get_linestyle() == "--", arrow=False))
        for label in axis_labels(ax):
            if isinstance(label, Annotation) and label.arrow_patch is not None:
                props = label.arrowprops or {}
                items.append(dict(kind="route", points=[xy(ax,label.xyann),xy(ax,label.xy)],
                                  color=colour(props.get("color",MUTED)), dashed=props.get("linestyle") == "--"))
        for shape in ax.patches:
            if isinstance(shape,Circle):
                cx,cy=shape.center; radius=shape.radius
                start=xy(ax,(cx-radius,cy+radius)); end=xy(ax,(cx+radius,cy-radius)); kind="ellipse"
            elif isinstance(shape,FancyBboxPatch):
                start=xy(ax,(shape.get_x(),shape.get_y()+shape.get_height()))
                end=xy(ax,(shape.get_x()+shape.get_width(),shape.get_y()))
                kind="roundRect" if shape.get_boxstyle().rounding_size > 0 else "rect"
            else:
                continue
            items.append(dict(kind="block",x=start[0],y=start[1],w=end[0]-start[0],h=end[1]-start[1],
                              shape=kind,fill=colour(shape.get_facecolor()),color=colour(shape.get_edgecolor())))
        for label in axis_labels(ax):
            if not label.get_text():
                continue
            bounds=label.get_window_extent(renderer)
            size=float(label.get_fontsize()*120/72)
            rows=label.get_text().count("\n")+1
            # Office reserves a little more line width than Matplotlib's ink
            # bounds. Keep that allowance in the shared scene so a single
            # label cannot wrap only in the editable PowerPoint version.
            w=float(bounds.width/fig.dpi*120*1.035+size*.45)
            h=float(max(bounds.height/fig.dpi*120,size*1.16*rows))
            u,v=fig.transFigure.inverted().transform(label.get_transform().transform(label.get_position()))
            pos=[float(u*width),float((1-v)*height)]; ha=label.get_ha(); va=label.get_va()
            x=pos[0]-(w/2 if ha=="center" else w if ha=="right" else 0)
            y=pos[1]-(h/2 if va in ("center","center_baseline") else h if va in ("bottom","baseline") else 0)
            text_item=dict(kind="text",text=label.get_text(),x=x,y=y,w=w,h=h,size=size,
                              color=colour(label.get_color()),bold=label.get_weight() in ("bold",700),
                              align={"center":"center","right":"right"}.get(ha,"left"))
            if label.get_bbox_patch() is not None:
                text_item["background"]=colour(label.get_bbox_patch().get_facecolor())
            items.append(text_item)
    for label in fig.texts:
        bounds=label.get_window_extent(renderer)
        size=float(label.get_fontsize()*120/72)
        w=float(bounds.width/fig.dpi*120*1.035+size*.45)
        h=float(max(bounds.height/fig.dpi*120,size*1.16*(label.get_text().count("\n")+1)))
        u,v=fig.transFigure.inverted().transform(label.get_transform().transform(label.get_position()))
        ha,va=label.get_ha(),label.get_va()
        x=float(u*width)-(w/2 if ha=="center" else w if ha=="right" else 0)
        y=float((1-v)*height)-(h/2 if va=="center" else h if va in ("bottom","baseline") else 0)
        items.append(dict(kind="text",text=label.get_text(),x=x,y=y,w=w,h=h,size=size,
                          color=colour(label.get_color()),bold=label.get_weight() in ("bold",700),
                          align={"center":"center","right":"right"}.get(ha,"left")))
    scene=dict(width=float(width),height=float(height),items=items)
    target=Path(path); scene_path=target.with_suffix(".scene.json")
    scene_path.write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding="utf-8")
    root=Path(__file__).resolve().parents[2]
    subprocess.run(["node",str(root/"tools"/"technical_diagrams.js"),"--scene",str(scene_path),str(target)],check=True)
    plt.close(fig)
