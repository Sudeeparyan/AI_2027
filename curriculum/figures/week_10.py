"""Week 10 figures: multimodal application diagrams plus REAL outputs harvested from the executed lab notebook."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import role_box, role_legend, routed_arrow, save_editable_scene, ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_10"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    icons = [("Aa", 0.2, 0.75, PRIMARY), ("▣", 0.75, 0.75, TEAL), ("♪", 0.2, 0.2, ACCENT), ("▶", 0.75, 0.2, "#A21CAF")]
    for t, x, y, c in icons:
        ax.add_patch(plt.Circle((x, y), 0.17, color=c, alpha=0.9))
        ax.text(x, y, t, ha="center", va="center", fontsize=34, color="white", fontweight="bold")
    for (x1, y1), (x2, y2) in [((0.2, 0.75), (0.75, 0.2)), ((0.75, 0.75), (0.2, 0.2)), ((0.2, 0.75), (0.75, 0.75)), ((0.2, 0.2), (0.75, 0.2))]:
        # Start and stop at the circle edges (radius 0.17) so no line runs under an icon.
        dx, dy = x2 - x1, y2 - y1
        k = 0.17 / (dx * dx + dy * dy) ** 0.5
        ax.plot([x1 + k * dx, x2 - k * dx], [y1 + k * dy, y2 - k * dy], color="#C7D2FE", lw=2, alpha=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, path, transparent=True)


def mllm_types(path):
    fig, ax = _diagram_canvas(3.1)
    ax.text(0.15, 2.82, "Modular bridge", fontsize=20, fontweight="bold", color=INK)
    ax.text(6.85, 2.82, "Joint multimodal training", fontsize=20, fontweight="bold", color=INK)
    labels = [("Read\nimage", "data"), ("Encode\npixels", "model"), ("Project\nfeatures", "model"), ("Predict\nwords", "model")]
    xs = [0.15, 1.83, 3.51, 5.19]
    for x, (label, role) in zip(xs, labels):
        role_box(ax, x, 1.45, 1.25, 0.9, label, role, size=18)
    for i, label in enumerate(("pixels", "features", "embeddings")):
        _labelled_flow(ax, xs[i]+1.25, xs[i+1], 1.9, label)
    role_box(ax, 6.85, 1.45, 2.45, 0.9, "Read supported\nmedia", "data", size=18)
    role_box(ax, 10.08, 1.45, 2.55, 0.9, "Generate supported\noutput", "model", size=18)
    _labelled_flow(ax, 9.3, 10.08, 1.9, "media")
    ax.text(0.15, 0.91, "Separate pretrained parts", fontsize=18, color=TEXT)
    ax.text(6.85, 0.91, "Checkpoint-specific routes", fontsize=18, color=TEXT)
    _compact_legend(ax, ("data", "model"), 0.3)
    _diagram_finish(fig, path)



def generation_map(path):
    fig, ax = _diagram_canvas(4.7)
    rows = [("Text→image", "Denoise image"),
            ("Image→image", "Noise; denoise"),
            ("Text→speech", "Generate waveform"),
            ("Speech→speech", "Listen; reply"),
            ("Text→music", "Generate audio"),
            ("Text→video", "Generate spacetime")]
    for x, heading in ((0.3, "Route"), (5.5, "Operation")):
        ax.text(x, 4.38, heading, fontsize=20, color=INK, fontweight="bold")
    for i, (route, operation) in enumerate(rows):
        y=3.76-i*0.61
        ax.axhspan(y-0.20, y+0.29, color=TINT if i%2==0 else "white", zorder=0)
        ax.text(0.3, y, route, fontsize=20, color=INK, va="center")
        ax.text(5.5, y, operation, fontsize=20, color=TEXT, va="center")
    _diagram_finish(fig, path)



def vlm_training(path):
    fig, ax = _diagram_canvas(2.75)
    labels=[("Load parts\npretrained", "model"), ("Align bridge\ncaptions", "loss"),
            ("Tune instructions\nquestions", "loss"), ("Adapt domain\noptional", "loss")]
    xs=[0.15, 3.39, 6.63, 9.87]
    for x,(label,role) in zip(xs, labels):
        role_box(ax, x, 1.15, 2.75, 1.02, label, role, size=19)
    for i,label in enumerate(("weights", "aligned weights", "tuned weights")):
        _labelled_flow(ax, xs[i]+2.75, xs[i+1], 1.66, label)
    ax.text(0.15, 0.77, "Recipe varies; preference tuning optional", fontsize=18, color=TEXT)
    _compact_legend(ax, ("model", "loss"), 0.28)
    _diagram_finish(fig, path)



def provenance(path):
    fig, ax = _diagram_canvas(3.2)
    rows=[("Disclosure", "Inform viewers", "Cropped"),
          ("C2PA credentials", "Sign history", "Not truth"),
          ("Watermark", "Embed signal", "Weakened"),
          ("Detector", "Estimate origin", "Mistakes")]
    for x,heading in ((0.3,"Layer"),(4.1,"Operation"),(9.2,"Limit")):
        ax.text(x,2.84,heading,fontsize=20,fontweight="bold",color=INK)
    for i,(layer,operation,limit) in enumerate(rows):
        y=2.28-i*0.57
        ax.axhspan(y-0.20,y+0.24,color=TINT if i%2==0 else "white",zorder=0)
        ax.text(0.3,y,layer,fontsize=20,color=INK,va="center")
        ax.text(4.1,y,operation,fontsize=20,color=TEXT,va="center")
        ax.text(9.2,y,limit,fontsize=20,color=ACCENT,va="center")
    _diagram_finish(fig,path)




def _diagram_canvas(height):
    fig, ax = plt.subplots(figsize=(13, height))
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.04, top=0.97)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, height)
    return fig, ax


def _sequence(ax, labels, y, width=12.6, height=0.95, size=18):
    gap = 0.28
    block = (width - gap * (len(labels) - 1)) / len(labels)
    for i, (label, role) in enumerate(labels):
        x = 0.15 + i * (block + gap)
        role_box(ax, x, y, block, height, label, role, size=size)
        if i < len(labels) - 1:
            arrow(ax, x + block, y + height / 2, x + block + gap, y + height / 2)


def _diagram_finish(fig, path):
    save_editable_scene(fig, path)



def _labelled_flow(ax, start, end, y, label):
    """Route data above the boxes so arrow labels remain readable."""
    routed_arrow(ax, [(start,y),(start+0.06,y),(start+0.06,y+0.54),(end-0.06,y+0.54),(end-0.06,y),(end,y)],
                 label=label,label_xy=((start+end)/2,y+0.65),size=17)


def _compact_legend(ax, roles, y):
    colours={"data":"#0072B2","model":"#009E73","loss":"#D55E00"}
    names={"data":"Data","model":"Model","loss":"Loss"}
    for i,role in enumerate(roles):
        x=0.2+1.6*i
        ax.scatter([x],[y],marker="s",s=64,color=colours[role])
        ax.text(x+0.17,y,names[role],fontsize=17,color=TEXT,va="center")
    x=0.2+1.6*len(roles)
    routed_arrow(ax,[(x,y),(x+0.65,y)])
    ax.text(x+0.85,y,"Flow",fontsize=17,color=TEXT,va="center")


FIGURES = {"cover": cover, "mllm_types": mllm_types, "generation_map": generation_map, "vlm_training": vlm_training,
           "provenance": provenance, **{n: asset(n) for n in ["lab_caption", "lab_chart", "lab_t2i", "lab_i2i"]}}
