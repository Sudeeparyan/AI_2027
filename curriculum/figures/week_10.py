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
    fig, ax = _diagram_canvas(2.9)
    ax.text(0.15, 2.52, "Modular bridge", fontsize=20, fontweight="bold", color=INK)
    ax.text(6.65, 2.52, "Joint multimodal training", fontsize=20, fontweight="bold", color=INK)
    left = [("Image", "data"), ("Encoder", "model"), ("Bridge", "model"), ("LLM", "model")]
    for i, (label, role) in enumerate(left):
        x = 0.15 + i * 1.5
        role_box(ax, x, 1.35, 1.2, 0.85, label, role, size=18)
        if i < 3:
            arrow(ax, x + 1.2, 1.775, x + 1.5, 1.775)
    role_box(ax, 6.65, 1.35, 2.4, 0.85, "Supported\ninput types", "data", size=18)
    role_box(ax, 9.45, 1.35, 3.2, 0.85, "Multimodal model", "model", size=18)
    arrow(ax, 9.05, 1.775, 9.45, 1.775)
    ax.text(0.15, 0.85, "LLaVA joins pretrained vision and language parts.", fontsize=16, color=TEXT)
    ax.text(6.65, 0.85, "Inputs and outputs depend on the checkpoint.", fontsize=16, color=TEXT)
    role_legend(ax, 0.08, size=16)
    _diagram_finish(fig, path)



def generation_map(path):
    fig, ax = _diagram_canvas(4.7)
    rows = [("Text to image", "Diffusion / flow models", "SD-Turbo, FLUX"),
            ("Image editing", "Noise, denoise, control", "SDEdit, ControlNet"),
            ("Text to speech", "Acoustics + waveform", "VITS / MMS"),
            ("Speech to speech", "Listen and respond", "Voice assistants"),
            ("Text to audio", "Diffusion or codec tokens", "MusicGen, Stable Audio"),
            ("Text to video", "Model space and time", "Video diffusion models")]
    # A table drawn as a diagram: column headings instead of a role key, since every box is a task.
    for x, head in ((0.15, "Task"), (3.5, "How it works"), (8.6, "Examples")):
        ax.text(x, 4.42, head, fontsize=18, fontweight="bold", color=INK, va="center")
    for i, (task, how, example) in enumerate(rows):
        y = 3.6 - i * 0.62
        role_box(ax, 0.15, y, 3.0, 0.48, task, "data", size=18)
        ax.text(3.5, y + 0.24, how, fontsize=18, color=TEXT, va="center")
        ax.text(8.6, y + 0.24, example, fontsize=18, color=INK, va="center")
    _diagram_finish(fig, path)



def vlm_training(path):
    fig, ax = _diagram_canvas(2.55)
    _sequence(ax, [("1. Pretrained\nvision + language", "model"), ("2. Train bridge\npaired captions", "loss"),
                   ("3. Instruction\nimage questions", "loss"), ("4. Preferences\noptional", "loss"),
                   ("5. Domain tuning\noptional", "loss")], 1.0, height=1.05, size=17)
    ax.text(0.15, 0.62, "Illustrative recipe: check which components each method trains.", fontsize=18, color=TEXT)
    role_legend(ax, 0.08, size=16)
    _diagram_finish(fig, path)



def provenance(path):
    fig, ax = _diagram_canvas(3.0)
    rows = [("Visible disclosure", "Tells the viewer", "May be cropped"),
            ("Signed C2PA history", "Records origin and edits", "Does not prove truth"),
            ("Invisible watermark", "Signals synthetic content", "May weaken after edits"),
            ("Detection classifier", "Estimates synthetic origin", "Can make mistakes")]
    for x, head in ((0.15, "Layer"), (4.1, "What it does"), (9.0, "Limit")):
        ax.text(x, 2.72, head, fontsize=18, fontweight="bold", color=INK, va="center")
    for i, (label, purpose, limit) in enumerate(rows):
        y = 1.98 - i * 0.6
        role_box(ax, 0.15, y, 3.6, 0.44, label, "tool", size=18)
        ax.text(4.1, y + 0.22, purpose, fontsize=18, color=TEXT, va="center")
        ax.text(9.0, y + 0.22, limit, fontsize=18, color=ACCENT, va="center")
    _diagram_finish(fig, path)




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


FIGURES = {"cover": cover, "mllm_types": mllm_types, "generation_map": generation_map, "vlm_training": vlm_training,
           "provenance": provenance, **{n: asset(n) for n in ["lab_caption", "lab_chart", "lab_t2i", "lab_i2i"]}}
