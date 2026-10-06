"""Week 9 figures: multimodal diagrams plus REAL results from the executed lab notebook
(curriculum/assets/week_09/results.json and lab_*.png harvested by tools/harvest_figures.py)."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_09"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def cover(path):
    rng = np.random.default_rng(9)
    fig, ax = plt.subplots(figsize=(5, 5))
    a = rng.normal(0, 1, (60, 2)) * 0.5 + [-1, 0.6]
    b = rng.normal(0, 1, (60, 2)) * 0.5 + [1, -0.6]
    for i in range(0, 60, 4):
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="#C7D2FE", alpha=0.3, lw=1)
    ax.scatter(*a.T, s=60, color=PRIMARY, alpha=0.85)
    ax.scatter(*b.T, s=60, color=ACCENT, alpha=0.85, marker="s")
    ax.axis("off")
    save(fig, path, transparent=True)


def modalities(path):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    rows = [("Text", "sub-word tokens", "tokenizer → embeddings", PRIMARY),
            ("Image", "patches (e.g. 16 × 16 px)", "vision transformer (ViT)", TEAL),
            ("Audio", "spectrogram frames or codec tokens", "audio encoder (e.g. Whisper)", ACCENT),
            ("Video", "frames × patches (+ audio)", "spatio-temporal encoder", "#A21CAF")]
    for i, (m, unit, enc, c) in enumerate(rows):
        y = 3.9 - i * 1.1
        box(ax, 0.2, y, 1.8, 0.75, m, fc="white", ec=c, color=c, size=14)
        box(ax, 2.6, y, 3.4, 0.75, unit, fc=TINT, size=11.5, weight="normal")
        box(ax, 6.6, y, 3.4, 0.75, enc, fc=TINT, size=11.5, weight="normal")
        arrow(ax, 2.0, y + 0.37, 2.6, y + 0.37)
        arrow(ax, 6.0, y + 0.37, 6.6, y + 0.37)
        arrow(ax, 10.0, y + 0.37, 10.7, 2.2)
    box(ax, 10.7, 1.6, 2.1, 1.3, "vectors in a\nshared space /\nshared model", fc=ORANGE, ec=ACCENT, size=12)
    save(fig, path)


def fusion(path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    titles = ["Early fusion", "Late fusion", "Cross-attention fusion"]
    for a, t in zip(axes, titles):
        a.axis("off")
        a.set_xlim(0, 4)
        a.set_ylim(0, 4)
        a.set_title(t, loc="left", fontsize=16)
    a = axes[0]
    box(a, 0.1, 0.4, 1.4, 0.7, "image", fc=TINT, size=13)
    box(a, 2.4, 0.4, 1.4, 0.7, "text", fc=TINT, size=13)
    box(a, 0.6, 1.8, 2.8, 0.9, "one joint model", fc=ORANGE, ec=ACCENT, size=13.5)
    arrow(a, 0.8, 1.1, 1.5, 1.8)
    arrow(a, 3.1, 1.1, 2.5, 1.8)
    a.text(0.1, 3.75, "combine inputs or tokens first;\none model learns interactions", fontsize=12.5, color=TEXT, va="center")
    b = axes[1]
    for x, lab in ((0.1, "image"), (2.4, "text")):
        box(b, x, 0.4, 1.4, 0.7, lab, fc=TINT, size=13)
        box(b, x, 1.6, 1.4, 0.7, "encoder", fc="white", size=13)
        arrow(b, x + 0.7, 1.1, x + 0.7, 1.6)
    box(b, 0.9, 2.65, 2.2, 0.7, "combine scores", fc=ORANGE, ec=ACCENT, size=13)
    arrow(b, 0.8, 2.3, 1.5, 2.65)
    arrow(b, 3.1, 2.3, 2.5, 2.65)
    b.text(0.1, 3.75, "separate encoders; merge at the\nend (e.g. CLIP similarity)", fontsize=12.5, color=TEXT, va="center")
    c = axes[2]
    box(c, 0.1, 0.4, 1.4, 0.7, "image\nfeatures", fc=TINT, size=12)
    box(c, 2.4, 0.4, 1.4, 0.7, "text\ntokens", fc=TINT, size=12)
    box(c, 1.1, 1.9, 2.8, 0.9, "text queries attend\nto image keys/values", fc=ORANGE, ec=ACCENT, size=12)
    arrow(c, 0.8, 1.1, 1.5, 1.9)
    arrow(c, 3.1, 1.1, 2.8, 1.9)
    c.text(0.1, 3.75, "one modality conditions on the\nother inside the network (week 5)", fontsize=12.5, color=TEXT, va="center")
    save(fig, path)


def architectures(path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    specs = [("Dual encoder", "CLIP, SigLIP", "retrieval, zero-shot\nclassification"),
             ("Fusion encoder", "ViLBERT, single-stream\nvision-language BERTs", "VQA, grounding\n(understanding)"),
             ("Encoder–decoder", "BLIP captioning,\nWhisper, image-to-text", "generate text\nfrom another modality"),
             ("Vision encoder + LLM", "LLaVA, Qwen-VL,\nmost multimodal LLMs", "chat about images,\ndocuments, charts")]
    for a, (t, ex, use) in zip(axes, specs):
        a.axis("off")
        a.set_xlim(0, 3)
        a.set_ylim(0, 4)
        a.set_title(t, loc="left", fontsize=14)
        a.text(0.05, 3.3, ex, fontsize=11.5, color=INK, va="top", fontweight="bold")
        a.text(0.05, 2.1, use, fontsize=11, color=TEXT, va="top")
    a = axes[0]
    box(a, 0.1, 0.2, 1.2, 0.6, "img enc", fc=TINT, size=10)
    box(a, 1.6, 0.2, 1.2, 0.6, "txt enc", fc=TINT, size=10)
    a.annotate("", xy=(1.6, 0.5), xytext=(1.3, 0.5), arrowprops=dict(arrowstyle="<->", color=ACCENT))
    b = axes[1]  # one transformer over image and text tokens together
    for k, (lab, c) in enumerate([("img", TINT), ("img", TINT), ("txt", "white"), ("txt", "white")]):
        box(b, 0.1 + k * 0.7, 0.1, 0.6, 0.45, lab, fc=c, size=9)
    box(b, 0.1, 0.75, 2.7, 0.5, "one joint transformer", fc=ORANGE, ec=ACCENT, size=10)
    c_ = axes[2]  # encoder reads one modality, decoder writes text with cross-attention
    box(c_, 0.05, 0.2, 1.1, 0.6, "encoder", fc=TINT, size=10)
    box(c_, 1.7, 0.2, 1.25, 0.6, "text decoder", fc="white", size=10)
    c_.annotate("", xy=(1.7, 0.5), xytext=(1.15, 0.5), arrowprops=dict(arrowstyle="-|>", color=ACCENT))
    c_.text(1.42, 0.9, "cross-attn", ha="center", fontsize=9, color=ACCENT)
    d = axes[3]
    box(d, 0.05, 0.2, 0.8, 0.6, "ViT", fc=TINT, size=10)
    box(d, 1.0, 0.2, 0.7, 0.6, "proj", fc=ORANGE, ec=ACCENT, size=10)
    box(d, 1.85, 0.2, 1.1, 0.6, "LLM", fc="white", size=10)
    arrow(d, 0.85, 0.5, 1.0, 0.5)
    arrow(d, 1.7, 0.5, 1.85, 0.5)
    save(fig, path)


def clip_matrix(path):
    rng = np.random.default_rng(1)
    n = 6
    m = rng.normal(0, 0.08, (n, n)) + 0.15 + np.eye(n) * 0.55
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(m, cmap="Purples", vmin=0, vmax=0.8)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ACCENT, lw=3))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"text {j + 1}" for j in range(n)])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"image {j + 1}" for j in range(n)])
    ax.set_title("A batch of N image–text pairs: maximise the diagonal,\nminimise everything else (both rows and columns)", loc="left", fontsize=13)
    ax.grid(False)
    save(fig, path)


def zeroshot_pipeline(path):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    box(ax, 0.1, 2.7, 3.0, 1.0, "class names → prompts\n'a photo of a {cat}.'", fc=TINT, size=11.5)
    box(ax, 3.7, 2.7, 1.9, 1.0, "text\nencoder", fc="white", size=12)
    box(ax, 0.1, 0.5, 3.0, 1.0, "new image", fc=TINT, size=12)
    box(ax, 3.7, 0.5, 1.9, 1.0, "image\nencoder", fc="white", size=12)
    box(ax, 6.3, 1.4, 2.9, 1.3, "cosine similarity\nimage vs each class", fc=ORANGE, ec=ACCENT, size=12)
    box(ax, 9.9, 1.55, 2.9, 1.0, "predict the most\nsimilar class", fc="white", size=12)
    arrow(ax, 3.1, 3.2, 3.7, 3.2)
    arrow(ax, 3.1, 1.0, 3.7, 1.0)
    arrow(ax, 5.6, 3.2, 6.3, 2.3)
    arrow(ax, 5.6, 1.0, 6.3, 1.8)
    arrow(ax, 9.2, 2.05, 9.9, 2.05)
    save(fig, path)


def whisper_pipeline(path):
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.4)
    t = np.linspace(0, 1, 400)
    ax.plot(0.2 + 2.2 * t, 1.7 + 0.5 * np.sin(60 * t) * np.exp(-3 * (t - 0.5) ** 2), color=PRIMARY)
    ax.text(1.3, 0.6, "waveform (16 kHz)", ha="center", fontsize=11.5, color=TEXT)
    box(ax, 3.0, 1.1, 2.4, 1.2, "log-mel\nspectrogram", fc=TINT, size=12)
    box(ax, 6.0, 1.1, 2.2, 1.2, "transformer\nencoder", fc="white", size=12)
    box(ax, 8.8, 1.1, 2.2, 1.2, "decoder\n(cross-attention)", fc="white", size=12)
    box(ax, 11.5, 1.2, 1.4, 1.0, "text", fc=ORANGE, ec=ACCENT, size=13)
    for x1, x2 in ((2.5, 3.0), (5.4, 6.0), (8.2, 8.8), (11.0, 11.5)):
        arrow(ax, x1, 1.7, x2, 1.7)
    ax.text(3.0, 2.8, "Whisper: trained on 680,000 hours of weakly supervised web audio (Radford et al., 2022)", fontsize=12.5, color=INK)
    save(fig, path)


def zeroshot_results(path):
    r = results()["zero_shot"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = list(r.keys())
    vals = list(r.values())
    ax.barh(names[::-1], vals[::-1], color=[TEAL, PRIMARY, MUTED][: len(vals)])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=12.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"zero-shot accuracy on {results()['n_images']} CIFAR-10 test images (10 classes; chance = 10%)")
    ax.set_title("Only the text prompt changed", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, path)


FIGURES = {
    "cover": cover, "modalities": modalities, "fusion": fusion, "architectures": architectures, "clip_matrix": clip_matrix,
    "zeroshot_pipeline": zeroshot_pipeline, "whisper_pipeline": whisper_pipeline, "zeroshot_results": zeroshot_results,
    **{n: asset(n) for n in ["lab_similarity", "lab_confusion", "lab_retrieval", "lab_modality_gap", "lab_spectrogram"]},
}
