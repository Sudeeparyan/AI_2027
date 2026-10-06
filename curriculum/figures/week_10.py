"""Week 10 figures: multimodal application diagrams plus REAL outputs harvested from the executed lab notebook."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

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
        ax.plot([x1, x2], [y1, y2], color="#C7D2FE", lw=2, alpha=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, path, transparent=True)


def mllm_types(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    a = axes[0]
    a.axis("off")
    a.set_xlim(0, 6.5)
    a.set_ylim(0, 4.4)
    a.set_title("Modular (bridge) MLLM", loc="left", fontsize=15)
    for i, (m, c) in enumerate([("image", TEAL), ("audio", ACCENT)]):
        y = 2.8 - i * 1.4
        box(a, 0.1, y, 1.2, 0.7, m, fc=TINT, size=11.5)
        box(a, 1.6, y, 1.5, 0.7, f"{m} encoder", fc="white", ec=c, color=c, size=10.5)
        box(a, 3.35, y, 0.9, 0.7, "proj", fc=ORANGE, ec=ACCENT, size=10.5)
        arrow(a, 1.3, y + 0.35, 1.6, y + 0.35)
        arrow(a, 3.1, y + 0.35, 3.35, y + 0.35)
        arrow(a, 4.25, y + 0.35, 4.7, 2.2)
    box(a, 4.7, 1.6, 1.7, 1.3, "LLM\n(text out)", fc="white", size=12)
    a.text(0.1, 0.35, "LLaVA, BLIP-2, Qwen-VL: pretrained parts joined; outputs text", fontsize=11, color=TEXT)
    b = axes[1]
    b.axis("off")
    b.set_xlim(0, 6.5)
    b.set_ylim(0, 4.4)
    b.set_title("Natively multimodal ('omni') model", loc="left", fontsize=15)
    toks = [("T", PRIMARY), ("T", PRIMARY), ("I", TEAL), ("I", TEAL), ("I", TEAL), ("A", ACCENT), ("A", ACCENT), ("T", PRIMARY)]
    for i, (t, c) in enumerate(toks):
        b.add_patch(plt.Rectangle((0.2 + i * 0.5, 3.0), 0.42, 0.42, color=c, alpha=0.85))
        b.text(0.41 + i * 0.5, 3.21, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    box(b, 0.9, 1.6, 3.4, 0.9, "one transformer trained on\ninterleaved text, image, audio (video)", fc=ORANGE, ec=ACCENT, size=11)
    arrow(b, 2.2, 3.0, 2.4, 2.5)
    box(b, 4.8, 1.6, 1.5, 0.9, "text, speech\n(or images)", fc="white", size=11)
    arrow(b, 4.3, 2.05, 4.8, 2.05)
    b.text(0.2, 0.35, "e.g. GPT, Gemini, Qwen-Omni families: real-time voice, any-to-any", fontsize=11, color=TEXT)
    save(fig, path)


def generation_map(path):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.2)
    rows = [("Text → image", "latent diffusion / flow transformers (week 4)", "Stable Diffusion, FLUX, Imagen, GPT image models"),
            ("Image → image", "noise-and-denoise editing, inpainting, control", "SDEdit, ControlNet, instruction-based editors"),
            ("Text → speech", "neural TTS: acoustic model + vocoder, or codec LMs", "VITS/MMS, Bark, Kokoro, voice cloning systems"),
            ("Speech → speech", "speech LLMs: listen and answer in voice, low latency", "real-time voice modes of assistants"),
            ("Text → music / audio", "diffusion or token LMs over audio codecs", "MusicGen, Stable Audio, Suno-style tools"),
            ("Text → video", "spatio-temporal diffusion transformers", "Sora, Veo families, open video models")]
    for i, (task, how, ex) in enumerate(rows):
        y = 4.75 - i * 0.84
        c = SERIES[i % len(SERIES)]
        box(ax, 0.05, y - 0.31, 2.55, 0.64, task, fc="white", ec=c, color=c, size=14)
        ax.text(2.85, y + 0.13, how, fontsize=14, color=TEXT, va="center")
        ax.text(2.85, y - 0.21, "e.g. " + ex, fontsize=12.5, color=MUTED, va="center")
    save(fig, path)


def vlm_training(path):
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.8)
    stages = [("1. Pre-train parts", "vision encoder (CLIP/SigLIP)\nand LLM, separately"),
              ("2. Alignment", "train the projector on\nimage–caption pairs"),
              ("3. Visual instruction\ntuning", "image-based conversations,\ncharts, documents, OCR"),
              ("4. Preference / RL", "reduce hallucination,\nimprove helpfulness"),
              ("5. Adapt (optional)", "LoRA on your domain\nimages (week 8)")]
    for i, (t, sub) in enumerate(stages):
        x = 0.1 + i * 2.6
        box(ax, x, 1.7, 2.3, 1.1, t, fc=TINT if i != 2 else ORANGE, ec=PRIMARY if i != 2 else ACCENT, size=11.5)
        ax.text(x + 1.15, 1.45, sub, ha="center", va="top", fontsize=10.5, color=TEXT)
        if i < 4:
            arrow(ax, x + 2.3, 2.25, x + 2.6, 2.25)
    save(fig, path)


def provenance(path):
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    layers = [("Visible label / disclosure", "'AI-generated' caption; required for deepfakes (EU AI Act Art. 50)", "easy to crop or omit"),
              ("Content credentials (C2PA)", "cryptographically signed history: tool, edits, author", "can be stripped; needs platform support"),
              ("Invisible watermark", "signal in pixels / audio / text (e.g. SynthID)", "only if the generator adds it; heavy edits weaken it"),
              ("Detection classifiers", "predict whether media is synthetic", "arms race; false positives on real media")]
    for i, (t, how, weak) in enumerate(layers):
        y = 3.75 - i * 1.0
        c = SERIES[i % len(SERIES)]
        box(ax, 0.05, y - 0.33, 3.05, 0.66, t, fc="white", ec=c, color=c, size=13.5)
        ax.text(3.3, y + 0.13, how, fontsize=13.5, color=TEXT, va="center")
        ax.text(3.3, y - 0.22, "✗ " + weak, fontsize=12.5, color="#B91C1C", va="center")
    save(fig, path)


FIGURES = {"cover": cover, "mllm_types": mllm_types, "generation_map": generation_map, "vlm_training": vlm_training,
           "provenance": provenance, **{n: asset(n) for n in ["lab_caption", "lab_chart", "lab_t2i", "lab_i2i"]}}
