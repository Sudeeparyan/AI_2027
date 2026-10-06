"""Week 8 figures: fine-tuning diagrams, plus REAL results harvested from the executed lab notebook
(curriculum/assets/week_08/results.json)."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_08"
ORANGE = "#FFF1E6"


def results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.05, 0.25), 0.55, 0.55, fc="#312E81", ec="#C7D2FE", lw=2))
    ax.text(0.325, 0.525, "W\nfrozen", ha="center", va="center", color="#C7D2FE", fontsize=20, fontweight="bold")
    ax.text(0.68, 0.52, "+", fontsize=34, color="white", va="center")
    ax.add_patch(plt.Rectangle((0.78, 0.25), 0.08, 0.55, fc=ACCENT, ec="white"))
    ax.add_patch(plt.Rectangle((0.9, 0.72), 0.55 * 0.18, 0.08, fc=TEAL, ec="white"))
    ax.text(0.82, 0.18, "B", ha="center", color="white", fontsize=18, fontweight="bold")
    ax.text(0.95, 0.85, "A", ha="center", color="white", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    save(fig, path, transparent=True)


def decision(path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    box(ax, 0.1, 2.1, 2.3, 0.9, "Start: prompt +\nevaluation set", fc=TINT, size=12)
    box(ax, 3.1, 2.1, 2.3, 0.9, "Meets the\nquality bar?", fc="white", size=12)
    box(ax, 6.2, 3.7, 2.6, 0.9, "Done: ship + monitor", fc="#E6F4F1", ec=TEAL, color=TEAL, size=12)
    box(ax, 6.2, 2.1, 2.6, 0.9, "Missing knowledge?\n(facts, documents)", fc="white", size=11.5)
    box(ax, 9.6, 2.1, 3.2, 0.9, "RAG: retrieve and\nground (week 12)", fc=TINT, size=12)
    box(ax, 6.2, 0.4, 2.6, 0.9, "Behaviour / format /\nstyle / cost gap?", fc="white", size=11.5)
    box(ax, 9.6, 0.4, 3.2, 0.9, "Fine-tune (LoRA) a\nsuitable model", fc=ORANGE, ec=ACCENT, size=12)
    arrow(ax, 2.4, 2.55, 3.1, 2.55)
    arrow(ax, 5.4, 2.8, 6.2, 4.1)
    ax.text(5.5, 3.6, "yes", color=TEAL, fontsize=12)
    arrow(ax, 5.4, 2.55, 6.2, 2.55)
    ax.text(5.55, 2.7, "no", color=ACCENT, fontsize=12)
    arrow(ax, 8.8, 2.55, 9.6, 2.55)
    arrow(ax, 7.5, 2.1, 7.5, 1.3)
    arrow(ax, 8.8, 0.85, 9.6, 0.85)
    ax.text(0.1, 4.6, "Try the cheapest adaptation first; combine them when needed", fontsize=14, fontweight="bold", color=INK)
    save(fig, path)


def rlhf(path):
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    steps = [("1. Supervised fine-tuning", "demonstrations of good answers", 0.1),
             ("2. Collect comparisons", "humans rank 2+ model answers per prompt", 3.35),
             ("3. Train a reward model", "predicts which answer people prefer", 6.6),
             ("4. Optimise the policy", "RL (PPO) to maximise reward − β·KL to the SFT model", 9.85)]
    for t, sub, x in steps:
        box(ax, x, 2.0, 3.0, 1.4, t, fc=TINT if "Optimise" not in t else ORANGE, ec=PRIMARY if "Optimise" not in t else ACCENT, size=12.5)
        ax.text(x + 1.5, 1.75, "\n".join(textwrap.wrap(sub, 26)), ha="center", va="top", fontsize=12, color=TEXT, linespacing=1.3)
    for x in (3.1, 6.35, 9.6):
        arrow(ax, x, 2.7, x + 0.25, 2.7)
    ax.text(0.1, 4.0, "RLHF as used for InstructGPT / early ChatGPT (Ouyang et al., 2022)", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.1, 0.35, "DPO skips steps 3–4: it optimises directly on the preference pairs with a classification-style loss.", fontsize=12.5, color=TEAL)
    save(fig, path)


def lora(path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.add_patch(plt.Rectangle((0.5, 0.6), 3.4, 3.4, fc=TINT, ec=PRIMARY, lw=2))
    ax.text(2.2, 2.3, "W\n(d × k)\nfrozen", ha="center", va="center", fontsize=16, color=INK, fontweight="bold")
    ax.text(4.3, 2.3, "+", fontsize=30, va="center", color=INK)
    ax.add_patch(plt.Rectangle((5.0, 0.6), 0.45, 3.4, fc=ORANGE, ec=ACCENT, lw=2))
    ax.text(5.22, 0.3, "B (d × r)", ha="center", fontsize=12, color=ACCENT)
    ax.text(5.75, 2.3, "×", fontsize=22, va="center", color=INK)
    ax.add_patch(plt.Rectangle((6.2, 3.55), 3.4, 0.45, fc=ORANGE, ec=ACCENT, lw=2))
    ax.text(7.9, 4.2, "A (r × k)", ha="center", fontsize=12, color=ACCENT)
    ax.text(10.0, 2.9, "Only A and B are trained.\nB starts at zero, so training\nbegins from the base model.", fontsize=12.5, color=TEXT, va="center")
    ax.text(10.0, 1.2, "d = k = 4096, r = 16:\nW has 16.8 M weights,\nA + B have 131 K (0.8%).", fontsize=12.5, color=TEAL, va="center")
    save(fig, path)


def param_compare(path):
    labels = ["Full fine-tuning", "LoRA r=16 (attention)", "LoRA r=8 (q, v only)"]
    d, L = 896, 24  # Qwen2.5-0.5B hidden size and layers (approximate illustration uses the real lab count below)
    try:
        r = results()
        total, trainable = r["total"], r["trainable"]
    except Exception:
        total, trainable = 494e6, 2.2e6
    vals = [total, trainable, trainable / 4]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.barh(labels[::-1], vals[::-1], color=[TEAL, ACCENT, PRIMARY])
    ax.set_xscale("log")
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, i, f"{v / 1e6:,.1f} M ({100 * v / total:.2f}%)", va="center", fontsize=12)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_title("Qwen2.5-0.5B: how many weights are trained?", loc="left")
    ax.set_xlim(1e5, total * 20)
    ax.grid(axis="y", visible=False)
    save(fig, path)


def results_bar(path):
    r = results()["comparison"]
    names = [x["model"] for x in r]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].bar(names, [x["accuracy"] for x in r], color=[MUTED, PRIMARY, ACCENT])
    ax[1].bar(names, [x["invalid"] for x in r], color=[MUTED, PRIMARY, ACCENT])
    top_invalid = max(0.1, max(x["invalid"] for x in r) * 1.4)  # its own scale, so small rates stay visible
    for a, key, t, top in ((ax[0], "accuracy", "Accuracy on held-out test messages", 1.1),
                           (ax[1], "invalid", "Invalid outputs (no valid intent)", top_invalid)):
        for i, x in enumerate(r):
            a.text(i, x[key] + 0.02 * top, f"{x[key]:.0%}" if key == "accuracy" else f"{x[key]:.1%}", ha="center", fontsize=12.5)
        a.set_ylim(0, top)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        a.set_title(t, loc="left")
        a.grid(axis="x", visible=False)
    save(fig, path)


def loss_curve(path):
    r = results()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(r["loss"]["step"], r["loss"]["loss"], color=PRIMARY, lw=2, label="training loss")
    if r.get("eval"):
        ax.plot(r["eval"]["step"], r["eval"]["eval_loss"], color=ACCENT, lw=2, marker="o", label="validation loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss on completion tokens")
    where = "a laptop CPU" if r["device"] == "cpu" else "a GPU"
    ax.set_title(f"LoRA SFT on {r['n_train']:,} examples ({r['train_minutes']:.0f} min on {where})", loc="left")
    ax.legend()
    save(fig, path)


FIGURES = {"cover": cover, "decision": decision, "rlhf": rlhf, "lora": lora, "param_compare": param_compare,
           "results_bar": results_bar, "loss_curve": loss_curve}
