"""Week 7 figures: prompt anatomy, context engineering, plus REAL results harvested from the executed lab notebook
(curriculum/assets/week_07/results.json, written by the solution notebook when GENAI_RESULTS_PATH is set)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_07"
ORANGE = "#FFF1E6"


def results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    lines = ["<system>", "  role · rules", "</system>", "<context>", "  documents", "</context>", "<examples> …", "task:", "format: JSON"]
    for i, t in enumerate(lines):
        ax.text(0.05, 0.92 - i * 0.1, t, fontsize=17, family="monospace", color=[PRIMARY, "#C7D2FE", PRIMARY, TEAL, "#C7D2FE", TEAL, ACCENT, "white", "white"][i])
    save(fig, path, transparent=True)


def anatomy(path):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    parts = [("System / developer message", "Role, rules, tone, safety, output policy", PRIMARY, 4.2),
             ("Context", "Documents, data, conversation history, tool results (clearly delimited)", TEAL, 3.35),
             ("Examples (optional)", "1–5 input → output pairs showing the exact format", ACCENT, 2.5),
             ("Task / question", "What to do, for whom, with which constraints", INK, 1.65),
             ("Output format", "Length, structure, JSON schema, what to do if unsure", "#A21CAF", 0.8)]
    for title, sub, c, y in parts:
        box(ax, 0.2, y, 3.6, 0.7, title, fc="white", ec=c, color=c, size=12.5)
        ax.text(4.1, y + 0.35, sub, fontsize=13, color=TEXT, va="center")
    ax.text(0.2, 5.0, "A well-structured prompt (order is a design choice; long documents often go before the question)", fontsize=13.5, fontweight="bold", color=INK)
    save(fig, path)


def context_eng(path):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    srcs = [("instructions", 3.8), ("user query", 3.1), ("retrieved docs", 2.4), ("memory / history", 1.7), ("tool outputs", 1.0)]
    for t, y in srcs:
        box(ax, 0.2, y, 2.4, 0.55, t, fc=TINT, size=11.5)
        arrow(ax, 2.6, y + 0.27, 4.0, 2.4)
    box(ax, 4.0, 1.6, 2.6, 1.6, "select\ncompress\norder", fc=ORANGE, ec=ACCENT, size=13)
    ax.add_patch(plt.Rectangle((7.3, 0.7), 2.6, 3.4, fc="white", ec=PRIMARY, lw=2))
    for i, (t, c) in enumerate([("system", PRIMARY), ("docs", TEAL), ("history", MUTED), ("question", INK), ("", "white")]):
        ax.add_patch(plt.Rectangle((7.45, 3.55 - i * 0.62), 2.3, 0.5, fc=c, alpha=0.18 if c != "white" else 0))
        ax.text(8.6, 3.8 - i * 0.62, t, ha="center", va="center", fontsize=11.5, color=INK)
    ax.text(8.6, 4.3, "context window (limited)", ha="center", fontsize=12, color=PRIMARY)
    arrow(ax, 6.6, 2.4, 7.3, 2.4)
    box(ax, 10.6, 1.8, 2.2, 1.2, "LLM", fc="white", size=15)
    arrow(ax, 9.9, 2.4, 10.6, 2.4)
    save(fig, path)


def _bar(ax, names, vals, title, color, fmt="{:.0%}", ylim=(0, 1.05), err=None):
    ax.bar(names, vals, color=color, yerr=err, capsize=6, error_kw={"ecolor": "#475569", "elinewidth": 1.2})
    top = ylim[1] if ylim else max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + (err[i] if err else 0) + 0.02 * top, fmt.format(v), ha="center", fontsize=12)
    ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=12)


def _margin(p, n):
    """Normal-approximation 95% margin of error for an accuracy p measured on n items."""
    return 1.96 * (p * (1 - p) / n) ** 0.5


def fewshot_results(path):
    r = results()["classification"]
    names = [x["prompt"] for x in r]
    acc = [x["accuracy"] for x in r]
    err = [_margin(a, x["n"]) for a, x in zip(acc, r)]
    invalid = [x.get("invalid", 0) for x in r]
    if any(invalid):  # only worth a panel when some answers were not valid labels
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        _bar(ax[1], names, invalid, "Invalid answers (not one of the 5 labels)", ACCENT, fmt="{:.0f}", ylim=(0, max(invalid) * 1.25))
        ax = ax[0]
    else:
        fig, ax = plt.subplots(figsize=(10, 4.2))
    _bar(ax, names, acc, f"Accuracy on {r[0]['n']} student-services messages (bars: 95% margin of error)", PRIMARY, ylim=(0, 1.15), err=err)
    save(fig, path)


def cot_results(path):
    r = results()["reasoning"]
    names = [x["prompt"] for x in r]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    _bar(ax[0], names, [x["accuracy"] for x in r], f"Accuracy on {r[0]['n']} word problems", TEAL)
    toks = [x["avg_tokens"] for x in r]
    _bar(ax[1], names, toks, "Output tokens per problem (cost)", ACCENT, fmt="{:.0f}", ylim=(0, max(toks) * 1.2))
    save(fig, path)


def injection_results(path):
    r = results()["injection"]
    names = sorted(r, key=lambda k: k != "naive")  # naive prompt first, then the defended one
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _bar(ax, [f"{k} prompt" for k in names], [r[k] for k in names], "Attack success rate on injected documents", "#B91C1C", ylim=(0, 1.15))
    ax.tick_params(axis="x", rotation=0, labelsize=13)
    save(fig, path)


FIGURES = {"cover": cover, "anatomy": anatomy, "context_eng": context_eng, "fewshot_results": fewshot_results,
           "cot_results": cot_results, "injection_results": injection_results}
