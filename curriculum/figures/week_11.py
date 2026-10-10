"""Week 11 figures: deployment diagrams plus REAL measurements harvested from the executed lab notebook."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from _style import role_box, role_legend, routed_arrow, save_editable_scene, ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_11"
ORANGE = "#FFF1E6"
GREEN = "#E6F4F1"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (label, c) in enumerate([("UI", PRIMARY), ("API", TEAL), ("Model", ACCENT)]):
        y = 0.72 - i * 0.27
        ax.add_patch(plt.Rectangle((0.18, y), 0.64, 0.2, color=c, alpha=0.9))
        ax.text(0.5, y + 0.1, label, ha="center", va="center", fontsize=28, color="white", fontweight="bold")
    ax.add_patch(plt.Circle((0.86, 0.12), 0.09, color=INK))
    ax.text(0.86, 0.12, "✓", ha="center", va="center", fontsize=30, color="white")
    save(fig, path, transparent=True)


def stack(path):
    fig, ax = _diagram_canvas(4.5)
    rows = [("Application", "Interface and request handling", "tool"),
            ("Orchestration", "Connect prompts, retrieval and tools", "tool"),
            ("Access / serving", "Hosted API, serving engine or local runner", "tool"),
            ("Model", "Weights and generation settings", "model"),
            ("Hardware", "Cloud, server or local device", "tool")]
    for i, (label, detail, role) in enumerate(rows):
        y = 3.66 - i * 0.72
        role_box(ax, 0.15, y, 3.35, 0.58, label, role, size=20)
        ax.text(3.9, y + 0.29, detail, fontsize=20, color=TEXT, va="center")
    # The boxes are layers, not pipeline roles, so a role key would mislabel them.
    ax.text(0.15, 0.25, "Monitor quality, latency, errors and cost across every layer.", fontsize=18, color=INK)
    _diagram_finish(fig, path)



def openai_compat(path):
    fig, ax = _diagram_canvas(4.7)
    role_box(ax, 0.15, 2.1, 3.4, 1.1, "Chat client\nmessages + model", "tool", size=20)
    backends = [("Hosted provider", "OpenAI / Azure / Gemini"),
                ("Inference router", "Hugging Face"),
                ("Serving engine", "vLLM / SGLang"),
                ("Local runner", "Ollama / llama.cpp"),
                ("Lab endpoint", "FastAPI on localhost")]
    for i, (label, example) in enumerate(backends):
        y = 3.8 - i * 0.68
        role_box(ax, 5.2, y, 3.2, 0.54, label, "tool", size=18)
        ax.text(8.8, y + 0.27, example, fontsize=18, color=TEXT, va="center")
        arrow(ax, 3.55, 2.65, 5.2, y + 0.27)
    ax.text(0.15, 0.35, "Set endpoint, key and model; verify supported fields and behaviour.", fontsize=18, color=INK)
    _diagram_finish(fig, path)



def request_anatomy(path):
    fig, ax = _diagram_canvas(2.7)
    stages = [("Network\nand auth", 0.15, 2.1, "tool"), ("Queue", 2.45, 1.5, "tool"),
              ("Prefill\nread prompt", 4.15, 2.5, "model"),
              ("Decode\ngenerate remaining tokens", 6.85, 5.85, "model")]
    for label, x, width, role in stages:
        role_box(ax, x, 1.5, width, 0.9, label, role, size=18)
    routed_arrow(ax, [(0.15, 1.12), (6.65, 1.12)], color=PRIMARY)
    ax.text(3.4, 0.75, "Wait to first token (TTFT)", ha="center", fontsize=18, color=PRIMARY)
    ax.text(7.05, 0.75, "Then continue streaming", fontsize=18, color=TEAL)
    _diagram_finish(fig, path)



def app_architecture(path):
    fig, ax = _diagram_canvas(4.1)
    ax.text(0.15, 3.65, "Proposed integrated design; the lab tests separate routes.", fontsize=20, color=INK)
    _sequence(ax, [("User", "data"), ("UI", "tool"), ("API", "tool"), ("Input\nchecks", "tool"),
                   ("Model", "model"), ("Output\nchecks", "tool"), ("Answer", "output")], 2.35, height=0.9, size=18)
    role_box(ax, 3.55, 0.9, 5.4, 0.95, "Logs, metrics and versions\nlatency / tokens / blocks", "output", size=18)
    role_box(ax, 9.45, 0.9, 3.25, 0.95, "Dashboard\nand alerts", "tool", size=18)
    arrow(ax, 8.95, 1.375, 9.45, 1.375)
    for x in (4.76, 6.61, 8.45):
        routed_arrow(ax, [(x, 2.35), (x, 1.85)], color=MUTED, dashed=True)
    role_legend(ax, 0.08, size=16)
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


def _measured_results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def latency_results(path):
    r = _measured_results()
    rows = r["latency"]
    fig, ax = plt.subplots(figsize=(13, 4.0))
    y = list(range(len(rows)))
    total = [row["total (s)"] for row in rows]
    first = [row["TTFT (s)"] for row in rows]
    ax.barh(y, total, color="#C7D2FE", label="Full answer")
    ax.barh(y, first, color=PRIMARY, label="First token")
    ax.set_yticks(y, ["What is an LLM?", "Three prompt tips", "What is a rate limit?"], fontsize=18)
    ax.invert_yaxis()
    for index, (t, f) in enumerate(zip(total, first)):
        ax.text(t + 0.1, index, f"{t:.2f} s total; {f:.3f} s first", fontsize=18, va="center", color=TEXT)
    ax.set_xlim(0, max(total) * 1.65)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Seconds", fontsize=20)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.23), fontsize=18, ncol=2)
    fig.tight_layout()
    save(fig, path)


def quant_results(path):
    rows = _measured_results()["quant"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    names = ["fp16\nreference", "8-bit\nper row", "4-bit\nper row", "4-bit\ngroups of 64"]
    colours = [MUTED, PRIMARY, ACCENT, TEAL]
    for ax, field, title in [(axes[0], "approx. size (MB)", "Hypothetical packed weight size (MB)"),
                              (axes[1], "perplexity", "Measured perplexity on one paragraph")]:
        vals = [row[field] for row in rows]
        bars = ax.bar(range(len(rows)), vals, color=colours)
        ax.bar_label(bars, labels=[f"{v:g}" for v in vals], fontsize=18, padding=4)
        ax.set_xticks(range(len(rows)), names, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title, fontsize=18, pad=14)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, path)


def batch_results(path):
    rows = _measured_results()["batch"]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    vals = [row["tokens/s"] for row in rows]
    bars = ax.bar(range(len(rows)), vals, color=TEAL)
    ax.bar_label(bars, labels=[f"{v:g}" for v in vals], fontsize=20, padding=4)
    ax.set_xticks(range(len(rows)), [row["batch size"] for row in rows], fontsize=20)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlabel("Batch size", fontsize=20)
    ax.set_ylabel("Generated tokens / second", fontsize=20)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, path)


def monitor_results(path):
    """Keep the historical per-request bars, whose rows were not saved in JSON."""
    image = plt.imread(ASSETS / "lab_monitor.png")
    if image.shape[:2] != (310, 886):
        raise ValueError("Recheck the historic monitoring-image crop")
    plot = image[31:258, 49:873]
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.imshow(plot, interpolation="none")
    ax.set_xticks([67 + 76.5 * i for i in range(10)], range(1, 11), fontsize=18)
    ax.set_yticks([226, 195, 164, 132, 101, 69, 38, 7], range(8), fontsize=18)
    ax.set_xlabel("Request number", fontsize=20)
    ax.set_ylabel("Latency (seconds)", fontsize=20)
    ax.grid(False)
    # Hide only the old small blocked/p95 labels over blank background.
    ax.add_patch(plt.Rectangle((210, 163), 21, 64, facecolor="white", edgecolor="none"))
    ax.add_patch(plt.Rectangle((800, 36), 22, 22, facecolor="white", edgecolor="none"))
    ax.text(220, 206, "blocked", fontsize=18, color="#B91C1C", rotation=90, ha="center")
    p95 = _measured_results()["monitor"]["p95 latency (s)"]
    ax.text(775, 42, f"p95: {p95:g} s", fontsize=18, color=TEXT, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PRIMARY, label="Answered"), Patch(color=ACCENT, label="Personal data redacted"),
                       Patch(color="#B91C1C", label="Blocked")], fontsize=18, loc="upper center",
              bbox_to_anchor=(0.5, 1.27), ncol=3)
    fig.tight_layout()
    save(fig, path)


FIGURES = {"cover": cover, "stack": stack, "openai_compat": openai_compat, "request_anatomy": request_anatomy,
           "app_architecture": app_architecture, "lab_latency": latency_results, "lab_quant": quant_results,
           "lab_batch": batch_results, "lab_monitor": monitor_results}
