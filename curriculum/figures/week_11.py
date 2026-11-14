"""Week 11 figures: deployment diagrams plus REAL measurements harvested from the executed lab notebook."""
import json
from datetime import datetime
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
    """Responsibility comparison, not a falsely connected data pipeline."""
    fig, ax = _diagram_canvas(4.5)
    rows = [("Application", "Receive and return requests", "tool"),
            ("Coordination", "Connect evidence and tools", "tool"),
            ("Serving", "Run inference for clients", "tool"),
            ("Model", "Predict continuation tokens", "model"),
            ("Hardware", "Execute numerical operations", "tool")]
    for i, (label, detail, role) in enumerate(rows):
        y = 3.66 - i * 0.65
        role_box(ax, 0.15, y, 3.35, 0.54, label, role, size=20)
        ax.text(3.9, y + 0.27, detail, fontsize=20, color=TEXT, va="center")
    ax.text(0.15, 0.54, "Roles share monitoring responsibility.", fontsize=18, color=INK)
    role_legend(ax, 0.10, size=16)
    _diagram_finish(fig, path)


def openai_compat(path):
    fig, ax = _diagram_canvas(3.6)
    _sequence(ax, [("Messages", "data"), ("Configure endpoint", "tool"),
                   ("Generate", "model"), ("Reply", "output")], 1.55,
              labels=["messages", "JSON", "answer + usage"])
    ax.text(0.15, 3.1, "Change URL, key and model; verify fields.", fontsize=18, color=INK)
    ax.text(0.15, 0.90, "Alternatives: hosted APIs, vLLM, Ollama, FastAPI.", fontsize=16, color=TEXT)
    role_legend(ax, 0.10, size=16)
    _diagram_finish(fig, path)


def request_anatomy(path):
    fig, ax = _diagram_canvas(3.1)
    _sequence(ax, [("Network / auth", "tool"), ("Queue", "tool"),
                   ("Prefill", "model"), ("Decode", "model")], 1.65,
              labels=["request", "prompt", "state"])
    # This lower strip compares elapsed intervals; it is not another data connector.
    ax.plot([0.15, 9.45], [1.08, 1.08], color=PRIMARY, lw=2)
    ax.plot([0.15, 0.15], [0.99, 1.17], color=PRIMARY, lw=2)
    ax.plot([9.45, 9.45], [0.99, 1.17], color=PRIMARY, lw=2)
    ax.text(4.8, 0.68, "TTFT: wait to first token", ha="center", fontsize=18, color=PRIMARY)
    ax.text(10.85, 0.68, "Remaining tokens", ha="center", fontsize=18, color=TEAL)
    role_legend(ax, 0.10, size=16)
    _diagram_finish(fig, path)


def app_architecture(path):
    fig, ax = _diagram_canvas(4.1)
    ax.text(0.15, 3.70, "Proposed integration; lab routes are separate.", fontsize=18, color=INK)
    _sequence(ax, [("Request", "data"), ("Entry", "tool"), ("Input\nchecks", "tool"),
                   ("Generate", "model"), ("Output\ncheck", "tool"), ("Reply", "output")],
              2.20, height=0.9, size=18, labels=["JSON", "fields", "clean input", "answer", "checked text"])
    role_box(ax, 6.90, 0.83, 4.70, 0.70, "Log / inspect metadata", "tool", size=18)
    # Solid branches carry actual records; neither is a training-gradient arrow.
    routed_arrow(ax, [(7.52, 2.20), (7.52, 1.53)], label="tokens", label_xy=(6.85, 1.83), size=16)
    routed_arrow(ax, [(9.65, 2.20), (9.65, 1.53)], label="outcome", label_xy=(10.35, 1.83), size=16)
    role_legend(ax, 0.10, size=16)
    _diagram_finish(fig, path)


def _diagram_canvas(height):
    fig, ax = plt.subplots(figsize=(13, height))
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.04, top=0.97)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, height)
    return fig, ax


def _sequence(ax, nodes, y, width=12.6, height=0.95, size=18, labels=None):
    assert labels is not None and len(labels) == len(nodes) - 1
    gap = 0.48
    block = (width - gap * (len(nodes) - 1)) / len(nodes)
    for i, (label, role) in enumerate(nodes):
        x = 0.15 + i * (block + gap)
        role_box(ax, x, y, block, height, label, role, size=size)
        if i < len(nodes) - 1:
            arrow(ax, x + block, y + height / 2, x + block + gap, y + height / 2, color=PRIMARY)
            ax.text(x + block + gap / 2, y + height + 0.13, labels[i],
                    ha="center", va="bottom", fontsize=16, color=TEXT)


def _diagram_finish(fig, path):
    save_editable_scene(fig, path)


def _measured_results():
    """Retain archived non-timing measurements for the quantisation probe."""
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def _core_measured_results():
    """Timing plots must use the CPU core export and its genuine run record."""
    result = json.loads((ASSETS / "results_cpu_core.json").read_text(encoding="utf-8"))
    if result["device"] != "cpu":
        raise ValueError("Week 11 core timing figures require a measured CPU export")
    recorded = result.get("evidence", {}).get("started_at_utc")
    if not recorded:
        raise ValueError("Week 11 core timing figures require dated run evidence")
    when = datetime.fromisoformat(recorded.replace("Z", "+00:00"))
    model = result["model"].split("/")[-1]
    context = f"{model} · CPU core · {when.day} {when:%b %Y}"
    return result, context


def latency_results(path):
    r, context = _core_measured_results()
    rows = r["latency"]
    fig, ax = plt.subplots(figsize=(13, 4.0))
    y = list(range(len(rows)))
    total = [row["total (s)"] for row in rows]
    first = [row["TTFT (s)"] if row["TTFT (s)"] is not None else float("nan") for row in rows]
    ax.barh(y, total, color="#C7D2FE", label="Full answer")
    ax.barh(y, first, color=PRIMARY, label="First text (approx. TTFT)")
    ax.set_yticks(y, ["What is an LLM?", "Three prompt tips", "What is a rate limit?"], fontsize=20)
    ax.invert_yaxis()
    for index, row in enumerate(rows):
        first_text = f"{row['TTFT (s)']:.3f} s first" if row["TTFT (s)"] is not None else "first text unavailable"
        ax.text(row["total (s)"] + 0.1, index, f"{row['total (s)']:.2f} s total; {first_text}",
                fontsize=20, va="center", color=TEXT)
    ax.set_xlim(0, max(total) * 1.9)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("Seconds", fontsize=20)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=20, ncol=2)
    fig.suptitle(context, fontsize=20, y=1.08)
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
    r, context = _core_measured_results()
    rows = r["batch"]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    vals = [row["tokens/s"] for row in rows]
    bars = ax.bar(range(len(rows)), vals, color=TEAL)
    ax.bar_label(bars, labels=[f"{v:g}" for v in vals], fontsize=20, padding=4)
    ax.set_xticks(range(len(rows)), [row["batch size"] for row in rows], fontsize=20)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlabel("Batch size", fontsize=20)
    ax.set_ylabel("Requested tokens / second", fontsize=20)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_title(context, fontsize=18, pad=14)
    fig.tight_layout()
    save(fig, path)


def monitor_results(path):
    """Plot stored aggregate evidence; do not invent missing per-request rows."""
    r, context = _core_measured_results()
    monitor = r["monitor"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    counts = [monitor["requests"], monitor["blocked"], monitor["redacted"]]
    bars = axes[0].bar(range(3), counts, color=[PRIMARY, "#D55E00", ACCENT])
    axes[0].bar_label(bars, labels=[str(x) for x in counts], fontsize=19, padding=4)
    axes[0].set_xticks(range(3), ["All requests", "Blocked", "Redacted"], fontsize=18)
    axes[0].set_ylabel("Requests", fontsize=19)
    axes[0].set_ylim(0, max(counts) * 1.3)
    axes[0].set_title("Counts include all requests; categories overlap", fontsize=17, pad=12)
    latency = [monitor["p50 latency (s)"], monitor["p95 latency (s)"]]
    bars = axes[1].bar(range(2), latency, color=[TEAL, ACCENT])
    axes[1].bar_label(bars, labels=[f"{x:.2f} s" for x in latency], fontsize=19, padding=4)
    axes[1].set_xticks(range(2), ["p50", "p95"], fontsize=18)
    axes[1].set_ylabel("Seconds", fontsize=19)
    axes[1].set_ylim(0, max(latency) * 1.3)
    axes[1].set_title("Allowed requests only; small sample", fontsize=18, pad=12)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(axis="x", visible=False)
    fig.suptitle(context, fontsize=18, y=1.05)
    fig.tight_layout(w_pad=2)
    save(fig, path)


FIGURES = {"cover": cover, "stack": stack, "openai_compat": openai_compat, "request_anatomy": request_anatomy,
           "app_architecture": app_architecture, "lab_latency": latency_results, "lab_quant": quant_results,
           "lab_batch": batch_results, "lab_monitor": monitor_results}
