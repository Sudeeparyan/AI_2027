"""Week 12 figures: RAG and agent diagrams plus REAL results harvested from the executed lab notebook."""
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

from _style import role_box, role_legend, routed_arrow, save_editable_scene, ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_12"
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
    for i in range(3):
        ax.add_patch(plt.Rectangle((0.1 + i * 0.06, 0.52 - i * 0.06), 0.38, 0.4, facecolor="white", edgecolor=PRIMARY, lw=3))
    for j in range(4):
        ax.plot([0.26, 0.56], [0.72 - j * 0.07, 0.72 - j * 0.07], color="#C7D2FE", lw=4)
    ax.add_patch(plt.Circle((0.72, 0.3), 0.2, color=ACCENT))
    ax.text(0.72, 0.3, "⚙", ha="center", va="center", fontsize=60, color="white")
    ax.annotate("", xy=(0.6, 0.45), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=3))
    save(fig, path, transparent=True)


def rag_pipeline(path):
    fig, ax = _diagram_canvas(4.7)
    ax.text(0.15, 4.28, "Offline: build the searchable index", fontsize=20, color=TEAL, fontweight="bold")
    _sequence(ax, [("Documents", "data"), ("Chunks + IDs", "data"), ("Embeddings", "model"),
                   ("Search index", "output")], 3.05, width=9.9, height=0.85, size=18)
    ax.text(0.15, 2.5, "Online answer", fontsize=20, color=PRIMARY, fontweight="bold")
    _sequence(ax, [("Question", "data"), ("Retrieve", "tool"), ("Re-rank", "model"),
                   ("Sources\nin prompt", "data"), ("Generate", "model"),
                   ("Cited answer\nor abstain", "output")], 1.25, height=0.9, size=18)
    routed_arrow(ax, [(9.02, 3.05), (9.02, 2.86), (3.37, 2.86), (3.37, 2.15)], color=TEAL)
    ax.text(0.15, 0.75, "Retrieved evidence can help; verify support and source access.", fontsize=18, color=TEXT)
    role_legend(ax, 0.08, size=16)
    _diagram_finish(fig, path)



def chunking(path):
    fig, ax = _diagram_canvas(2.9)
    role_box(ax, 0.15, 2.12, 12.55, 0.52, "Document with headings, paragraphs and source metadata", "data", size=20)
    for i in range(4):
        x = 0.15 + i * 3.0
        y = 1.2 - (i % 2) * 0.4
        role_box(ax, x, y, 3.5, 0.45, f"Chunk {i + 1}", "data", size=18)
    arrow(ax, 3.4, 0.57, 3.15, 0.57, color=ACCENT)
    arrow(ax, 3.4, 0.57, 3.65, 0.57, color=ACCENT)
    ax.text(3.4, 0.29, "Overlap", fontsize=17, color=ACCENT, ha="center")
    ax.text(5.5, 0.47, "Test size, boundaries and overlap on real questions.", fontsize=17, color=TEXT)
    _diagram_finish(fig, path)



def agent_loop(path):
    fig, ax = _diagram_canvas(4.9)
    ax.text(0.15, 4.48, "Propose, check, act, observe; stop at the limit", fontsize=20, color=INK, fontweight="bold")
    entries = [("Task + history", 0.15, "data"), ("Model proposes\none tool call", 3.4, "model"),
               ("Code checks\nname + arguments", 6.65, "tool"), ("Allowed tool\nsearch / calculate", 9.9, "tool")]
    for i, (label, x, role) in enumerate(entries):
        role_box(ax, x, 3.0, 2.8, 1.0, label, role, size=18)
        if i < 3:
            arrow(ax, x + 2.8, 3.5, x + 3.25, 3.5)
    role_box(ax, 9.9, 1.2, 2.8, 1.0, "Observation\nresult or error", "output", size=18)
    arrow(ax, 11.3, 3.0, 11.3, 2.2, color=TEAL)
    routed_arrow(ax, [(8.05, 3.0), (8.05, 2.45), (10.25, 2.45), (10.25, 2.2)], color=ACCENT)
    ax.text(8.2, 2.65, "Blocked", fontsize=17, color=ACCENT)
    routed_arrow(ax, [(9.9, 1.7), (1.55, 1.7), (1.55, 3.0)], color=TEAL, dashed=True)
    ax.text(0.15, 0.9, "Repeat: append observation; next model call reads it", fontsize=17, color=TEAL)
    role_box(ax, 3.4, 2.05, 2.8, 0.75, "Final answer\n(no tool call)", "output", size=16)
    arrow(ax, 4.8, 3.0, 4.8, 2.8, color=ACCENT)
    ax.text(0.15, 0.55, "Email and approval are simulated in the lab; no real message is sent.", fontsize=17, color=MUTED)
    role_legend(ax, 0.08, size=16)
    _diagram_finish(fig, path)



def workflows(path):
    fig, ax = _diagram_canvas(2.55)
    cards = [("Chaining", "Fixed sequence"), ("Routing", "Choose handler"), ("Parallel", "Split then merge"),
             ("Workers", "Delegate tasks"), ("Evaluate", "Critique, revise")]
    width = 2.27
    for i, (title, detail) in enumerate(cards):
        x = 0.15 + i * 2.6
        role_box(ax, x, 1.16, width, 0.82, title, "tool", size=20)
        ax.text(x + width / 2, 0.88, detail, fontsize=17, color=TEXT, ha="center")
    ax.text(0.15, 0.25, "Workflow: code defines paths. Agent: model chooses the next step.", fontsize=19, color=INK)
    _diagram_finish(fig, path)



def mcp(path):
    fig, ax = _diagram_canvas(3.0)
    role_box(ax, 0.15, 1.12, 3.5, 1.05, "Host application\nmanages clients", "tool", size=19)
    for i, (label, detail) in enumerate([("Files server", "Read files"), ("Database server", "Query tables"),
                                       ("Search server", "Retrieve evidence"), ("Action server", "Check permissions")]):
        y = 2.44 - i * 0.57
        role_box(ax, 6.2, y, 3.1, 0.44, label, "tool", size=18)
        ax.text(9.7, y + 0.22, detail, fontsize=18, color=TEXT, va="center")
        arrow(ax, 3.65, 1.645, 6.2, y + 0.22)
    ax.text(0.15, 0.25, "One client per server; expose tools, resources and prompts with access controls.", fontsize=18, color=INK)
    _diagram_finish(fig, path)



def agent_trace(path):
    """Render the real agent trajectory saved by the lab run (curriculum/assets/week_12/results.json)."""
    res = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    lines = [("Task: I scored 72 in the group project and 64 in the exam. Using the weights in the handbook, what is my final mark?", INK)]
    for t in res["agent_trace"]:
        colour = TEAL if t["type"] == "tool" else (ACCENT if t["type"] == "answer" else MUTED)
        label = "tool call" if t["type"] == "tool" else t["type"]
        content = " ".join(t["content"].split())
        if t["type"] == "tool" and " -> " in content:
            call, obs = content.split(" -> ", 1)
            obs = obs.lstrip("- ")
            if len(obs) >= 85:  # the lab stores the first 90 characters of each observation: end on a whole content word
                words = obs[:obs.rstrip().rfind(" ")].rstrip(" ,;:").split()
                while words and words[-1].lower() in {"a", "an", "and", "the", "of", "or", "to", "in", "is"}:
                    words.pop()
                obs = " ".join(words) + " …"
            content = f"{call} -> {obs}"
        lines.append((f"step {t['step']} · {label}: {content}", colour))
    width = 78
    n_lines = sum(len(textwrap.wrap(s.replace("\n", " "), width)) for s, _ in lines)
    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.42 * n_lines))
    ax.axis("off")
    y = 0.97
    step = 0.94 / max(n_lines, 1)
    for s, c in lines:
        for w in textwrap.wrap(s.replace("\n", " "), width):
            ax.text(0.0, y, w, fontsize=15, color=c, family="monospace" if s.startswith("step") else None, va="top")
            y -= step
    save(fig, path)



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


def retrieval_results(path):
    r = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    methods = list(r["retrieval"])
    import numpy as np
    fig, ax = plt.subplots(figsize=(13, 3.9))
    x = np.arange(len(methods))
    for index, (field, label, colour) in enumerate([
        ("recall@1", "Gold-document hit@1", PRIMARY),
        ("recall@3", "Gold-document hit@3", TEAL),
        ("MRR", "Mean reciprocal rank", ACCENT),
    ]):
        values = [r["retrieval"][name][field] for name in methods]
        bars = ax.bar(x + (index - 1) * 0.26, values, 0.26, label=label, color=colour)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], fontsize=18, padding=3)  # as in the slide text
    ax.set_xticks(x, ["BM25\nkeywords", "Dense\nembeddings", "Hybrid\nRRF", "Hybrid +\nre-rank"], fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 1.22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), fontsize=18)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, path)


def rag_results(path):
    r = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    rows = r["summary"]
    names = list(rows)
    values = [rows[name]["accuracy (answerable)"] for name in names]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    bars = ax.barh(range(len(values)), values, color=[MUTED, PRIMARY])
    ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], fontsize=22, padding=7)
    ax.set_yticks(range(len(values)), ["Without retrieval", "RAG"], fontsize=22)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Accuracy on answerable handbook questions", fontsize=20)
    where = r.get("gpu") or "GPU" if str(r["device"]).startswith("cuda") else "CPU"
    ax.set_title(f"Stored run: {r['llm'].split('/')[-1]} on {where.replace('NVIDIA GeForce ', '')}", fontsize=18, pad=14)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)


FIGURES = {"cover": cover, "rag_pipeline": rag_pipeline, "chunking": chunking, "agent_loop": agent_loop, "workflows": workflows, "mcp": mcp,
           "agent_trace": agent_trace, "lab_retrieval": retrieval_results, "lab_rag": rag_results}
