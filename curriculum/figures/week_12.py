"""Week 12 figures: RAG and agent diagrams plus REAL results harvested from the executed lab notebook."""
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

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
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.text(0.1, 4.85, "Indexing (offline)", fontsize=14, color=TEAL, fontweight="bold")
    idx = [("documents", 0.1), ("chunk", 2.3), ("embed", 4.5), ("vector store\n+ keyword index", 6.7)]
    for i, (t, x) in enumerate(idx):
        box(ax, x, 3.7, 1.9, 0.9, t, fc=GREEN, ec=TEAL, size=12)
        if i < len(idx) - 1:
            arrow(ax, x + 1.9, 4.15, x + 2.3, 4.15)
    ax.text(0.1, 2.75, "Answering (online)", fontsize=14, color=PRIMARY, fontweight="bold")
    q = [("question", 0.1, 1.5), ("retrieve\n(dense + BM25)", 1.95, 1.75), ("re-rank\ntop k", 4.05, 1.5), ("prompt with\nnumbered sources", 5.9, 1.9),
         ("LLM", 8.15, 1.1), ("answer with\ncitations", 9.6, 1.7)]
    for i, (t, x, w) in enumerate(q):
        box(ax, x, 1.2, w, 1.0, t, fc=TINT if t != "LLM" else ORANGE, ec=PRIMARY if t != "LLM" else ACCENT, size=12)
        if i < len(q) - 1:
            arrow(ax, x + w, 1.7, q[i + 1][1], 1.7)
    arrow(ax, 7.65, 3.7, 2.9, 2.2)
    ax.text(11.5, 1.7, "or 'I don't know'", fontsize=11.5, color=MUTED, va="center")
    ax.text(0.1, 0.35, "The model answers from retrieved evidence instead of memory: current, private and checkable.", fontsize=12.5, color=TEXT)
    save(fig, path)


def chunking(path):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.add_patch(plt.Rectangle((0.2, 2.5), 12.6, 0.6, color="#E5E7EB"))
    ax.text(6.5, 2.8, "document text …………………………………………………………………………………………", ha="center", va="center", fontsize=12, color=MUTED)
    for i in range(4):
        x = 0.2 + i * 3.0
        c = SERIES[i % len(SERIES)]
        ax.add_patch(plt.Rectangle((x, 1.35 - (i % 2) * 0.55), 3.6, 0.45, facecolor="white", edgecolor=c, lw=2.5))
        ax.text(x + 1.8, 1.575 - (i % 2) * 0.55, f"chunk {i + 1}", ha="center", va="center", fontsize=11.5, color=c, fontweight="bold")
    ax.annotate("", xy=(3.8, 0.55), xytext=(3.2, 0.55), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.5))
    ax.text(3.5, 0.2, "overlap", ha="center", fontsize=11.5, color=ACCENT)
    ax.text(12.8, 0.35, "size: 200–1,000 tokens · overlap 10–20% · respect headings and paragraphs", ha="right", fontsize=12, color=TEXT)
    save(fig, path)


def agent_loop(path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    box(ax, 0.2, 2.0, 1.9, 1.0, "user task", fc=TINT, size=12.5)
    arrow(ax, 2.1, 2.5, 3.2, 2.5)
    box(ax, 3.2, 1.7, 2.6, 1.6, "LLM\nreason: what next?", fc=ORANGE, ec=ACCENT, size=12.5)
    box(ax, 7.4, 3.3, 2.5, 1.0, "act: call a tool\n(JSON arguments)", fc=TINT, size=11.5)
    box(ax, 7.4, 0.7, 2.5, 1.0, "observe: tool\nresult", fc=TINT, size=11.5)
    arrow(ax, 5.8, 2.9, 7.4, 3.8)
    arrow(ax, 8.65, 3.3, 8.65, 1.7)
    arrow(ax, 7.4, 1.2, 5.8, 2.1)
    tools = ["search (RAG)", "calculator", "code / APIs", "e-mail (needs approval)"]
    for i, t in enumerate(tools):
        box(ax, 10.6, 3.9 - i * 0.95, 2.2, 0.7, t, fc="white", ec=SERIES[i % len(SERIES)], color=SERIES[i % len(SERIES)], size=10.5)
    arrow(ax, 9.9, 3.8, 10.6, 3.6)
    box(ax, 3.9, 0.2, 2.6, 0.8, "memory: messages,\nnotes, state", fc="white", ec=MUTED, size=10.5)
    ax.text(3.2, 4.4, "repeat until done or a step limit is reached", fontsize=12, color=ACCENT, fontweight="bold")
    box(ax, 0.2, 0.2, 1.9, 0.9, "final answer", fc="white", ec=ACCENT, color=ACCENT, size=11.5)
    arrow(ax, 3.2, 1.9, 2.1, 0.8)
    ax.text(1.0, 1.35, "no tool needed", fontsize=10.5, color=MUTED)
    save(fig, path)


def workflows(path):
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.7))
    names = [("Prompt chaining", "step → step → step"), ("Routing", "classify, then send to\nthe right handler"),
             ("Parallelisation", "split or vote,\nthen aggregate"), ("Orchestrator–workers", "LLM plans sub-tasks\nfor worker LLMs"),
             ("Evaluator–optimiser", "generate, critique,\nrevise")]
    for a, (t, d), c in zip(axes, names, SERIES):
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.add_patch(plt.Rectangle((0.03, 0.08), 0.94, 0.84, facecolor="white", edgecolor=c, lw=2.5))
        a.text(0.5, 0.68, t, ha="center", va="center", fontsize=12, color=c, fontweight="bold")
        a.text(0.5, 0.35, d, ha="center", va="center", fontsize=10.5, color=TEXT)
    fig.suptitle("Workflows: fixed code paths (predictable, cheaper)   vs   agents: the LLM decides the steps (flexible, riskier)",
                 fontsize=12.5, color=INK, y=0.02)
    save(fig, path)


def mcp(path):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    box(ax, 0.2, 1.4, 3.2, 1.8, "Host application\n(assistant, IDE, agent)\nwith MCP client", fc=TINT, size=12)
    servers = [("Files server", "read project files"), ("Database server", "query tables"), ("Search server", "our handbook RAG"),
               ("Calendar / e-mail server", "actions: need approval")]
    for i, (s, d) in enumerate(servers):
        y = 3.7 - i * 0.95
        c = SERIES[i % len(SERIES)]
        box(ax, 6.2, y, 3.0, 0.75, s, fc="white", ec=c, color=c, size=11.5)
        ax.text(9.45, y + 0.37, d, fontsize=11, color=MUTED, va="center")
        arrow(ax, 3.4, 2.3, 6.2, y + 0.37)
    ax.text(4.1, 4.15, "MCP (JSON-RPC)", fontsize=12, color=ACCENT, fontweight="bold")
    ax.text(0.2, 0.2, "Servers expose tools, resources and prompts; any MCP client can discover and use them.", fontsize=12, color=TEXT)
    save(fig, path)


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


FIGURES = {"cover": cover, "rag_pipeline": rag_pipeline, "chunking": chunking, "agent_loop": agent_loop, "workflows": workflows, "mcp": mcp,
           "agent_trace": agent_trace, **{n: asset(n) for n in ["lab_retrieval", "lab_rag"]}}
