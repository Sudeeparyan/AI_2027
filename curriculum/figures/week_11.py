"""Week 11 figures: deployment diagrams plus REAL measurements harvested from the executed lab notebook."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

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
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    layers = [("Applications & UI", "Gradio, Streamlit, web and mobile front ends, FastAPI back ends", PRIMARY),
              ("Orchestration", "LangChain / LangGraph, LlamaIndex, agent SDKs: prompts, tools, RAG", "#A21CAF"),
              ("Access & serving", "cloud APIs (OpenAI, Azure, Gemini, Anthropic) · vLLM, SGLang · Ollama, llama.cpp", TEAL),
              ("Models", "proprietary (GPT, Gemini, Claude) · open-weight (Llama, Qwen, Mistral, Gemma)", ACCENT),
              ("Hardware", "GPUs / TPUs in the cloud, on-premises servers, laptops and phones (NPUs)", MUTED)]
    for i, (name, desc, c) in enumerate(layers):
        y = 4.1 - i * 0.92
        box(ax, 0.1, y, 2.9, 0.72, name, fc="white", ec=c, color=c, size=13)
        ax.text(3.25, y + 0.36, desc, fontsize=12, color=TEXT, va="center")
    ax.add_patch(plt.Rectangle((12.35, 0.42), 0.55, 4.4, color=TINT))
    ax.text(12.62, 2.62, "LLMOps: logs, traces, evals, guardrails", rotation=90, ha="center", va="center", fontsize=11.5, color=INK)
    save(fig, path)


def openai_compat(path):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    box(ax, 0.2, 1.55, 3.3, 1.5, "your code\nclient.chat.completions.create(\n  model=..., messages=...)", fc=TINT, size=11.5)
    backends = [("OpenAI / Azure OpenAI", "api.openai.com · <resource>.openai.azure.com"),
                ("Gemini API", "generativelanguage.googleapis.com/v1beta/openai/"),
                ("Hugging Face Inference Providers", "router.huggingface.co/v1"),
                ("vLLM server (self-hosted)", "http://<server>:8000/v1"),
                ("Ollama (local)", "http://localhost:11434/v1"),
                ("Your FastAPI app (lab)", "http://127.0.0.1:8011/v1")]
    for i, (name, url) in enumerate(backends):
        y = 4.0 - i * 0.72
        c = SERIES[i % len(SERIES)]
        box(ax, 5.2, y - 0.25, 3.4, 0.52, name, fc="white", ec=c, color=c, size=11)
        ax.text(8.8, y, url, fontsize=10.5, color=MUTED, va="center", family="monospace")
        arrow(ax, 3.5, 2.3, 5.2, y)
    ax.text(0.2, 0.5, "change only base_url, api_key and model", fontsize=12.5, color=ACCENT, fontweight="bold")
    save(fig, path)


def request_anatomy(path):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    parts = [("network +\nauth", 0.9, "#CBD5E1"), ("queue", 0.9, "#E2E8F0"), ("prefill\n(read prompt)", 1.9, "#C7D2FE"),
             ("decode: one token at a time (streamed)", 7.6, "#99F6E4")]
    x = 0.2
    for label, w, c in parts:
        ax.add_patch(plt.Rectangle((x, 1.6), w, 0.9, color=c))
        ax.text(x + w / 2, 2.05, label, ha="center", va="center", fontsize=11.5, color=INK)
        x += w
    ax.annotate("", xy=(3.9, 1.35), xytext=(0.2, 1.35), arrowprops=dict(arrowstyle="<->", color=PRIMARY, lw=1.6))
    ax.text(2.05, 0.95, "time to first token (TTFT)", ha="center", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.annotate("", xy=(11.5, 0.55), xytext=(0.2, 0.55), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6))
    ax.text(7.7, 0.15, "total latency ≈ TTFT + output tokens ÷ tokens per second", ha="center", fontsize=12, color=ACCENT, fontweight="bold")
    ax.text(0.2, 3.1, "Long prompts raise prefill time; long answers raise decode time; queues grow under load.", fontsize=12, color=TEXT)
    save(fig, path)


def app_architecture(path):
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0.2, 4.0)
    ax.text(0.1, 3.85, "Proposed integrated design; the lab's Gradio route calls generate directly.", fontsize=11, color=MUTED)
    steps = [("User", 0.1, TINT, PRIMARY), ("UI\n(Gradio)", 1.75, TINT, PRIMARY), ("API\n(FastAPI)", 3.4, TINT, PRIMARY),
             ("input\nguardrails", 5.05, ORANGE, ACCENT), ("model\n(local or cloud)", 6.7, GREEN, TEAL),
             ("output\nguardrails", 8.35, ORANGE, ACCENT), ("answer", 10.0, TINT, PRIMARY)]
    for i, (label, x, fc, ec) in enumerate(steps):
        box(ax, x, 2.7, 1.45, 1.0, label, fc=fc, ec=ec, size=11.5)
        if i < len(steps) - 1:
            arrow(ax, x + 1.45, 3.2, x + 1.65, 3.2)
    box(ax, 3.4, 0.5, 6.4, 0.95, "logs · traces · metrics (latency, tokens,\ncost, blocks) · model & prompt versions", fc="white", ec=MUTED, size=11.5)
    for x in (4.12, 5.77, 7.42, 9.07):
        ax.plot([x, x], [2.7, 1.45], color=MUTED, lw=1.2, ls="--")
    box(ax, 10.3, 0.5, 2.5, 0.95, "dashboard\n& alerts", fc="white", ec=INK, size=11.5)
    arrow(ax, 9.8, 0.97, 10.3, 0.97)
    ax.text(11.6, 3.2, "packaged in\na container\n(Docker)", fontsize=11, color=MUTED, va="center")
    save(fig, path)


FIGURES = {"cover": cover, "stack": stack, "openai_compat": openai_compat, "request_anatomy": request_anatomy,
           "app_architecture": app_architecture, **{n: asset(n) for n in ["lab_latency", "lab_quant", "lab_batch", "lab_monitor"]}}
