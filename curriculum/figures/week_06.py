"""Week 6 figures: NLP evolution, scaling laws, LLM family tree, reasoning models, plus REAL results."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_06"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(9):
        y = np.cumsum(rng.normal(0, 1, 60)) + k * 6
        ax.plot(y, color=SERIES[k % len(SERIES)], lw=2, alpha=0.8)
    ax.axis("off")
    save(fig, path, transparent=True)


def nlp_timeline(path):
    ev = [(1990, "Rules &\nstatistics", "grammars, n-gram\nlanguage models"),
          (2013, "word2vec", "words as\nvectors"),
          (2014, "Seq2seq\nRNN/LSTM", "+ attention (2015)\nfor translation"),
          (2017, "Transformer", "attention only;\nparallel training"),
          (2018, "BERT / GPT", "pre-train, then\nfine-tune"),
          (2020, "GPT-3", "scale → few-shot\nprompting"),
          (2022, "ChatGPT", "instruction +\npreference tuning"),
          (2024, "Open, multimodal,\nreasoning", "open weights; test-\ntime compute; agents")]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    xs = np.arange(len(ev))
    ax.set_xlim(-0.6, len(ev) - 0.4)
    ax.set_ylim(-2.6, 2.6)
    ax.plot([-0.4, len(ev) - 0.6], [0, 0], color=MUTED, lw=2.5)
    for i, (yr, h, sub) in enumerate(ev):
        up = i % 2 == 0
        s = 1 if up else -1
        c = SERIES[i % len(SERIES)]
        ax.scatter([i], [0], s=150, color=c, zorder=3)
        ax.plot([i, i], [0, s * 0.5], color=c, lw=2)
        ax.text(i, s * 0.58, h, ha="center", va="bottom" if up else "top", fontsize=13.5, fontweight="bold", color=INK)
        ax.text(i, s * 1.55, sub, ha="center", va="bottom" if up else "top", fontsize=11.5, color=MUTED)
        ax.text(i, -s * 0.15, "1990s–2000s" if yr == 1990 else str(yr), ha="center", va="top" if up else "bottom", fontsize=11, color=TEXT)
    save(fig, path)


def scaling(path):
    # Chinchilla parametric fit (Hoffmann et al., 2022, approach 3)
    E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28
    C = np.logspace(19, 25, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for N, c in zip([1e8, 1e9, 1e10, 7e10], SERIES):
        D = C / (6 * N)
        L = E + A / N ** a + B / D ** b
        axes[0].semilogx(C, L, color=c, lw=2.2, label=f"{N / 1e9:g} B parameters" if N >= 1e9 else f"{N / 1e6:g} M parameters")
    Nopt = np.logspace(8, 12, 200)
    Lopt = [min(E + A / n ** a + B / (c_ / (6 * n)) ** b for n in Nopt) for c_ in C]
    axes[0].semilogx(C, Lopt, color=INK, lw=3, ls="--", label="best size for each budget")
    axes[0].set_ylim(1.8, 4.2)
    axes[0].set_xlabel("training compute C (FLOPs) ≈ 6 N D")
    axes[0].set_ylabel("predicted loss")
    axes[0].set_title("Loss falls smoothly with compute", loc="left")
    axes[0].legend(fontsize=11.5)
    b_ = axes[1]
    Ns = np.logspace(8, 12, 50)
    b_.loglog(Ns, 20 * Ns, color=TEAL, lw=2.5, label="≈ 20 tokens per parameter (Chinchilla)")
    pts = [("GPT-3 (2020)", 175e9, 300e9), ("Chinchilla (2022)", 70e9, 1.4e12), ("Llama 3 8B (2024)", 8e9, 15e12)]
    for (n_, x, y), c in zip(pts, (ACCENT, PRIMARY, "#A21CAF")):
        b_.scatter([x], [y], s=90, color=c, zorder=3)
        b_.text(x * 1.3, y * 0.75, n_, fontsize=12.5, color=c)
    b_.set_xlabel("parameters N")
    b_.set_ylabel("training tokens D")
    b_.set_title("How much data per parameter?", loc="left")
    b_.legend(fontsize=11.5, loc="lower right")
    for ax_ in axes:
        ax_.tick_params(labelsize=11.5)
    save(fig, path)


def family_tree(path):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    box(ax, 5.5, 4.8, 2.0, 0.6, "Transformer (2017)", fc=TINT, size=12)
    cols = [(0.3, "Encoder-only", PRIMARY, ["BERT (2018)", "RoBERTa, DeBERTa", "ModernBERT (2024)", "embedding models"]),
            (4.6, "Encoder–decoder", TEAL, ["T5 / Flan-T5", "BART, mT5", "Whisper (speech)"]),
            (8.6, "Decoder-only", ACCENT, ["GPT family (OpenAI)", "Claude (Anthropic), Gemini (Google)", "Llama (Meta), Gemma (Google)",
                                           "Qwen (Alibaba), DeepSeek, Mistral", "Phi (Microsoft), OLMo (AI2, fully open)", "gpt-oss (OpenAI, open-weight)"])]
    for x, name, c, items in cols:
        box(ax, x, 3.7, 3.9 if x > 8 else 3.6, 0.6, name, fc="white", ec=c, color=c, size=13)
        ax.annotate("", xy=(x + 1.8, 4.3), xytext=(6.5, 4.8), arrowprops=dict(arrowstyle="-|>", color=MUTED))
        for k, it in enumerate(items):
            ax.text(x + 0.1, 3.35 - k * 0.5, "• " + it, fontsize=11.5, color=TEXT, va="top")
    ax.text(0.3, 0.2, "Families shown with typical examples; versions change monthly. Proprietary and open-weight members exist in several families.", fontsize=11, color=MUTED)
    save(fig, path)


def reasoning(path):
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.6)
    ax.text(0.1, 3.35, "Standard chat model", fontsize=14, fontweight="bold", color=INK, va="center")
    box(ax, 0.1, 2.2, 1.8, 0.75, "prompt", fc=TINT, size=13)
    box(ax, 2.5, 2.2, 1.8, 0.75, "answer", fc="white", size=13)
    arrow(ax, 1.9, 2.575, 2.5, 2.575)
    ax.text(5.0, 2.6, "Trained with reinforcement learning on problems\nwith checkable answers (week 8). More thinking\n"
            "tokens → often better on maths and code,\nbut slower and costlier.", fontsize=12.5, color=TEXT, va="center")
    ax.text(0.1, 1.45, "Reasoning model (test-time compute)", fontsize=14, fontweight="bold", color=INK, va="center")
    box(ax, 0.1, 0.35, 1.8, 0.75, "prompt", fc=TINT, size=13)
    box(ax, 2.5, 0.35, 5.0, 0.75, "long hidden reasoning: plan, try, check, revise …", fc=ORANGE, ec=ACCENT, size=13)
    box(ax, 8.1, 0.35, 1.8, 0.75, "answer", fc="white", size=13)
    arrow(ax, 1.9, 0.725, 2.5, 0.725)
    arrow(ax, 7.5, 0.725, 8.1, 0.725)
    save(fig, path)


FIGURES = {
    "cover": cover, "nlp_timeline": nlp_timeline, "scaling": scaling, "family_tree": family_tree, "reasoning": reasoning,
    **{n: asset(n) for n in ["tokenizers", "next_token"]},
}
