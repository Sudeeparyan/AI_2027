"""Week 5 figures: transformer and attention diagrams, plus REAL results from curriculum/assets/week_05."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_05"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    rng = np.random.default_rng(5)
    n = 14
    a = rng.random((n, n)) ** 3
    a = np.tril(a) + np.eye(n) * 0.6
    a = a / a.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(a, cmap="magma")
    ax.axis("off")
    save(fig, path, transparent=True)


def rnn_vs_attention(path):
    words = ["The", "animal", "didn't", "cross", "…", "it", "was", "tired"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.4))
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(-0.5, len(words) - 0.3)
        ax.set_ylim(-0.4, 1.5)
    a = axes[0]
    a.set_title("RNN: information travels step by step (slow, and early words fade)", loc="left", fontsize=14)
    for i, w in enumerate(words):
        box(a, i - 0.4, 0.0, 0.8, 0.5, w, size=12)
        box(a, i - 0.3, 0.85, 0.6, 0.45, "h", fc=ORANGE, ec=ACCENT, size=12)
        a.annotate("", xy=(i, 0.85), xytext=(i, 0.5), arrowprops=dict(arrowstyle="-|>", color=MUTED))
        if i < len(words) - 1:
            a.annotate("", xy=(i + 0.7, 1.07), xytext=(i + 0.3, 1.07), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.8))
    b = axes[1]
    b.set_title("Self-attention: every word looks at every other word directly, in parallel", loc="left", fontsize=14)
    for i, w in enumerate(words):
        box(b, i - 0.4, 0.0, 0.8, 0.5, w, size=12, fc=TINT if w != "it" else ORANGE, ec=PRIMARY if w != "it" else ACCENT)
    src = words.index("it")
    for j in range(len(words)):
        if j != src:
            wgt = 0.9 if words[j] == "animal" else 0.25
            b.annotate("", xy=(j, 0.52), xytext=(src, 0.52), arrowprops=dict(arrowstyle="-|>", color=ACCENT if words[j] == "animal" else MUTED,
                                                                              lw=1 + 3 * wgt, alpha=0.4 + 0.6 * wgt, connectionstyle="arc3,rad=-0.35"))
    save(fig, path)


def qkv(path):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    box(ax, 0.1, 1.8, 1.5, 1.2, "token\nvectors X\n(n × d)", size=12)
    for k, (name, y, c) in enumerate((("Q = X W_Q", 3.4, PRIMARY), ("K = X W_K", 2.1, TEAL), ("V = X W_V", 0.8, ACCENT))):
        box(ax, 2.3, y - 0.3, 1.9, 0.8, name, fc="white", ec=c, color=c, size=13)
        arrow(ax, 1.6, 2.4, 2.3, y + 0.1)
    box(ax, 4.9, 2.4, 2.0, 1.3, "scores\nQ Kᵀ / √d_k\n(n × n)", fc=TINT, size=12)
    box(ax, 7.5, 2.4, 1.8, 1.3, "softmax\nper row\n(+ mask)", fc=TINT, size=12)
    box(ax, 9.9, 1.3, 1.6, 1.3, "weights\n× V", fc=ORANGE, ec=ACCENT, size=13)
    box(ax, 11.9, 1.3, 1.0, 1.3, "new\nvectors", size=11)
    arrow(ax, 4.2, 3.5, 4.9, 3.2)
    arrow(ax, 4.2, 2.2, 4.9, 2.8)
    arrow(ax, 6.9, 3.05, 7.5, 3.05)
    arrow(ax, 9.3, 3.05, 10.3, 2.6)
    arrow(ax, 4.2, 0.9, 9.9, 1.6)
    arrow(ax, 11.5, 1.95, 11.9, 1.95)
    ax.text(0.1, 4.45, "Query: what am I looking for?   Key: what do I contain?   Value: what do I pass on?", fontsize=13.5, color=INK, fontweight="bold")
    save(fig, path)


def attn_worked(path):
    toks = ["cat", "sat", "mat"]
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s = Q @ K.T / np.sqrt(2)
    w = np.exp(s) / np.exp(s).sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a, M, t, cmap in ((axes[0], s, "scores QKᵀ/√2", "Blues"), (axes[1], w, "weights = softmax(row)", "Purples")):
        a.imshow(M, cmap=cmap, vmin=0)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=14, color=INK)
        a.set_xticks(range(3))
        a.set_xticklabels(toks)
        a.set_yticks(range(3))
        a.set_yticklabels(toks)
        a.set_title(t, loc="left")
        a.set_xlabel("key")
        a.set_ylabel("query")
        a.grid(False)
    save(fig, path)


def causal_mask(path):
    toks = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(toks)
    m = np.tril(np.ones((n, n)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, M, t in ((axes[0], np.ones((n, n)), "Bidirectional (encoder, e.g. BERT)"), (axes[1], m, "Causal mask (decoder, e.g. GPT)")):
        a.imshow(M, cmap="Purples", vmin=0, vmax=1.4)
        for i in range(n):
            for j in range(n):
                a.text(j, i, "✓" if M[i, j] else "−∞", ha="center", va="center", fontsize=12, color=INK if M[i, j] else ACCENT)
        a.set_xticks(range(n))
        a.set_xticklabels(toks)
        a.set_yticks(range(n))
        a.set_yticklabels(toks)
        a.set_title(t, loc="left")
        a.set_xlabel("can attend to (key)")
        a.set_ylabel("token (query)")
        a.grid(False)
    save(fig, path)


def multihead(path):
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    box(ax, 0.1, 1.6, 1.4, 1.2, "X", size=16)
    heads = ["head 1:\nsyntax", "head 2:\ncoreference", "head 3:\nnext word", "head h: …"]
    for i, h in enumerate(heads):
        y = 3.5 - i * 0.92
        box(ax, 2.4, y, 2.6, 0.78, h, fc=TINT, size=12.5)
        arrow(ax, 1.5, 2.2, 2.4, y + 0.39)
        arrow(ax, 5.0, y + 0.39, 6.0, 2.2)
    box(ax, 6.0, 1.6, 2.0, 1.2, "concatenate", fc="white", size=13)
    box(ax, 8.7, 1.6, 1.8, 1.2, "× W_O", fc=ORANGE, ec=ACCENT, size=14)
    box(ax, 11.1, 1.6, 1.6, 1.2, "output\n(n × d)", size=12)
    arrow(ax, 8.0, 2.2, 8.7, 2.2)
    arrow(ax, 10.5, 2.2, 11.1, 2.2)
    ax.text(0.1, 0.22, "Each head has its own W_Q, W_K, W_V of size d × (d/h): same total cost as one big head, several 'views' at once.", fontsize=13, color=TEXT)
    save(fig, path)


def positional(path):
    d, n = 64, 100
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].imshow(pe.T, cmap="RdBu", aspect="auto")
    axes[0].set_xlabel("position in the sequence")
    axes[0].set_ylabel("embedding dimension")
    axes[0].set_title("Sinusoidal positional encoding (original transformer)", loc="left")
    axes[0].grid(False)
    b = axes[1]
    b.set_xlim(-1.4, 1.4)
    b.set_ylim(-1.4, 1.4)
    b.set_aspect("equal")
    b.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    b.plot(np.cos(th), np.sin(th), color=MUTED, lw=1)
    for p, c in zip(range(0, 6), SERIES):
        a = 0.45 * p + 0.3
        b.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        b.text(1.15 * np.cos(a), 1.15 * np.sin(a), f"pos {p}", ha="center", va="center", fontsize=11, color=c)
    b.set_title("Rotary (RoPE): rotate Q and K by an\nangle ∝ position; dot products then\ndepend on relative distance", loc="left", fontsize=13)
    save(fig, path)


def block(path):
    fig, ax = plt.subplots(figsize=(6.4, 7))
    ax.axis("off")
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7)
    items = [(0.3, "token + position embeddings", "white"), (1.3, "LayerNorm", TINT), (2.1, "Multi-head self-attention", ORANGE),
             (3.3, "LayerNorm", TINT), (4.1, "Feed-forward MLP (4× wider)", ORANGE), (5.4, "→ next block (× N)", "white")]
    for y, t, fc in items:
        box(ax, 1.0, y, 4.0, 0.6, t, fc=fc, ec=ACCENT if fc == ORANGE else PRIMARY, size=12.5)
    for y1, y2 in ((0.9, 1.3), (1.9, 2.1), (2.7, 3.3), (3.9, 4.1), (4.7, 5.4)):
        arrow(ax, 3.0, y1, 3.0, y2)
    for ys, ye in ((1.1, 3.05), (3.1, 5.15)):
        ax.annotate("", xy=(5.05, ye), xytext=(5.05, ys), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2, connectionstyle="arc3,rad=-0.6"))
    ax.text(5.8, 2.1, "+ residual", color=TEAL, fontsize=12, rotation=90, va="center")
    ax.text(5.8, 4.1, "+ residual", color=TEAL, fontsize=12, rotation=90, va="center")
    ax.text(0.2, 6.5, "One (pre-norm) transformer block", fontsize=15, fontweight="bold", color=INK)
    save(fig, path)


def three_archs(path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    specs = [("Encoder-only", "BERT, RoBERTa,\nembedding models", "understand: classify,\nsearch, tag", np.ones((5, 5))),
             ("Decoder-only", "GPT, LLaMA, Claude,\nGemini, Qwen", "generate text\ntoken by token", np.tril(np.ones((5, 5)))),
             ("Encoder–decoder", "T5, BART, Whisper,\noriginal Transformer", "map input → output:\ntranslate, transcribe", None)]
    for a, (t, ex, use, m) in zip(axes, specs):
        a.set_title(t, loc="left", fontsize=15)
        a.axis("off")
        if m is not None:
            a.imshow(m, cmap="Purples", vmin=0, vmax=1.5, extent=(0.1, 0.6, 0.35, 0.95))
        else:
            a.imshow(np.ones((3, 3)), cmap="Purples", vmin=0, vmax=1.5, extent=(0.05, 0.3, 0.55, 0.85))
            a.imshow(np.tril(np.ones((3, 3))), cmap="Oranges", vmin=0, vmax=1.5, extent=(0.35, 0.6, 0.55, 0.85))
            a.text(0.33, 0.47, "enc  →  cross-attn  →  dec", ha="center", fontsize=10, color=MUTED)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.text(0.65, 0.8, ex, fontsize=12, color=INK, va="center")
        a.text(0.65, 0.5, use, fontsize=11.5, color=TEXT, va="center")
        a.text(0.1, 0.2, "attention mask", fontsize=10.5, color=MUTED)
    save(fig, path)


def kv_cache(path):
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.8)
    toks = ["The", "cat", "sat", "on", "the"]
    for i, t in enumerate(toks):
        box(ax, 0.2 + i * 1.3, 2.2, 1.1, 0.6, t, fc=TINT, size=12)
        box(ax, 0.2 + i * 1.3, 1.2, 1.1, 0.6, "K,V", fc="white", ec=TEAL, color=TEAL, size=11)
    box(ax, 0.2 + 5 * 1.3, 2.2, 1.1, 0.6, "mat?", fc=ORANGE, ec=ACCENT, size=12)
    box(ax, 0.2 + 5 * 1.3, 1.2, 1.1, 0.6, "Q", fc="white", ec=ACCENT, color=ACCENT, size=12)
    ax.add_patch(plt.Rectangle((0.1, 1.05), 6.4, 0.9, fill=False, ec=TEAL, lw=2, ls="--"))
    ax.text(0.15, 0.6, "KV cache: keys and values of earlier tokens are stored, not recomputed", color=TEAL, fontsize=12.5)
    ax.text(8.2, 2.6, "Without cache: step t recomputes all t tokens.\nWith cache: step t computes 1 new token\n(but memory grows with context length).", fontsize=12.5, color=TEXT, va="center")
    save(fig, path)


def moe(path):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    box(ax, 0.2, 1.9, 1.4, 0.8, "token", size=13)
    box(ax, 2.3, 1.9, 1.6, 0.8, "router", fc=ORANGE, ec=ACCENT, size=13)
    for i in range(8):
        y = 4.0 - i * 0.5
        on = i in (1, 5)
        box(ax, 5.0, y - 0.2, 2.2, 0.4, f"expert MLP {i + 1}", fc=TINT if on else "white", ec=PRIMARY if on else MUTED, color=INK if on else MUTED, size=10.5, weight="bold" if on else "normal")
        if on:
            arrow(ax, 3.9, 2.3, 5.0, y, color=ACCENT)
            arrow(ax, 7.2, y, 8.2, 2.3, color=ACCENT)
    box(ax, 8.2, 1.9, 1.6, 0.8, "weighted\nsum", size=12)
    arrow(ax, 1.6, 2.3, 2.3, 2.3)
    ax.text(10.0, 3.3, "Only the top-2 of 8 experts\nrun for this token:\nmany parameters, modest\ncompute per token.", fontsize=12, color=TEXT)
    ax.text(10.0, 1.2, "e.g. Mixtral 8×7B: ~47 B total,\n~13 B active per token", fontsize=11, color=MUTED)
    save(fig, path)


def vit_patches(path):
    from sklearn.datasets import load_sample_image

    img = load_sample_image("flower.jpg")[:, 64:64 + 416]
    img = img[::2, ::2][:208, :208]
    p = 52
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].imshow(img)
    for k in range(1, 4):
        axes[0].axhline(k * p, color="white", lw=2)
        axes[0].axvline(k * p, color="white", lw=2)
    axes[0].axis("off")
    axes[0].set_title("Split the image into patches", loc="left")
    b = axes[1]
    b.axis("off")
    b.set_title("…and treat the patches as a sequence of tokens", loc="left")
    for k in range(16):
        r, c = divmod(k, 4)
        ext = (k * 0.6, k * 0.6 + 0.52, 0.2, 0.72)
        b.imshow(img[r * p:(r + 1) * p, c * p:(c + 1) * p], extent=ext)
    b.set_xlim(-0.1, 9.7)
    b.set_ylim(-0.2, 1.3)
    b.text(0, 1.0, "patch embedding + position → transformer encoder (ViT)", fontsize=12.5, color=INK)
    save(fig, path)


def complexity(path):
    n = np.array([512, 2048, 8192, 32768, 131072])
    gb = n.astype(float) ** 2 * 2 / 1e9
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog(n, gb, marker="o", color=ACCENT, lw=2.5)
    for x, y in zip(n, gb):  # labels below-right of each point, away from the rising line and the axis
        ax.text(x * 1.18, y * 0.5, f"{y * 1e3:.3g} MB" if y < 1 else f"{y:.3g} GB", ha="left", va="top", fontsize=12)
    ax.set_xlim(300, 8e5)
    ax.set_ylim(1e-4, 200)
    ax.set_xlabel("sequence length n (tokens)")
    ax.set_ylabel("memory for one n × n score matrix (GB)")
    ax.set_title("Attention cost grows as n² (one head, fp16)", loc="left")
    save(fig, path)


FIGURES = {
    "cover": cover, "rnn_vs_attention": rnn_vs_attention, "qkv": qkv, "attn_worked": attn_worked,
    "causal_mask": causal_mask, "multihead": multihead, "positional": positional, "block": block,
    "three_archs": three_archs, "kv_cache": kv_cache, "moe": moe, "vit_patches": vit_patches, "complexity": complexity,
    **{n: asset(n) for n in ["bert_attention", "gpt2_attention", "chargpt_loss"]},
}
