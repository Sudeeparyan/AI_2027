"""Week 1 figures: landscape, tokens, next-token distribution, temperature, generative vs discriminative, latent variables."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from _style import ACCENT, GRID, INK, LAV, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save


def cover(path):
    # Samples from a learned-looking distribution: a mixture of curved clusters.
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(5, 6.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for k, c in enumerate([PRIMARY, ACCENT, "#8B5CF6", TEAL, LAV]):
        t = rng.uniform(0, np.pi, 260)
        r = 1.0 + 0.35 * k
        x = r * np.cos(t + k) + rng.normal(0, 0.08, t.size)
        y = r * np.sin(t + k) * 1.3 + rng.normal(0, 0.08, t.size)
        ax.scatter(x, y, s=rng.uniform(4, 40, t.size), color=c, alpha=0.55, lw=0)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, path, transparent=True)


def timeline(path):
    events = [
        (2013, "VAE", "latent variables"),
        (2014, "GAN", "adversarial training"),
        (2017, "Transformer", "attention"),
        (2018, "GPT-1, BERT", "pre-train, then adapt"),
        (2020, "GPT-3, DDPM", "few-shot LLMs;\ndiffusion"),
        (2021, "CLIP, DALL·E", "text meets images"),
        (2022, "Stable Diffusion,\nChatGPT", "open image model;\nchat assistant"),
        (2023, "GPT-4, Llama,\nClaude, Gemini", "multimodal;\nopen weights"),
        (2024, "Reasoning\nmodels, MCP", "test-time compute;\nvideo generation"),
        (2025, "Open reasoning,\nagents", "DeepSeek-R1;\nagent SDKs"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    ax.set_xlim(2012.3, 2025.9)
    ax.set_ylim(-3.1, 3.1)
    ax.plot([2012.5, 2025.7], [0, 0], color=MUTED, lw=2.5, zorder=1)
    for i, (yr, head, sub) in enumerate(events):
        up = i % 2 == 0
        c = SERIES[i % len(SERIES)]
        sgn = 1 if up else -1
        ax.scatter([yr], [0], s=160, color=c, zorder=3)
        ax.plot([yr, yr], [0, sgn * 0.55], color=c, lw=2, zorder=2)
        ax.text(yr, sgn * 0.62, head, ha="center", va="bottom" if up else "top", fontsize=14, fontweight="bold", color=INK, linespacing=1.05)
        ax.text(yr, sgn * 1.75, sub, ha="center", va="bottom" if up else "top", fontsize=12, color=MUTED, linespacing=1.05)
        ax.text(yr, -sgn * 0.18, str(yr), ha="center", va="top" if up else "bottom", fontsize=12, color=TEXT)
    save(fig, path)


def tokens(path):
    text = ["Gener", "ative", " AI", " writes", " one", " token", " at", " a", " time", "."]
    ids = [5648, 1413, 15592, 14652, 832, 4037, 520, 257, 640, 13]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    x = 0
    for i, (t, n) in enumerate(zip(text, ids)):
        w = 0.3 + 0.16 * len(t)
        c = SERIES[i % 4]
        ax.add_patch(plt.Rectangle((x, 0.5), w, 0.9, fc=c, alpha=0.18, ec=c, lw=1.5))
        ax.text(x + w / 2, 0.95, t.replace(" ", "·"), ha="center", va="center", fontsize=19, color=INK, fontweight="bold")
        ax.text(x + w / 2, 0.2, str(n), ha="center", va="center", fontsize=14, color=MUTED)
        x += w + 0.08
    ax.text(0, 1.7, "Text is split into tokens (sub-word pieces); each token has an integer ID.", fontsize=15, color=TEXT)
    ax.text(0, -0.25, "Token IDs shown are illustrative; each tokenizer has its own vocabulary.  · marks a leading space", fontsize=12, color=MUTED, style="italic")
    ax.set_xlim(0, x)
    ax.set_ylim(-0.5, 2.0)
    save(fig, path)


def next_token(path):
    words = ["mat", "floor", "sofa", "bed", "roof", "chair", "moon", "(others)"]
    p = np.array([0.42, 0.17, 0.12, 0.09, 0.06, 0.05, 0.01, 0.08])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(words[::-1], p[::-1], color=[PRIMARY] * 7 + [MUTED])
    bars[-1].set_color(ACCENT)
    for b, v in zip(bars, p[::-1]):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=12, color=TEXT)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("probability of the next token")
    ax.set_title('"The cat sat on the ___"', loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, path)


def temperature(path):
    z = np.array([2.0, 1.1, 0.8, 0.5, 0.1, -0.1, -1.5])
    labels = ["mat", "floor", "sofa", "bed", "roof", "chair", "moon"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, T, c in zip(axes, [0.5, 1.0, 1.5], [TEAL, PRIMARY, ACCENT]):
        q = np.exp(z / T) / np.exp(z / T).sum()
        ax.bar(labels, q, color=c)
        ax.set_title(f"T = {T}" + ("  (sharper)" if T < 1 else "  (flatter)" if T > 1 else "  (unchanged)"), loc="left")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="x", visible=False)
        ent = -(q * np.log2(q)).sum()
        ax.text(0.98, 0.92, f"top-1 = {q[0]:.2f}\nentropy = {ent:.2f} bits", transform=ax.transAxes, ha="right", va="top", fontsize=11, color=TEXT)
    axes[0].set_ylabel("sampling probability")
    save(fig, path)


def gen_vs_disc(path):
    rng = np.random.default_rng(3)
    a = rng.multivariate_normal([-1.2, 0.3], [[0.6, 0.25], [0.25, 0.4]], 120)
    b = rng.multivariate_normal([1.1, -0.2], [[0.5, -0.2], [-0.2, 0.6]], 120)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax in axes:
        ax.scatter(a[:, 0], a[:, 1], s=16, color=PRIMARY, alpha=0.7, label="class A (e.g. cat)")
        ax.scatter(b[:, 0], b[:, 1], s=16, color=ACCENT, alpha=0.7, label="class B (e.g. dog)")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2.8, 2.8)
        ax.set_xticks([])
        ax.set_yticks([])
    xs = np.linspace(-3.5, 3.5, 10)
    axes[0].plot(xs, 0.9 * xs + 0.1, color=INK, lw=2.5)
    axes[0].set_title("Discriminative: learn p(y | x)\n→ a boundary that separates classes", loc="left")
    axes[0].legend(loc="lower right", fontsize=10.5)
    for data, c in ((a, PRIMARY), (b, ACCENT)):
        mu, cov = data.mean(0), np.cov(data.T)
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
        for k in (1, 2):
            axes[1].add_patch(Ellipse(mu, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang, fc="none", ec=c, lw=2, alpha=0.9 / k))
    new = rng.multivariate_normal(a.mean(0), np.cov(a.T), 6)
    axes[1].scatter(new[:, 0], new[:, 1], s=140, marker="*", color=INK, zorder=5, label="new samples drawn from p(x | A)")
    axes[1].set_title("Generative: learn p(x) or p(x | y)\n→ a model of the data you can sample", loc="left")
    axes[1].legend(loc="lower right", fontsize=10.5)
    save(fig, path)


def latent(path):
    rng = np.random.default_rng(11)
    z = rng.normal(size=(700, 2))
    t = 1.5 * np.pi * (1 + 2 * (0.5 + 0.5 * np.tanh(z[:, 0])))
    x = np.c_[t * np.cos(t), t * np.sin(t)] / 6 + 0.05 * z[:, 1:2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.45})
    axes[0].scatter(z[:, 0], z[:, 1], s=8, c=z[:, 0], cmap="viridis")
    axes[0].set_title("Latent space: z ~ N(0, I)\n(simple, low-dimensional)", loc="left")
    axes[1].scatter(x[:, 0], x[:, 1], s=8, c=z[:, 0], cmap="viridis")
    axes[1].set_title("Data space: x = g(z)\n(complex, structured)", loc="left")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    fig.text(0.5, 0.5, "decoder / generator g", ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
    fig.patches.append(plt.matplotlib.patches.FancyArrow(0.44, 0.44, 0.12, 0, width=0.012, color=MUTED, transform=fig.transFigure, figure=fig))
    save(fig, path)


def course_map(path):
    weeks = [
        ("1", "Intro & responsible AI", 0), ("2", "VAEs", 1), ("3", "GANs", 1), ("4", "Diffusion", 1),
        ("5", "Transformers", 2), ("6", "LLM families", 2), ("7", "Prompting & eval", 2), ("8", "Fine-tuning", 3),
        ("9", "Multimodal foundations", 4), ("10", "Multimodal apps", 4), ("11", "Deployment", 5), ("12", "RAG & agents", 5),
    ]
    themes = ["Foundations", "Image generators", "Language models", "Adaptation", "Multimodal", "Systems"]
    colors = [INK, PRIMARY, TEAL, ACCENT, "#A21CAF", "#0369A1"]
    wrapped = {"Intro & responsible AI": "Intro &\nresponsible AI", "Multimodal foundations": "Multimodal\nfoundations",
               "Multimodal apps": "Multimodal\napps", "Prompting & eval": "Prompting\n& evaluation", "LLM families": "LLM\nfamilies",
               "RAG & agents": "RAG &\nagents", "Transformers": "Transform-\ners", "Fine-tuning": "Fine-\ntuning", "Deployment": "Deploy-\nment"}
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.35, 2.75)
    for i, (n, name, th) in enumerate(weeks):
        c = colors[th]
        col, row = i % 6, i // 6
        y = 1.45 - row * 1.35
        box(ax, col + 0.06, y, 0.88, 1.0, f"Week {n}\n{wrapped.get(name, name)}", fc="white", ec=c, color=c, size=14, lw=2)
    for th in range(len(themes)):
        idx = [i for i, w in enumerate(weeks) if w[2] == th]
        for row in sorted({i // 6 for i in idx}):  # a theme may continue onto the next row
            cols = [i % 6 for i in idx if i // 6 == row]
            a, b = min(cols), max(cols) + 1
            y = 1.45 - row * 1.35 - 0.12
            ax.plot([a + 0.1, b - 0.1], [y, y], color=colors[th], lw=6, solid_capstyle="round")
            label = themes[th] if row == idx[0] // 6 else themes[th] + " (cont.)"
            ax.text((a + b) / 2, y - 0.06, label, ha="center", va="top", fontsize=14, color=colors[th], fontweight="bold")
    ax.text(0.06, 2.6, "Responsible AI and evaluation run through every week  →  group project (60%) and exam (40%)", fontsize=15, color=TEXT)
    save(fig, path)


def families(path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), gridspec_kw={"hspace": 0.35, "wspace": 0.12})
    axes = axes.ravel()
    titles = ["Autoregressive (LLMs) · weeks 5–6", "VAE · week 2", "GAN · week 3", "Diffusion / flow · week 4"]
    for ax, t in zip(axes, titles):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(t, fontsize=17, loc="left")
    orange = "#FFF1E6"
    a = axes[0]
    for i, w in enumerate(["The", "cat", "sat", "on", "?"]):
        box(a, 0.01 + i * 0.198, 0.45, 0.17, 0.3, w, fc=TINT if w != "?" else orange, ec=PRIMARY if w != "?" else ACCENT, size=16)
    a.text(0.0, 0.12, "Predict the next token from the previous ones,\nappend it, repeat.", fontsize=14, color=TEXT)
    v = axes[1]
    box(v, 0.0, 0.42, 0.22, 0.34, "image x", size=15)
    box(v, 0.39, 0.47, 0.2, 0.24, "latent z", fc=orange, ec=ACCENT, size=15)
    box(v, 0.76, 0.42, 0.24, 0.34, "rebuilt x̂", size=15)
    arrow(v, 0.22, 0.59, 0.39, 0.59)
    arrow(v, 0.59, 0.59, 0.76, 0.59)
    v.text(0.28, 0.8, "encoder", fontsize=13, color=MUTED)
    v.text(0.62, 0.8, "decoder", fontsize=13, color=MUTED)
    v.text(0.0, 0.12, "Compress to a latent code and decode back;\nsample new z to generate.", fontsize=14, color=TEXT)
    g = axes[2]
    box(g, 0.0, 0.45, 0.16, 0.26, "noise z", fc=orange, ec=ACCENT, size=14)
    box(g, 0.26, 0.45, 0.24, 0.26, "Generator G", size=15)
    box(g, 0.62, 0.45, 0.36, 0.26, "Discriminator D:\nreal or fake?", size=14)
    arrow(g, 0.16, 0.58, 0.26, 0.58)
    arrow(g, 0.5, 0.58, 0.62, 0.58)
    g.text(0.0, 0.12, "Two networks compete: G learns to fool D,\nD learns to catch G.", fontsize=14, color=TEXT)
    d = axes[3]
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[-1:1:16j, -1:1:16j]
    target = np.exp(-((xx ** 2 + yy ** 2) * 3)) - 0.6 * np.exp(-(((xx - 0.3) ** 2 + (yy + 0.2) ** 2) * 18))
    for i, s in enumerate([1.0, 0.65, 0.35, 0.12, 0.0]):
        img = np.sqrt(1 - s ** 2) * target + s * rng.normal(0, 0.6, target.shape)
        d.imshow(img, extent=(0.01 + i * 0.198, 0.18 + i * 0.198, 0.42, 0.78), cmap="magma", vmin=-1, vmax=1)
        if i < 4:
            arrow(d, 0.185 + i * 0.198, 0.6, 0.207 + i * 0.198, 0.6, lw=1.2)
    d.set_xlim(0, 1)
    d.set_ylim(0, 1)
    d.text(0.0, 0.12, "Start from pure noise and remove a little\nnoise at each step.", fontsize=14, color=TEXT)
    save(fig, path)


FIGURES = {
    "cover": cover, "timeline": timeline, "tokens": tokens, "next_token": next_token,
    "temperature": temperature, "gen_vs_disc": gen_vs_disc, "latent": latent,
    "course_map": course_map, "families": families,
}
