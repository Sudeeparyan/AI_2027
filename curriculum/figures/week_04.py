"""Week 4 figures: diffusion diagrams (matplotlib) plus REAL results from curriculum/assets/week_04."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_04"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    src = ASSETS / "sd_seeds.png"
    img = plt.imread(src)
    h, w = img.shape[:2]
    crop = img[int(h * 0.12):, : w // 2]  # first two generated images, without titles
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(crop)
    ax.axis("off")
    save(fig, path, transparent=True)


def schedules(path):
    T = 1000
    t = np.arange(T)
    beta = np.linspace(1e-4, 0.02, T)
    abar_lin = np.cumprod(1 - beta)
    s = 0.008
    f = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    abar_cos = f / f[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    ax[0].plot(t, abar_lin, color=PRIMARY, lw=2.5, label="linear β (DDPM)")
    ax[0].plot(t, abar_cos, color=ACCENT, lw=2.5, label="cosine (improved DDPM)")
    ax[0].set_title("ᾱ_t: fraction of signal variance kept", loc="left")
    ax[0].set_xlabel("timestep t")
    ax[0].legend()
    snr_lin = abar_lin / (1 - abar_lin)
    snr_cos = abar_cos / (1 - abar_cos + 1e-12)
    ax[1].semilogy(t, snr_lin, color=PRIMARY, lw=2.5, label="linear β")
    ax[1].semilogy(t[1:-1], snr_cos[1:-1], color=ACCENT, lw=2.5, label="cosine")
    ax[1].set_title("Signal-to-noise ratio ᾱ_t / (1 − ᾱ_t)", loc="left")
    ax[1].set_xlabel("timestep t")
    ax[1].legend()
    save(fig, path)


def trilemma(path):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.axis("off")
    pts = {"High quality": (0.5, 0.95), "Diversity / coverage": (0.02, 0.08), "Fast sampling": (0.98, 0.08)}
    tri = np.array(list(pts.values()) + [pts["High quality"]])
    ax.plot(tri[:, 0], tri[:, 1], color=MUTED, lw=2)
    for name, (x, y) in pts.items():
        ax.text(x, y + (0.04 if y > 0.5 else -0.07), name, ha="center", fontsize=15, fontweight="bold", color=INK)
    fams = [("GAN", (0.63, 0.52), ACCENT, "sharp, fast;\nmode collapse"),
            ("VAE", (0.5, 0.27), PRIMARY, "coverage, fast; blurry"),
            ("Diffusion", (0.37, 0.52), TEAL, "quality + coverage;\nmany steps")]
    for n, (x, y), c, sub in fams:
        ax.scatter([x], [y], s=2300 if len(n) > 3 else 1100, color=c, alpha=0.9)
        ax.text(x, y, n, ha="center", va="center", color="white", fontsize=11 if len(n) > 3 else 12, fontweight="bold")
        ax.text(x, y - 0.1, sub, ha="center", va="top", fontsize=11.5, color=TEXT)
    ax.annotate("distillation /\nflow matching", xy=(0.5, 0.6), xytext=(0.36, 0.72), fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.08, 1.05)
    save(fig, path)


def unet_dit(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    a = axes[0]
    a.axis("off")
    a.set_xlim(0, 10)
    a.set_ylim(0, 6)
    a.set_title("U-Net (Stable Diffusion 1.x/2.x, SDXL)", loc="left", fontsize=15)
    levels = [(0.3, 4.2, 1.6, 1.2), (1.6, 2.8, 1.2, 1.0), (2.9, 1.4, 0.9, 0.8)]
    for x, y, w, h in levels:
        box(a, x, y, w, h, "", fc=TINT)
    for x, y, w, h in levels:
        box(a, 9.7 - x - w, y, w, h, "", fc=TINT)
    box(a, 4.2, 0.3, 1.6, 0.8, "bottleneck", fc=ORANGE, ec=ACCENT, size=11)
    for x, y, w, h in levels:
        a.annotate("", xy=(9.7 - x - w, y + h / 2), xytext=(x + w, y + h / 2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, ls="--"))
    a.text(5, 5.55, "skip connections", ha="center", color=MUTED, fontsize=11)
    a.text(0.05, 2.65, "downsample\n(conv + attention)", fontsize=10.5, color=TEXT, va="top")
    a.text(7.3, 4.1, "upsample", fontsize=10.5, color=TEXT, va="top")
    a.text(0.2, 0.3, "timestep and text enter\nevery block", fontsize=10.5, color=ACCENT)
    b = axes[1]
    b.axis("off")
    b.set_xlim(0, 10)
    b.set_ylim(0, 6)
    b.set_title("Diffusion Transformer, DiT (SD3, FLUX, video models)", loc="left", fontsize=15)
    for i in range(4):
        for j in range(4):
            b.add_patch(plt.Rectangle((0.3 + j * 0.45, 3.4 + i * 0.45), 0.4, 0.4, fc=TINT, ec=PRIMARY))
    b.text(1.2, 3.25, "latent split into\npatches (tokens)", ha="center", va="top", fontsize=10.5, color=TEXT)
    b.text(5.6, 5.45, "transformer blocks", ha="center", fontsize=12, color=INK)
    for k in range(4):
        box(b, 3.0 + k * 1.35, 3.3, 1.15, 1.9, f"block\n{k + 1}" if k < 3 else "…", size=12)
    box(b, 3.0, 0.9, 5.2, 1.3, "conditioning: timestep + text tokens\n(adaLN / joint attention)", fc=ORANGE, ec=ACCENT, size=11.5, weight="normal")
    arrow(b, 2.2, 4.25, 3.0, 4.25)
    b.text(8.5, 4.25, "un-patch\n→ noise\nprediction", fontsize=11.5, color=TEXT, va="center")
    save(fig, path)


def ldm_pipeline(path):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    box(ax, 0.1, 3.1, 2.2, 1.0, "prompt: 'a lighthouse\nat sunset'", fc="white", size=11.5, weight="normal")
    box(ax, 2.8, 3.1, 2.0, 1.0, "Text encoder\n(CLIP / T5)", fc=TINT, size=12)
    box(ax, 0.1, 0.8, 2.2, 1.2, "random latent\nz_T ~ N(0, I)\n64 × 64 × 4", fc=ORANGE, ec=ACCENT, size=11)
    box(ax, 3.3, 0.6, 3.6, 1.6, "Denoiser (U-Net or DiT)\nrepeat T steps:\nz_t → z_{t−1}", fc="white", size=12.5)
    box(ax, 7.6, 0.8, 1.9, 1.2, "clean latent\nz_0", fc=TINT, size=12)
    box(ax, 10.1, 0.8, 1.6, 1.2, "VAE\ndecoder", fc=TINT, size=12)
    box(ax, 12.0, 0.7, 0.9, 1.4, "image\n512²", fc="white", size=10.5)
    arrow(ax, 2.3, 3.6, 2.8, 3.6)
    ax.annotate("", xy=(5.1, 2.2), xytext=(3.8, 3.1), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
    ax.text(4.9, 2.62, "cross-attention", color=ACCENT, fontsize=11)
    arrow(ax, 2.3, 1.4, 3.3, 1.4)
    arrow(ax, 6.9, 1.4, 7.6, 1.4)
    arrow(ax, 9.5, 1.4, 10.1, 1.4)
    arrow(ax, 11.7, 1.4, 12.0, 1.4)
    ax.annotate("", xy=(4.6, 0.6), xytext=(5.6, 0.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5, connectionstyle="arc3,rad=-0.9"))
    ax.text(7.7, 3.4, "All the expensive denoising happens in the\nsmall latent space (48× fewer numbers).", fontsize=12.5, color=INK)
    save(fig, path)


def cfg(path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_xlim(-0.3, 6.2)
    ax.set_ylim(-0.5, 4.3)
    ax.axis("off")
    o = np.array([0, 0])
    u = np.array([2.6, 0.5])
    c = np.array([3.0, 1.0])
    # Arrow lengths are schematic but monotonic in w (w = 7.5 is drawn shortened to fit the panel).
    ax.plot([u[0], u[0] + 6.8 * (c - u)[0]], [u[1], u[1] + 6.8 * (c - u)[1]], ls="--", color=MUTED, lw=1)
    for w, shown, col in ((1, 1, PRIMARY), (3, 3, ACCENT), (7.5, 6.5, "#B91C1C")):
        p = u + shown * (c - u)
        ax.annotate("", xy=p, xytext=o, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.text(p[0] + 0.12, p[1] - 0.05, f"w = {w}" + ("  (conditional)" if w == 1 else "  (over-saturated)" if w > 5 else "  (typical range)"),
                color=col, fontsize=12, va="center")
    ax.annotate("", xy=u, xytext=o, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.2))
    ax.text(u[0] + 0.1, u[1] - 0.3, "unconditional ε(x_t, ∅)", color=MUTED, fontsize=12)
    ax.text(-0.25, 4.05, "Guided prediction = unconditional + w · (conditional − unconditional)", fontsize=12.5, color=INK, fontweight="bold")
    save(fig, path)


def flow_paths(path):
    rng = np.random.default_rng(3)
    data = np.c_[rng.choice([-2.0, 2.0], 12) + rng.normal(0, 0.2, 12), rng.choice([-1.2, 1.2], 12) + rng.normal(0, 0.2, 12)]
    noise = rng.normal(0, 1, (12, 2)) * 1.0 + np.array([0, -4.5])
    t = np.linspace(0, 1, 50)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for a, title, kind in ((axes[0], "Diffusion (variance-preserving) paths: curved", "vp"), (axes[1], "Flow matching / rectified flow: straight", "rf")):
        for x0, z in zip(data, noise):
            if kind == "vp":
                ang = t * np.pi / 2
                pts = np.cos(ang) * z + np.sin(ang) * x0
            else:
                pts = (1 - t) * z + t * x0
            a.plot(pts[:, 0], pts[:, 1], color=PRIMARY if kind == "vp" else TEAL, lw=1.6, alpha=0.8)
        a.scatter(noise[:, 0], noise[:, 1], color=MUTED, s=30, label="noise sample")
        a.scatter(data[:, 0], data[:, 1], color=ACCENT, s=40, label="data sample")
        a.set_title(title, loc="left")
        a.set_xticks([])
        a.set_yticks([])
        a.legend(loc="upper left", fontsize=10.5)
        a.set_aspect("equal")
    fig.text(0.5, 0.01, "Straighter paths can be followed accurately with far fewer steps (SD3, FLUX use rectified flow).", ha="center", fontsize=12.5, color=TEXT)
    save(fig, path)


def seeds_negative(path):
    a = plt.imread(ASSETS / "sd_seeds.png")
    b = plt.imread(ASSETS / "sd_negative.png")
    # one row (seeds left, negative prompt right) so each image is as large as the slide allows
    ratios = [a.shape[1] / a.shape[0], b.shape[1] / b.shape[0]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 14 / sum(ratios) + 0.5), gridspec_kw={"width_ratios": ratios, "wspace": 0.04})
    for ax, im, t in ((axes[0], a, "Same prompt, four seeds"), (axes[1], b, "Negative prompt (same seed)")):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(t, loc="left", fontsize=14)
    save(fig, path)


FIGURES = {
    "cover": cover,
    "schedules": schedules,
    "trilemma": trilemma,
    "unet_dit": unet_dit,
    "ldm_pipeline": ldm_pipeline,
    "cfg": cfg,
    "flow_paths": flow_paths,
    "seeds_negative": seeds_negative,
    **{n: asset(n) for n in ["forward_mnist", "reverse_2d", "reverse_mnist", "mnist_samples", "sd_guidance",
                             "sd_steps", "sd_seeds", "sd_negative", "sd_turbo"]},
}
