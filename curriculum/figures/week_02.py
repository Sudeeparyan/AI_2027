"""Week 2 figures: VAE diagrams (matplotlib) plus REAL training results from curriculum/assets/week_02."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_02"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    img = plt.imread(ASSETS / "manifold.png")
    img = img[int(img.shape[0] * 0.075):]  # drop the figure title
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(1 - img[..., :3].mean(axis=2) if img.ndim == 3 else 1 - img, cmap="magma")
    ax.axis("off")
    fig.patch.set_alpha(0)
    save(fig, path, transparent=True)


def vae_arch(path):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    box(ax, 0.1, 1.6, 1.5, 1.6, "input\nimage x", size=14)
    box(ax, 2.2, 1.3, 2.0, 2.2, "Encoder\nqφ(z | x)", fc="white", size=15)
    box(ax, 4.9, 2.6, 1.3, 0.9, "μ", fc=TINT, size=17)
    box(ax, 4.9, 1.3, 1.3, 0.9, "log σ²", fc=TINT, size=15)
    box(ax, 6.9, 1.8, 1.9, 1.2, "z = μ + σ ⊙ ε", fc=ORANGE, ec=ACCENT, size=15)
    box(ax, 6.95, 3.6, 1.8, 0.8, "ε ~ N(0, I)", fc="white", ec=ACCENT, size=13, weight="normal")
    box(ax, 9.4, 1.3, 2.0, 2.2, "Decoder\npθ(x | z)", fc="white", size=15)
    box(ax, 11.8, 1.6, 1.1, 1.6, "x̂", size=18)
    arrow(ax, 1.6, 2.4, 2.2, 2.4)
    arrow(ax, 4.2, 2.8, 4.9, 3.05)
    arrow(ax, 4.2, 2.0, 4.9, 1.75)
    arrow(ax, 6.2, 3.05, 6.9, 2.6)
    arrow(ax, 6.2, 1.75, 6.9, 2.2)
    arrow(ax, 7.85, 3.6, 7.85, 3.0, color=ACCENT)
    arrow(ax, 8.8, 2.4, 9.4, 2.4)
    arrow(ax, 11.4, 2.4, 11.8, 2.4)
    ax.text(0.9, 0.55, "Loss = reconstruction(x, x̂)  +  β · KL( qφ(z | x) ‖ N(0, I) )", fontsize=16, color=INK, fontweight="bold")
    ax.text(5.55, 4.25, "the code is a distribution,\nnot a single point", fontsize=12, color=MUTED, ha="center")
    save(fig, path)


def reparam(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for a in axes:
        a.axis("off")
        a.set_xlim(0, 6)
        a.set_ylim(0, 4.6)
    a = axes[0]
    a.set_title("Without the trick: sampling blocks gradients", loc="left", fontsize=15, color=INK)
    box(a, 0.2, 2.9, 1.2, 0.8, "μ, σ", fc=TINT, size=15)
    box(a, 2.3, 2.7, 1.6, 1.2, "sample\nz ~ N(μ, σ²)", fc=ORANGE, ec=ACCENT, size=13)
    box(a, 4.6, 2.9, 1.2, 0.8, "loss", size=15)
    arrow(a, 1.4, 3.3, 2.3, 3.3)
    arrow(a, 3.9, 3.3, 4.6, 3.3)
    a.annotate("", xy=(3.9, 2.2), xytext=(4.6, 2.2), arrowprops=dict(arrowstyle="-|>", color="#B91C1C", lw=2, ls="--"))
    a.text(3.3, 1.6, "✗ no gradient through a random draw", color="#B91C1C", fontsize=13, ha="center")
    b = axes[1]
    b.set_title("With the trick: randomness becomes an input", loc="left", fontsize=15, color=INK)
    box(b, 0.2, 2.9, 1.2, 0.8, "μ, σ", fc=TINT, size=15)
    box(b, 0.2, 0.9, 1.4, 0.8, "ε ~ N(0, I)", fc="white", ec=ACCENT, size=13, weight="normal")
    box(b, 2.3, 2.7, 1.6, 1.2, "z = μ + σ ε\n(deterministic)", fc=TINT, size=13)
    box(b, 4.6, 2.9, 1.2, 0.8, "loss", size=15)
    arrow(b, 1.4, 3.3, 2.3, 3.3)
    arrow(b, 1.6, 1.3, 2.6, 2.7, color=ACCENT)
    arrow(b, 3.9, 3.3, 4.6, 3.3)
    b.annotate("", xy=(1.4, 3.9), xytext=(4.6, 3.9), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2, connectionstyle="arc3,rad=0.25"))
    b.text(3.0, 4.35, "✓ gradients flow back to μ and σ", color=TEAL, fontsize=13, ha="center")
    save(fig, path)


def kl_gaussians(path):
    x = np.linspace(-4, 5, 400)
    p = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    cases = [(0, 1), (1, 1), (0, 0.5), (2, 0.5)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for a, (m, s) in zip(axes, cases):
        q = np.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
        kl = 0.5 * (m ** 2 + s ** 2 - np.log(s ** 2) - 1)
        a.fill_between(x, p, color=MUTED, alpha=0.25, label="prior N(0, 1)")
        a.plot(x, q, color=PRIMARY, lw=2.5, label="encoder q")
        a.set_title(f"μ = {m}, σ = {s}\nKL = {kl:.2f}", loc="left", fontsize=14)
        a.set_yticks([])
        a.grid(False)
    axes[0].legend(fontsize=10.5, loc="upper left")
    save(fig, path)


def beta_tradeoff(path):
    m = json.loads((ASSETS / "metrics.json").read_text())["beta"]
    betas = list(m.keys())
    rec = [m[b]["rec"] for b in betas]
    kl = [m[b]["kl"] for b in betas]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar([f"β = {b}" for b in betas], rec, color=PRIMARY)
    ax[0].set_title("Reconstruction loss (lower = sharper)", loc="left")
    ax[1].bar([f"β = {b}" for b in betas], kl, color=ACCENT)
    ax[1].set_title("KL to the prior (lower = closer to N(0, I))", loc="left")
    for a, vals in zip(ax, (rec, kl)):
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=12)
        a.grid(axis="x", visible=False)
    save(fig, path)


def ae_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.fill([2.0, 4.6, 4.6, 2.0], [0.4, 1.2, 2.2, 3.0], color=TINT, ec=PRIMARY, lw=1.5)
    ax.fill([7.4, 10.0, 10.0, 7.4], [1.2, 0.4, 3.0, 2.2], color=TINT, ec=PRIMARY, lw=1.5)
    box(ax, 5.2, 1.2, 1.6, 1.0, "code z", fc=ORANGE, ec=ACCENT, size=15)
    box(ax, 0.1, 1.0, 1.5, 1.4, "x\n784 pixels", size=13)
    box(ax, 10.4, 1.0, 1.5, 1.4, "x̂\n784 pixels", size=13)
    ax.text(3.3, 1.7, "Encoder", ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(8.7, 1.7, "Decoder", ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(6.0, 0.8, "bottleneck, e.g. 2–32 numbers", ha="center", fontsize=12, color=MUTED)
    arrow(ax, 1.6, 1.7, 2.0, 1.7)
    arrow(ax, 4.6, 1.7, 5.2, 1.7)
    arrow(ax, 6.8, 1.7, 7.4, 1.7)
    arrow(ax, 10.0, 1.7, 10.4, 1.7)
    save(fig, path)


FIGURES = {
    "cover": cover,
    "vae_arch": vae_arch,
    "reparam": reparam,
    "kl_gaussians": kl_gaussians,
    "beta_tradeoff": beta_tradeoff,
    "ae_diagram": ae_diagram,
    **{n: asset(n) for n in ["training_curves", "latent_ae_vs_vae", "samples_ae_vs_vae", "reconstructions",
                             "manifold", "interpolation", "beta_samples", "anomaly"]},
}
