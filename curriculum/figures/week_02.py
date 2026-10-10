"""Week 2 figures: VAE diagrams (matplotlib) plus REAL training results from curriculum/assets/week_02."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, TEXT, TINT, arrow, box, routed_arrow, save

# Full-width lecture figures use at least 18pt source labels.
_LECTURE_LABELS = {"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 20,
                   "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_02"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def latent_ae_vs_vae(path):
    """Keep the archived experiment unchanged; replace its overstrong headings."""
    image = plt.imread(ASSETS / "latent_ae_vs_vae.png")
    # The first 130 rows contain only the old headings. Axes, points and labels
    # remain the exact original experimental graphic below that header.
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.imshow(image[130:], extent=(0, 1880, 936, 0))
    ax.set_xlim(0, 1880)
    ax.set_ylim(936, -130)
    ax.axis("off")
    ax.text(150, -92, "Plain AE: encoded points", fontsize=18, color=INK,
            fontweight="bold", va="top")
    ax.text(150, -42, "Inspect gaps and random-code outputs", fontsize=18,
            color=INK, va="top")
    ax.text(962, -92, "VAE: codes regularised towards N(0, I)", fontsize=18,
            color=INK, fontweight="bold", va="top")
    ax.text(962, -42, "Inspect coverage and decoded samples", fontsize=18,
            color=INK, va="top")
    save(fig, path)


def anomaly(path):
    """Keep the measured histogram bars pixel for pixel; redraw its small text at lecture size.

    The saved plot (200 dpi) put 10–13 pt labels on the slide and a title that rounded AUC
    0.9995 up to 1.000. Only the plot interior is reused: its gridlines fix the data scale.
    """
    from matplotlib.patches import Patch
    auc = json.loads((ASSETS / "metrics.json").read_text(encoding="utf-8"))["anomaly_auc"]
    image = plt.imread(ASSETS / "anomaly.png")[..., :3].copy()
    grid = image[300, 486].copy()  # gridline colour
    # Bar colours, sampled from a digit bar, a clothing bar and a bin where both overlap.
    colours = [tuple(image[650, 250]), tuple(image[700, 600]), tuple(image[715, 290])]
    # Old legend (x 745–1385, y 100–200 px) is cleared; the gridlines that ran behind it are restored.
    image[100:200, 745:1385] = 1.0
    image[100:200, [747, 748, 1008, 1009, 1270, 1271]] = grid
    image[[160, 161], 745:1385] = grid
    # Plot interior is x 170–1408 and y 77–722 px. Gridline centres give the scale:
    # BCE 0, 500, …, 2000 at x = 225, 486, 748, 1009, 1271 (0.523 px per unit);
    # counts 200, …, 1000 at y = 612, 499, 386, 273, 161 (0.56375 px per image) above the x axis at y = 724.
    bce = lambda x: (x - 225.0) / 0.523
    count = lambda y: (724.0 - y) / 0.56375
    extent = (bce(170), bce(1409), count(723), count(77))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.imshow(image[77:723, 170:1409], extent=extent, aspect="auto", interpolation="nearest")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(0, extent[3])
    ax.grid(False)  # the measured image already carries its gridlines
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    ax.tick_params(labelsize=18)
    ax.set_xlabel("reconstruction error (BCE per image)", fontsize=18)
    ax.set_ylabel("number of test images", fontsize=18)
    ax.set_title(f"Measured ROC AUC = {auc:.4f}", loc="left", fontsize=18, color=INK, fontweight="bold")
    labels = ["MNIST digits (seen type)", "Fashion-MNIST clothing (unseen)", "both overlap"]
    ax.legend([Patch(color=c) for c in colours], labels, loc="upper right", fontsize=18,
              frameon=True, framealpha=1.0, facecolor="white", edgecolor="white")
    save(fig, path)


def manifold(path):
    """Keep the measured 15 × 15 decoded grid; replace the old "(z = 2)" title with code axes."""
    # Grid cells occupy x 30–1034 and y 80–1076 px of the saved image (measured from the cell ink).
    image = plt.imread(ASSETS / "manifold.png")[80:1076, 30:1034]
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.imshow(image, extent=(0, 15, 15, 0))
    # Codes were norm.ppf(linspace(0.03, 0.97, 15)): first, middle and last values are −1.88, 0 and +1.88.
    ticks, values = [0.5, 7.5, 14.5], ["−1.88", "0", "+1.88"]
    ax.set_xticks(ticks, values, fontsize=20)
    ax.set_yticks(ticks, values[::-1], fontsize=20)
    ax.set_xlabel("first code number z₁", fontsize=20)
    ax.set_ylabel("second code number z₂", fontsize=20)
    ax.grid(False)  # grid lines would cross the decoded digits
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, path)


def latent_ae_vs_vae_slide(path):
    """Enlarge labels around the exact archived scatter panels; do not refit."""
    image = plt.imread(ASSETS / "latent_ae_vs_vae.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, limits, title in zip(axes, [(151, 827), (963, 1639)],
                                ["AE: encoded points", "VAE: encoded means"]):
        left, right = limits
        ax.imshow(image[131:934, left:right], aspect="equal")
        ax.set_title(title, loc="left", fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("latent coordinate z₁", fontsize=20)
        ax.set_ylabel("latent coordinate z₂", fontsize=20)
    # Digit colours and every scatter point are copied from the saved result.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=plt.get_cmap("tab10")(i), marker="o", linestyle="", label=str(i)) for i in range(10)]
    fig.legend(handles=handles, title="Digit colour", ncol=10, loc="lower center",
               bbox_to_anchor=(0.5, -0.15), fontsize=18, title_fontsize=18,
               handletextpad=0.15, columnspacing=0.7)
    save(fig, path)


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
    box(ax, 0.1, 1.6, 1.5, 1.6, "input\nimage x", size=18)
    box(ax, 2.2, 1.3, 2.0, 2.2, "Encoder\nqφ(z | x)", fc="white", size=18)
    box(ax, 4.9, 2.6, 1.3, 0.9, "μ", fc=TINT, size=18)
    box(ax, 4.9, 1.3, 1.3, 0.9, "log σ²", fc=TINT, size=18)
    box(ax, 6.9, 1.8, 1.9, 1.2, "z = μ + σ ⊙ ε", fc=ORANGE, ec=ACCENT, size=18)
    box(ax, 6.95, 3.6, 1.8, 0.8, "ε ~ N(0, I)", fc="white", ec=ACCENT, size=18, weight="normal")
    box(ax, 9.4, 1.3, 2.0, 2.2, "Decoder\npθ(x | z)", fc="white", size=18)
    box(ax, 11.8, 1.6, 1.1, 1.6, "x̂", size=18)
    arrow(ax, 1.6, 2.4, 2.2, 2.4)
    arrow(ax, 4.2, 2.8, 4.9, 3.05)
    arrow(ax, 4.2, 2.0, 4.9, 1.75)
    arrow(ax, 6.2, 3.05, 6.9, 2.6)
    arrow(ax, 6.2, 1.75, 6.9, 2.2)
    arrow(ax, 7.85, 3.6, 7.85, 3.0, color=ACCENT)
    arrow(ax, 8.8, 2.4, 9.4, 2.4)
    arrow(ax, 11.4, 2.4, 11.8, 2.4)
    ax.text(0.9, 0.55, "Loss = reconstruction(x, x̂)  +  β · KL( qφ(z | x) ‖ N(0, I) )", fontsize=18, color=INK, fontweight="bold")
    ax.text(4.55, 4.25, "the code is a distribution,\nnot a single point", fontsize=18, color=MUTED, ha="center")
    save(fig, path)


def reparam(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for a in axes:
        a.axis("off")
        a.set_xlim(0, 6)
        a.set_ylim(0.6, 5.0)
    red = "#B91C1C"
    a = axes[0]
    a.set_title("Sampling directly: the gradient stops", loc="left", fontsize=18, color=INK)
    box(a, 0.2, 2.9, 1.2, 0.8, "μ, σ", fc=TINT, size=18)
    box(a, 2.3, 2.7, 1.6, 1.2, "sample\nz ~ N(μ, σ²)", fc=ORANGE, ec=ACCENT, size=18)
    box(a, 4.6, 2.9, 1.2, 0.8, "loss", size=18)
    arrow(a, 1.4, 3.5, 2.3, 3.5)
    arrow(a, 3.9, 3.5, 4.6, 3.5)
    # The backward (gradient) arrow reaches the random draw and stops there.
    a.annotate("", xy=(3.9, 3.05), xytext=(4.6, 3.05), arrowprops=dict(arrowstyle="-|>", color=red, lw=2, ls="--"))
    a.text(1.0, 1.9, "✗ gradient cannot pass through a random draw,\n   so μ and σ (and the encoder) never learn", color=red, fontsize=17, va="top")
    b = axes[1]
    b.set_title("Reparameterised: noise is an input", loc="left", fontsize=18, color=INK)
    box(b, 0.2, 2.9, 1.2, 0.8, "μ, σ", fc=TINT, size=18)
    box(b, 0.2, 0.9, 1.4, 0.8, "ε ~ N(0, I)", fc="white", ec=ACCENT, size=18, weight="normal")
    box(b, 2.3, 2.7, 1.6, 1.2, "z = μ + σ ε", fc=TINT, size=18)
    box(b, 4.6, 2.9, 1.2, 0.8, "loss", size=18)
    arrow(b, 1.4, 3.5, 2.3, 3.5)
    routed_arrow(b, [(1.6, 1.3), (3.1, 1.3), (3.1, 2.7)], color=ACCENT)
    arrow(b, 3.9, 3.5, 4.6, 3.5)
    # Gradients travel back from the loss through z to μ and σ (above the boxes, so no line crosses).
    routed_arrow(b, [(5.2, 3.7), (5.2, 4.25), (0.8, 4.25), (0.8, 3.7)], color=TEAL, dashed=True)
    b.text(3.0, 4.6, "✓ gradient flows back to μ and σ", color=TEAL, fontsize=17, ha="center", va="center")
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
        a.set_title(f"μ = {m}, σ = {s}\nKL = {kl:.2f}", loc="left", fontsize=18)
        a.set_yticks([])
        a.grid(False)
    axes[0].legend(fontsize=18, loc="upper left")
    save(fig, path)


def beta_tradeoff(path):
    m = json.loads((ASSETS / "metrics.json").read_text())["beta"]
    betas = list(m.keys())
    rec = [m[b]["rec"] for b in betas]
    kl = [m[b]["kl"] for b in betas]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar([f"β = {b}" for b in betas], rec, color=PRIMARY)
    ax[0].set_title("Reconstruction BCE\n(lower: closer pixel fit)", loc="left")
    ax[1].bar([f"β = {b}" for b in betas], kl, color=ACCENT)
    ax[1].set_title("KL to N(0, I)\n(lower: closer prior fit)", loc="left")
    for a, vals in zip(ax, (rec, kl)):
        a.set_ylim(0, max(vals) * 1.2)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=18)
        a.grid(axis="x", visible=False)
    save(fig, path)


def ae_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.fill([2.0, 4.6, 4.6, 2.0], [0.4, 1.2, 2.2, 3.0], color=TINT, ec=PRIMARY, lw=1.5)
    ax.fill([7.4, 10.0, 10.0, 7.4], [1.2, 0.4, 3.0, 2.2], color=TINT, ec=PRIMARY, lw=1.5)
    box(ax, 5.2, 1.2, 1.6, 1.0, "code z", fc=ORANGE, ec=ACCENT, size=18)
    box(ax, 0.1, 1.0, 1.5, 1.4, "x\n784 pixels", size=18)
    box(ax, 10.4, 1.0, 1.5, 1.4, "x̂\n784 pixels", size=18)
    ax.text(3.3, 1.7, "Encoder", ha="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(8.7, 1.7, "Decoder", ha="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(6.0, 0.16, "bottleneck, e.g. 2–32 numbers", ha="center", fontsize=18, color=MUTED)
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
    "latent_ae_vs_vae": latent_ae_vs_vae,
    "latent_ae_vs_vae_slide": latent_ae_vs_vae_slide,
    **{n: asset(n) for n in ["training_curves", "samples_ae_vs_vae", "reconstructions",
                             "interpolation", "beta_samples"]},
    "manifold": manifold,
    "anomaly": anomaly,
}


def _with_lecture_labels(make):
    def draw(path):
        with plt.rc_context(_LECTURE_LABELS):
            make(path)
    return draw


FIGURES = {name: _with_lecture_labels(make) for name, make in FIGURES.items()}
