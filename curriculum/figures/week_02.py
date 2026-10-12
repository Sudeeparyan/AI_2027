"""Week 2 figures: VAE diagrams (matplotlib) plus REAL training results from curriculum/assets/week_02."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import (ACCENT, INK, MUTED, PRIMARY, ROLE_COLORS, TEAL, TEXT, TINT, arrow, box, role_box,
                    role_legend, routed_arrow, save as save_raster, save_editable_scene)

BLUE, GREEN, LOSS = ROLE_COLORS["data"], ROLE_COLORS["model"], ROLE_COLORS["loss"]


def save(fig, path, transparent=False):
    # Block schematics become editable PowerPoint objects; charts and images stay raster.
    if not transparent and all(not ax.axison and not ax.images for ax in fig.axes):
        save_editable_scene(fig, path)
    else:
        save_raster(fig, path, transparent=transparent)


def dashed_key(ax, x, y, label, size=18):
    """Legend entry for dashed backward (gradient) routes."""
    ax.plot([x, x + 0.5], [y + 0.11, y + 0.11], color=LOSS, lw=1.8, linestyle="--")
    ax.text(x + 0.62, y + 0.11, label, fontsize=size, color=TEXT, va="center")

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
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    entries = [(0.1, "Flatten\nx", "data"),
               (2.7, "Encode\nμ, log σ²", "model"),
               (5.3, "Sample\nz=μ+σε", "model"),
               (7.9, "Decode\nlogits", "model"),
               (10.5, "Reshape\n[B,28,28]", "output")]
    for left, text, role in entries:
        role_box(ax, left, 1.55, 2.0, 1.3, text, role, size=18)
    labels = ["x [B,784]", "μ, logvar [B,d]", "z [B,d]", "logits [B,784]"]
    for left, label in zip([2.1, 4.7, 7.3, 9.9], labels):
        arrow(ax, left, 2.2, left + 0.6, 2.2, color=GREEN)
        ax.text(left + 0.3, 1.2, label, ha="center", va="center", fontsize=18, color=BLUE)
    # The noise is an independent input: nothing flows into it from the encoder.
    role_box(ax, 5.3, 3.75, 2.0, 0.75, "Draw ε~N(0,I)", "data", size=18)
    arrow(ax, 6.3, 3.75, 6.3, 2.85, color=GREEN)
    ax.text(6.45, 3.3, "ε [B,d]", fontsize=18, ha="left", va="center", color=BLUE)
    ax.text(6.5, 0.72, "Loss: reconstruction + β × KL", fontsize=18,
            ha="center", va="center", color=TEXT)
    role_legend(ax, 0.08, size=18)
    save(fig, path)


def reparam(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 6)
        ax.set_ylim(0.3, 5.0)
    fig.subplots_adjust(left=.01, right=.99, bottom=.12, top=.9, wspace=.12)
    left, right = axes
    left.set_title("sample(): gradient stops", loc="left", fontsize=18, color=INK)
    right.set_title("Independent noise", loc="left", fontsize=18, color=INK)
    for ax in axes:
        role_box(ax, 0.1, 2.5, 1.5, 1.0, "Encoder\nμ, σ", "data", size=18)
        role_box(ax, 4.5, 2.5, 1.4, 1.0, "Loss", "loss", size=18)
        arrow(ax, 1.6, 3.0, 2.2, 3.0, color=GREEN)
        arrow(ax, 3.9, 3.0, 4.5, 3.0, color=GREEN)
        ax.text(1.9, 2.2, "μ, σ", ha="center", va="center", fontsize=18, color=BLUE)
        ax.text(4.2, 2.2, "z", ha="center", va="center", fontsize=18, color=BLUE)
    role_box(left, 2.2, 2.5, 1.7, 1.0, "sample()", "model", size=18)
    routed_arrow(left, [(5.2, 3.5), (5.2, 4.0), (3.05, 4.0), (3.05, 3.5)], color=LOSS, dashed=True)
    left.text(4.1, 4.25, "Stopped", ha="center", va="center", fontsize=18, color=LOSS)
    role_box(right, 2.2, 2.5, 1.7, 1.0, "Shift/scale\nμ+σε", "model", size=18)
    role_box(right, 2.2, 0.75, 1.7, 0.75, "Draw ε", "data", size=18)
    arrow(right, 3.05, 1.5, 3.05, 2.5, color=GREEN)
    right.text(3.25, 2.0, "ε", va="center", fontsize=18, color=BLUE)
    routed_arrow(right, [(5.2, 3.5), (5.2, 4.0), (0.85, 4.0), (0.85, 3.5)], color=LOSS, dashed=True)
    right.text(3.0, 4.25, "Gradients: μ,σ", ha="center", va="center", fontsize=18, color=LOSS)
    legend = fig.add_axes([.01, .0, .98, .09])
    legend.set_xlim(0, 13)
    legend.set_ylim(0, .4)
    legend.axis("off")
    role_legend(legend, .06, size=18)
    dashed_key(legend, 8.0, .06, "Gradient (backward)")
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
    fig, ax = plt.subplots(figsize=(13, 3.3))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.3)
    entries = [(0.2, "Flatten\n[B,1,28,28]", "data"),
               (3.4, "Encode\nshort code", "model"),
               (6.6, "Decode\nlogits", "model"),
               (9.8, "Reshape\n[B,28,28]", "output")]
    for left, text, role in entries:
        role_box(ax, left, 1.2, 2.2, 1.2, text, role, size=18)
    for left, label in [(2.4, "pixels [B,784]"), (5.6, "code [B,d]"), (8.8, "logits [B,784]")]:
        arrow(ax, left, 1.8, left + 1.0, 1.8, color=GREEN)
        ax.text(left + 0.5, 2.75, label, ha="center", va="center", fontsize=18, color=BLUE)
    ax.text(6.4, 0.78, "d = 2 example; the loss compares the output with the original pixels", ha="center",
            va="center", fontsize=18, color=TEXT)
    role_legend(ax, 0.08, size=18)
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
