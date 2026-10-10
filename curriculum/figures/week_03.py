"""Week 3 figures: GAN diagrams (matplotlib) plus REAL training results from curriculum/assets/week_03."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, TEXT, TINT, arrow, box, routed_arrow, save

# Full-width lecture figures use at least 18pt source labels.
_LECTURE_LABELS = {"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 20,
                   "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_03"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    img = plt.imread(ASSETS / "samples.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img
    ax.imshow(1 - gray, cmap="magma")
    ax.axis("off")
    save(fig, path, transparent=True)


def epochs_slide(path):
    """The five saved sample strips, labelled at the left so no heading covers an image."""
    im = plt.imread(ASSETS / "epochs.png")
    stages = [("Before training", "pure noise"), ("After 1 epoch", "rough strokes"),
              ("After 2 epochs", "first digit shapes"), ("After 4 epochs", "many digits readable"),
              ("After 8 epochs", "sharper; some malformed")]
    fig, axes = plt.subplots(len(stages), 1, figsize=(13, 3.9))
    fig.subplots_adjust(left=0.25, right=0.995, top=0.99, bottom=0.01, hspace=0.25)
    for k, (ax, (stage, note)) in enumerate(zip(axes, stages)):
        y = 71 + 193 * k  # each saved strip is 107 px tall and starts 193 px below the previous one
        ax.imshow(im[y:y + 107, 30:1735])
        ax.axis("off")
        ax.text(-0.02, 0.68, stage, transform=ax.transAxes, ha="right", va="center", fontsize=20,
                fontweight="bold", color=INK)
        ax.text(-0.02, 0.2, note, transform=ax.transAxes, ha="right", va="center", fontsize=18, color=MUTED)
    save(fig, path)


def mode_collapse_slide(path):
    """Keep every archived scatter panel; give enlarged headers their own space."""
    im = plt.imread(ASSETS / "mode_collapse.png")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), gridspec_kw={"hspace": .45, "wspace": .15})
    for row, (y, name, covered) in enumerate([(110, "GAN", [0, 8, 7, 7]), (650, "WGAN-GP", [0, 0, 6, 8])]):
        for ax, x, step, count in zip(axes[row], [30, 556, 1081, 1607], [0, 500, 2000, 5000], covered):
            ax.imshow(im[y:y + 439, x:x + 439])
            ax.set_title(f"{name}: {step}\n{count}/8 modes", fontsize=20, loc="left")
            ax.axis("off")
    save(fig, path)


def gan_arch(path):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.6, 4.8)
    box(ax, 0.1, 0.5, 1.7, 1.1, "noise\nz ~ N(0, I)", fc=ORANGE, ec=ACCENT, size=18)
    box(ax, 2.4, 0.4, 2.2, 1.3, "Generator G", fc="white", size=18)
    box(ax, 5.2, 0.5, 1.8, 1.1, "fake image\nG(z)", size=18)
    box(ax, 5.2, 3.0, 1.8, 1.1, "real image\nx", size=18)
    box(ax, 7.8, 1.6, 2.3, 1.5, "Discriminator D\nP(real)", fc="white", size=18)
    box(ax, 10.8, 1.85, 2.0, 1.0, "real or fake?", fc=TINT, size=18)
    arrow(ax, 1.8, 1.05, 2.4, 1.05)
    arrow(ax, 4.6, 1.05, 5.2, 1.05)
    arrow(ax, 7.0, 1.05, 7.8, 2.0)
    arrow(ax, 7.0, 3.55, 7.8, 2.7)
    arrow(ax, 10.1, 2.35, 10.8, 2.35)
    # Feedback runs under the boxes: from D back to G, crossing no box or arrow.
    routed_arrow(ax, [(8.95, 1.6), (8.95, 0.05), (3.5, 0.05), (3.5, 0.4)], color=ACCENT, dashed=True)
    ax.text(6.2, -0.35, "gradient from D tells G how to look more real", color=ACCENT, fontsize=18, ha="center", va="center")
    ax.text(0.1, 4.35, "D is trained to tell real from fake; G is trained to fool D.", fontsize=18, color=INK, fontweight="bold")
    save(fig, path)


def optimal_d(path):
    x = np.linspace(-4, 6, 500)
    pdata = 0.6 * np.exp(-(x - 0) ** 2 / 0.8) + 0.4 * np.exp(-(x - 3) ** 2 / 0.5)
    pdata /= np.trapezoid(pdata, x)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for a, (mu, s, title) in zip(axes, [(2.0, 1.6, "Early: generator far from the data"), (None, None, r"Equilibrium: $p_g = p_{\mathrm{data}}$")]):
        pg = pdata.copy() if mu is None else np.exp(-(x - mu) ** 2 / (2 * s ** 2))
        pg /= np.trapezoid(pg, x)
        dstar = pdata / (pdata + pg + 1e-12)
        a.fill_between(x, pdata, color=TEAL, alpha=0.25, label=r"$p_{\mathrm{data}}$ (real)")
        a.plot(x, pg, color=ACCENT, lw=2.2, label=r"$p_g$ (generator)")
        a2 = a.twinx()
        a2.plot(x, dstar, color=PRIMARY, lw=2.5, ls="--", label="D*(x)")
        a2.set_ylim(0, 1.05)
        a2.set_ylabel("D*(x)", color=PRIMARY)
        a2.grid(False)
        a.set_title(title, loc="left")
        a.set_yticks([])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=TEAL, lw=5, label=r"Real density $p_{\mathrm{data}}$"),
                        Line2D([], [], color=ACCENT, lw=2.5, label=r"Generator density $p_g$"),
                        Line2D([], [], color=PRIMARY, lw=2.5, ls="--", label="D*(x)")],
               ncol=3, loc="lower center", bbox_to_anchor=(.5, -.13), fontsize=18)
    save(fig, path)


def saturating(path):
    s = np.linspace(-8, 8, 300)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(s, -np.logaddexp(0, s), color=MUTED, lw=2.5, label="Minimax: log(1 − D)")
    ax.plot(s, np.logaddexp(0, -s), color=ACCENT, lw=2.5, label="Non-saturating: −log D")
    ax.axvspan(-8, -2, color=ACCENT, alpha=0.08)
    ax.text(-7.8, -1.0, "D rejects fakes:\nweak minimax slope", fontsize=18, color=TEXT, va="top")
    ax.set_xlabel("D logit s for a generated image; probability = sigmoid(s)")
    ax.set_ylabel("generator loss")
    ax.set_ylim(-8.5, 8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), fontsize=18, ncol=2)
    ax.set_title("The sigmoid path explains weak minimax gradients", loc="left")
    save(fig, path)


def dcgan_arch(path):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    specs = [(.1, 2.0, "noise z\n64 numbers", ORANGE), (3.0, 2.2, "128 × 7 × 7\nfeatures", TINT),
             (6.6, 2.3, "64 × 14 × 14\nfeatures", TINT), (10.3, 2.5, "1 × 28 × 28\nimage", "white")]
    for x, w, lab, fc in specs:
        box(ax, x, 1.0, w, 1.5, lab, fc=fc, size=18)
    for (x1, w, _, _), (x2, _, _, _) in zip(specs, specs[1:]):
        arrow(ax, x1 + w, 1.75, x2, 1.75)
    for x, t in ((4.1, "Transposed conv\nBatchNorm + ReLU"), (7.75, "Transposed conv\nBatchNorm + ReLU"), (11.55, "Transposed conv\ntanh output")):
        ax.text(x, 3.05, t, ha="center", fontsize=18, color=MUTED)
    ax.text(.1, .25, "Example DCGAN generator. Notes explain architecture and optimisation choices.", fontsize=18, color=TEXT)
    save(fig, path)


def cgan(path):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    box(ax, 0.1, 2.2, 1.6, 0.9, "noise z", fc=ORANGE, ec=ACCENT, size=18)
    box(ax, 0.1, 0.6, 1.6, 0.9, "label y = '7'", fc=TINT, size=18)
    box(ax, 2.5, 1.2, 2.0, 1.3, "G(z, y)", fc="white", size=18)
    box(ax, 5.3, 1.35, 1.6, 1.0, "fake 7", size=18)
    box(ax, 7.8, 1.2, 2.4, 1.3, "D(x, y)\nreal + matches\nthe condition?", fc="white", size=18)
    box(ax, 5.3, 0.05, 1.6, 0.8, "label y", fc=TINT, size=18)
    arrow(ax, 1.7, 2.65, 2.5, 2.1)
    arrow(ax, 1.7, 1.05, 2.5, 1.55)
    arrow(ax, 4.5, 1.85, 5.3, 1.85)
    arrow(ax, 6.9, 1.85, 7.8, 1.85)
    arrow(ax, 6.9, 0.45, 7.8, 1.3)
    ax.text(10.6, 2.3, "Condition can be a class,\na text prompt, a sketch,\nor another image\n(e.g. pix2pix)", fontsize=18, color=TEXT, va="center")
    save(fig, path)


def fid(path):
    rng = np.random.default_rng(4)
    real = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 0.8]], 400)
    good = rng.multivariate_normal([0.2, 0.1], [[0.95, 0.25], [0.25, 0.85]], 400)
    bad = rng.multivariate_normal([1.3, 0.9], [[0.25, 0], [0, 0.2]], 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, fake, t in ((axes[0], good, "Low FID: similar mean and spread"), (axes[1], bad, "High FID: shifted and too narrow\n(low diversity, e.g. mode collapse)")):
        a.scatter(real[:, 0], real[:, 1], s=8, color=TEAL, alpha=0.45, label="real images (features)")
        a.scatter(fake[:, 0], fake[:, 1], s=8, color=ACCENT, alpha=0.45, label="generated images (features)")
        a.set_title(t, loc="left")
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(-3.5, 3.5)
        a.set_ylim(-3, 3)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=TEAL, label="Real-image features"),
                        Line2D([], [], marker="o", linestyle="", color=ACCENT, label="Generated-image features")],
               loc="lower center", bbox_to_anchor=(.5, -.08), ncol=2, fontsize=18)
    save(fig, path)


FIGURES = {
    "cover": cover,
    "gan_arch": gan_arch,
    "optimal_d": optimal_d,
    "saturating": saturating,
    "dcgan_arch": dcgan_arch,
    "cgan": cgan,
    "fid": fid,
    "epochs_slide": epochs_slide,
    "mode_collapse_slide": mode_collapse_slide,
    **{n: asset(n) for n in ["epochs", "losses", "interpolation", "samples", "mode_collapse"]},
}


def _with_lecture_labels(make):
    def draw(path):
        with plt.rc_context(_LECTURE_LABELS):
            make(path)
    return draw


FIGURES = {name: _with_lecture_labels(make) for name, make in FIGURES.items()}
