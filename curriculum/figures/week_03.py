"""Week 3 figures: GAN diagrams (matplotlib) plus REAL training results from curriculum/assets/week_03."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, TEXT, TINT, arrow, box, save

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


def gan_arch(path):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    box(ax, 0.1, 0.5, 1.7, 1.1, "noise\nz ~ N(0, I)", fc=ORANGE, ec=ACCENT, size=13)
    box(ax, 2.4, 0.4, 2.2, 1.3, "Generator G", fc="white", size=16)
    box(ax, 5.2, 0.5, 1.8, 1.1, "fake image\nG(z)", size=13)
    box(ax, 5.2, 3.0, 1.8, 1.1, "real image\nx", size=13)
    box(ax, 7.8, 1.6, 2.3, 1.5, "Discriminator D\nP(real)", fc="white", size=15)
    box(ax, 10.8, 1.85, 2.0, 1.0, "real or fake?", fc=TINT, size=13)
    arrow(ax, 1.8, 1.05, 2.4, 1.05)
    arrow(ax, 4.6, 1.05, 5.2, 1.05)
    arrow(ax, 7.0, 1.05, 7.8, 2.0)
    arrow(ax, 7.0, 3.55, 7.8, 2.7)
    arrow(ax, 10.1, 2.35, 10.8, 2.35)
    ax.annotate("", xy=(3.5, 1.75), xytext=(8.9, 1.55), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2, ls="--", connectionstyle="arc3,rad=0.35"))
    ax.text(6.0, 0.05, "gradient from D tells G how to look more real", color=ACCENT, fontsize=12.5, ha="center")
    ax.text(0.1, 4.35, "D is trained to tell real from fake; G is trained to fool D.", fontsize=15, color=INK, fontweight="bold")
    save(fig, path)


def optimal_d(path):
    x = np.linspace(-4, 6, 500)
    pdata = 0.6 * np.exp(-(x - 0) ** 2 / 0.8) + 0.4 * np.exp(-(x - 3) ** 2 / 0.5)
    pdata /= np.trapezoid(pdata, x)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for a, (mu, s, title) in zip(axes, [(2.0, 1.6, "Early: generator far from the data"), (None, None, "Equilibrium: p_g = p_data")]):
        pg = pdata.copy() if mu is None else np.exp(-(x - mu) ** 2 / (2 * s ** 2))
        pg /= np.trapezoid(pg, x)
        dstar = pdata / (pdata + pg + 1e-12)
        a.fill_between(x, pdata, color=TEAL, alpha=0.25, label="p_data (real)")
        a.plot(x, pg, color=ACCENT, lw=2.2, label="p_g (generator)")
        a2 = a.twinx()
        a2.plot(x, dstar, color=PRIMARY, lw=2.5, ls="--", label="D*(x)")
        a2.set_ylim(0, 1.05)
        a2.set_ylabel("D*(x)", color=PRIMARY)
        a2.grid(False)
        a.set_title(title, loc="left")
        a.set_yticks([])
        a.legend(loc="upper left", fontsize=10.5)
        a2.legend(loc="upper right", fontsize=10.5)
    save(fig, path)


def saturating(path):
    d = np.linspace(0.005, 0.995, 300)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(d, np.log(1 - d), color=MUTED, lw=2.5, label="minimax: G minimises log(1 − D(G(z)))")
    ax.plot(d, -np.log(d), color=ACCENT, lw=2.5, label="non-saturating: G minimises −log D(G(z))")
    ax.axvspan(0, 0.1, color=ACCENT, alpha=0.08)
    ax.text(0.02, 3.6, "early training:\nD easily spots fakes", fontsize=11.5, color=TEXT)
    ax.set_xlabel("D(G(z)): discriminator's belief that a fake is real")
    ax.set_ylabel("generator loss")
    ax.set_ylim(-4.5, 5.5)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("Same fixed point, very different gradients when D wins", loc="left")
    save(fig, path)


def dcgan_arch(path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    specs = [(0.1, 0.3, "z\n64", ORANGE), (1.55, 1.4, "128 × 7 × 7", TINT), (3.85, 2.0, "64 × 14 × 14", TINT), (6.35, 2.8, "1 × 28 × 28\nimage", "white")]
    width = lambda h: 0.65 + h * 0.45  # noqa: E731
    for x, h, lab, fc in specs:
        box(ax, x, 2.0 - h / 2 - 0.2, width(h), h + 0.4, lab, fc=fc, size=13.5)
    for (x1, h1, _, _), (x2, _, _, _) in zip(specs, specs[1:]):
        arrow(ax, x1 + width(h1), 2.0, x2, 2.0)
    for x, t in ((1.15, "transposed conv\n+ BatchNorm + ReLU"), (3.3, "transposed conv\n+ BatchNorm + ReLU"), (5.75, "transposed conv\n+ tanh")):
        ax.text(x, 3.75, t, ha="center", fontsize=12.5, color=MUTED)
    ax.text(8.75, 3.6, "DCGAN guidelines\n(Radford et al., 2015)", fontsize=14, fontweight="bold", color=INK, va="center")
    for i, t in enumerate(["• strided / transposed\n   convolutions, no pooling",
                           "• BatchNorm in G and D\n   (not on G output or D input)",
                           "• ReLU in G, tanh output;\n   LeakyReLU in D",
                           "• Adam, lr 2·10⁻⁴, β₁ = 0.5"]):
        ax.text(8.75, 2.95 - i * 0.72, t, fontsize=12.5, color=TEXT, va="top")
    save(fig, path)


def cgan(path):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    box(ax, 0.1, 2.2, 1.6, 0.9, "noise z", fc=ORANGE, ec=ACCENT, size=13)
    box(ax, 0.1, 0.6, 1.6, 0.9, "label y = '7'", fc=TINT, size=13)
    box(ax, 2.5, 1.2, 2.0, 1.3, "G(z, y)", fc="white", size=16)
    box(ax, 5.3, 1.35, 1.6, 1.0, "fake 7", size=13)
    box(ax, 7.8, 1.2, 2.4, 1.3, "D(x, y)\nreal AND matches y?", fc="white", size=13)
    box(ax, 5.3, 0.05, 1.6, 0.8, "label y", fc=TINT, size=12)
    arrow(ax, 1.7, 2.65, 2.5, 2.1)
    arrow(ax, 1.7, 1.05, 2.5, 1.55)
    arrow(ax, 4.5, 1.85, 5.3, 1.85)
    arrow(ax, 6.9, 1.85, 7.8, 1.85)
    arrow(ax, 6.9, 0.45, 7.8, 1.3)
    ax.text(10.6, 2.3, "Condition can be a class,\na text prompt, a sketch,\nor another image\n(pix2pix, CycleGAN)", fontsize=12.5, color=TEXT, va="center")
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
        a.legend(loc="lower right", fontsize=10.5)
        a.set_xlim(-3.5, 3.5)
        a.set_ylim(-3, 3)
    save(fig, path)


FIGURES = {
    "cover": cover,
    "gan_arch": gan_arch,
    "optimal_d": optimal_d,
    "saturating": saturating,
    "dcgan_arch": dcgan_arch,
    "cgan": cgan,
    "fid": fid,
    **{n: asset(n) for n in ["epochs", "losses", "interpolation", "samples", "mode_collapse"]},
}
