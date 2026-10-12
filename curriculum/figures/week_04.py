"""Week 4 figures: diffusion diagrams (matplotlib) plus REAL results from curriculum/assets/week_04."""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, TEXT, TINT, arrow, box, save

# Full-width lecture figures use at least 18pt source labels.
_LECTURE_LABELS = {"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 20,
                   "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}

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


def forward_mnist_slide(path):
    """Enlarge the headings, keeping all 21 saved noisy-digit images."""
    im = plt.imread(ASSETS / "forward_mnist.png")
    fig, axes = plt.subplots(3, 7, figsize=(12, 5.2), gridspec_kw={"hspace": .08, "wspace": .15})
    for row, y in enumerate([115, 408, 702]):
        for ax, x, t, signal in zip(axes[row], [30, 302, 574, 847, 1119, 1391, 1663],
                                    [0, 50, 100, 250, 500, 750, 999],
                                    [1.000, .970, .895, .521, .078, .003, .000]):
            ax.imshow(im[y:y + 227, x:x + 227])
            if row == 0:
                ax.set_title(f"t = {t}\nᾱ = {signal:.3f}", fontsize=20)
            ax.axis("off")
    save(fig, path)


def reverse_2d_slide(path):
    """Shorter, larger stage labels over the unchanged saved scatter panels."""
    im = plt.imread(ASSETS / "reverse_2d.png")
    fig, ax = plt.subplots(figsize=(14, 2.9))
    ax.imshow(im, extent=(0, 2230, 411, 0))
    ax.add_patch(plt.Rectangle((0, 0), 2230, 69, facecolor="white", edgecolor="none"))
    for x, t in zip([183, 555, 927, 1299, 1671, 2043], [199, 150, 100, 50, 20, 0]):
        ax.text(x, 10, f"t = {t}", fontsize=22, color=INK, fontweight="bold", ha="center", va="top")
    ax.axis("off")
    save(fig, path)


def reverse_mnist_slide(path):
    """Replot the saved NPZ samples, preserving all seven stages and images."""
    saved = np.load(ASSETS / "reverse_mnist.npz")
    order = sorted((int(k) for k in saved.files), reverse=True)
    fig, axes = plt.subplots(len(order), 1, figsize=(9.8, 7.55))
    for ax, t in zip(axes, order):
        ax.imshow(np.hstack(saved[str(t)][:, 0]), cmap="gray_r", vmin=-1, vmax=1)
        ax.text(-8, 14, f"t = {t}", ha="right", va="center", fontsize=30, color=INK)
        ax.axis("off")
    fig.subplots_adjust(left=.20, right=.99, top=.99, bottom=.01, hspace=.12)
    save(fig, path)


def schedules(path):
    T = 1000
    t = np.arange(T)
    beta = np.linspace(1e-4, 0.02, T)
    abar_lin = np.cumprod(1 - beta)
    s = 0.008
    f = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    abar_cos = f / f[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    ax[0].plot(t, abar_lin, color=PRIMARY, lw=2.5, label="linear β schedule (DDPM)")
    ax[0].plot(t, abar_cos, color=ACCENT, lw=2.5, label="cosine schedule (improved DDPM)")
    ax[0].set_title(r"$\bar{\alpha}_t$: fraction of signal variance kept", loc="left")
    ax[0].set_xlabel("timestep t")
    snr_lin = abar_lin / (1 - abar_lin)
    snr_cos = abar_cos / (1 - abar_cos + 1e-12)
    ax[1].semilogy(t, snr_lin, color=PRIMARY, lw=2.5)
    ax[1].semilogy(t[1:-1], snr_cos[1:-1], color=ACCENT, lw=2.5)
    ax[1].set_title("SNR: unit-variance clean data", loc="left")
    ax[1].set_xlabel("timestep t")
    # One shared legend below both panels (under the axis labels), so no curve runs through it.
    fig.canvas.draw()
    bottom = min(a.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for a in ax)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, bottom - 0.02))
    save(fig, path)


def trilemma(path):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off")
    top, left, right = (6.5, 4.6), (2.6, 1.4), (10.4, 1.4)
    tri = np.array([top, left, right, top])
    ax.plot(tri[:, 0], tri[:, 1], color=MUTED, lw=2)
    ax.text(top[0], top[1] + 0.2, "High quality", ha="center", va="bottom", fontsize=18, fontweight="bold", color=INK)
    ax.text(left[0], left[1] - 0.2, "Diversity / coverage", ha="center", va="top", fontsize=18, fontweight="bold", color=INK)
    ax.text(right[0], right[1] - 0.2, "Fast sampling", ha="center", va="top", fontsize=18, fontweight="bold", color=INK)
    # Each family sits beside the edge joining the two strengths it usually has.
    for name, note, x, y, c in [("Diffusion", "many network calls", 0.35, 3.1, TEAL),
                                ("GAN", "fast; collapse risk", 10.65, 3.1, ACCENT),
                                ("VAE", "fast; may blur", 5.5, 0.15, PRIMARY)]:
        box(ax, x, y, 2.0, 0.65, name, fc="white", ec=c, color=c, size=20, lw=2)
        ax.text(x + 1.0, y - 0.12, note, ha="center", va="top", fontsize=18, color=TEXT)
    ax.text(6.5, 2.75, "Distillation (e.g. SD-Turbo)\ncuts diffusion's network calls", ha="center", va="center",
            fontsize=18, color=TEAL)
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.75, 5.3)
    save(fig, path)


def unet_dit(path):
    """Six U-Net operations and three DiT operations; solid skips carry data."""
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.axis("off")
    ax.set_xlim(-.1, 13.2); ax.set_ylim(-.7, 7.1)
    ax.text(.1, 6.85, "U-Net: multiscale features", fontsize=20, color=INK, weight="bold")
    nodes = [(.1, "Encode\n28 × 28"), (2.25, "Shrink\n14 × 14"),
             (4.4, "Bottleneck\n7 × 7"), (6.55, "Expand + join\n14 × 14"),
             (8.7, "Expand + join\n28 × 28"), (10.85, "Predict noise\n28 × 28")]
    for x, label in nodes:
        box(ax, x, 4.95, 1.9, 1.2, label, fc=TINT, ec=PRIMARY, size=18)
    for (x, _), (nx, _) in zip(nodes, nodes[1:]):
        arrow(ax, x + 1.9, 5.55, nx, 5.55)
        ax.text((x + 1.9 + nx) / 2, 6.32, "features", ha="center", fontsize=18, color=TEXT)
    # Nested orthogonal routes preserve the saved features at the two matching resolutions.
    for source, target, lane, label in ((1.05, 9.65, 3.75, "saved 28 × 28 features"),
                                       (3.2, 7.5, 4.35, "saved 14 × 14 features")):
        ax.plot([source, source, target], [4.95, lane, lane], color=TEAL, lw=2)
        arrow(ax, target, lane, target, 4.95, color=TEAL, lw=2)
        ax.text((source + target) / 2, lane - .28, label, fontsize=18, color=TEAL, ha="center")
    ax.text(.1, 2.75, "DiT: latent patches as tokens", fontsize=20, color=INK, weight="bold")
    for x, label in ((.1, "Embed patches"), (4.4, "Transformer blocks"), (8.7, "Project + reshape")):
        box(ax, x, .75, 3.1, 1.2, label, fc=TINT, ec=PRIMARY, size=18)
    arrow(ax, 3.2, 1.35, 4.4, 1.35); ax.text(3.8, 1.75, "tokens", fontsize=18, ha="center", color=TEXT)
    arrow(ax, 7.5, 1.35, 8.7, 1.35); ax.text(8.1, 1.75, "features", fontsize=18, ha="center", color=TEXT)
    ax.text(5.95, 2.25, "time + condition", fontsize=18, color=ACCENT, ha="center")
    arrow(ax, 5.95, 2.15, 5.95, 1.95, color=ACCENT)
    ax.text(.1, -.05, "Blue: learned operations. Solid arrows: feature data. Both outputs match the noisy input shape.",
            fontsize=18, color=TEXT)
    save(fig, path)


def ldm_pipeline(path):
    """Historical SD 1.5 inference: separate learned prediction and numerical update."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off")
    ax.set_xlim(-.1, 13.3); ax.set_ylim(-1.0, 4.6)
    box(ax, .1, 1.2, 2.1, 1.2, "Draw noise\n[1, 4, 64, 64]", fc="white", ec=MUTED, size=18)
    box(ax, 3.0, 1.2, 2.1, 1.2, "Predict noise\nU-Net", fc=TINT, ec=PRIMARY, size=18)
    box(ax, 5.9, 1.2, 2.1, 1.2, "Update latent\nScheduler", fc=ORANGE, ec=ACCENT, size=18)
    box(ax, 8.8, 1.2, 1.8, 1.2, "Decode\nVAE", fc=TINT, ec=PRIMARY, size=18)
    box(ax, 11.3, 1.2, 1.8, 1.2, "View image\n512 × 512", fc="white", ec=TEAL, size=18)
    box(ax, 3.0, 3.25, 2.1, .9, "Encode text", fc=TINT, ec=PRIMARY, size=18)
    ax.text(.1, 3.7, "Prompt tokens", fontsize=18, color=TEXT)
    arrow(ax, 2.2, 3.7, 3.0, 3.7)
    arrow(ax, 4.05, 3.25, 4.05, 2.4, color=PRIMARY)
    ax.text(4.2, 2.95, "text vectors", fontsize=18, color=PRIMARY)
    for begin, finish, label in ((2.2, 3.0, "latent"), (5.1, 5.9, "noise"),
                                 (8.0, 8.8, "final latent"), (10.6, 11.3, "pixels")):
        arrow(ax, begin, 1.8, finish, 1.8)
        ax.text((begin + finish) / 2, 2.6, label, fontsize=18, ha="center", color=TEXT)
    ax.plot([6.95, 6.95, 4.05], [1.2, .4, .4], ls="--", lw=1.8, color=MUTED)
    ax.annotate("", xy=(4.05, 1.2), xytext=(4.05, .4),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8, ls="--"))
    ax.text(5.5, .05, "updated latent + next timestep: repeat", ha="center", fontsize=18, color=TEXT)
    ax.text(.1, -.55, "Grey: data. Blue: learned. Orange: update rule. Teal: output.", fontsize=18, color=TEXT)
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
        ax.text(p[0] + 0.12, p[1] - 0.05, f"w = {w}" + ("  (conditional)" if w == 1 else "  (stronger guidance)" if w > 5 else "  (illustrative)"),
                color=col, fontsize=18, va="center")
    ax.annotate("", xy=u, xytext=o, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.2))
    ax.text(u[0] + 0.1, u[1] - 0.3, r"unconditional $\epsilon(x_t, \emptyset)$", color=MUTED, fontsize=18)
    ax.text(-0.25, 4.05, "Guided prediction = unconditional + w · (conditional − unconditional)", fontsize=18, color=INK, fontweight="bold")
    save(fig, path)


def flow_paths(path):
    rng = np.random.default_rng(3)
    data = np.c_[rng.choice([-2.0, 2.0], 12) + rng.normal(0, 0.2, 12), rng.choice([-1.2, 1.2], 12) + rng.normal(0, 0.2, 12)]
    noise = rng.normal(0, 1, (12, 2))
    t = np.linspace(0, 1, 50)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for a, title, kind in ((axes[0], "Schematic VP training pairs: curved", "vp"), (axes[1], "Linear training pairs: straight", "rf")):
        for x0, z in zip(data, noise):
            if kind == "vp":
                ang = t * np.pi / 2
                pts = np.cos(ang) * x0 + np.sin(ang) * z
            else:
                pts = (1 - t) * x0 + t * z
            a.plot(pts[:, 0], pts[:, 1], color=PRIMARY if kind == "vp" else TEAL, lw=1.6, alpha=0.8)
        a.scatter(noise[:, 0], noise[:, 1], color=MUTED, s=30, label="noise: t = 1")
        a.scatter(data[:, 0], data[:, 1], color=ACCENT, s=40, label="data: t = 0")
        a.set_title(title, loc="left")
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect("equal")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=MUTED, label="Noise: t = 1"),
                        Line2D([], [], marker="o", linestyle="", color=ACCENT, label="Data: t = 0")],
               ncol=2, loc="lower center", bbox_to_anchor=(.5, -.03), fontsize=18)
    fig.text(0.5, -.08, "Training pairs: data at 0, noise at 1. Learned sampling paths can curve.", ha="center", fontsize=18, color=TEXT)
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
        ax.set_title(t, loc="left", fontsize=18)
    save(fig, path)


def seeds_negative_slide(path):
    """All six archived outputs with fresh legible labels outside the images."""
    a = plt.imread(ASSETS / "sd_seeds.png")
    b = plt.imread(ASSETS / "sd_negative.png")
    # These bounds are the complete photograph panels; only old labels/white
    # margins are omitted. The full original montage remains in Word notes.
    panels = [a[74:478, x:x + 404] for x in [30, 516, 1001, 1486]]
    panels += [b[68:581, x:x + 513] for x in [30, 642]]
    titles = ["Seed 1", "Seed 2", "Seed 3", "Seed 4", "No negative\nprompt", "Negative\nprompt given"]
    fig, axes = plt.subplots(1, 6, figsize=(13, 3.2), gridspec_kw={"wspace": .12})
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=20)
        ax.axis("off")
    fig.text(.5, .02, "Negative prompt: lamp, blurry, low quality (same seed for the last two images)",
             ha="center", fontsize=18, color=TEXT)
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
    "seeds_negative_slide": seeds_negative_slide,
    "forward_mnist_slide": forward_mnist_slide,
    "reverse_2d_slide": reverse_2d_slide,
    "reverse_mnist_slide": reverse_mnist_slide,
    **{n: asset(n) for n in ["forward_mnist", "reverse_2d", "reverse_mnist", "mnist_samples", "sd_guidance",
                             "sd_steps", "sd_seeds", "sd_negative", "sd_turbo"]},
}


def _with_lecture_labels(make):
    def draw(path):
        with plt.rc_context(_LECTURE_LABELS):
            make(path)
    return draw


FIGURES = {name: _with_lecture_labels(make) for name, make in FIGURES.items()}
