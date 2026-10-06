"""Train small VAEs / an autoencoder on MNIST and save REAL result images for week 2 slides and notes.

Run once with the lab environment (CPU is fine, ~5 minutes):
    .venv-labs/Scripts/python.exe curriculum/assets/make_week_02.py
Outputs: curriculum/assets/week_02/*.png and metrics.json (checked into the project as source assets).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, save  # noqa: E402

OUT = Path(__file__).resolve().parent / "week_02"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parents[2] / "build" / "data"
torch.manual_seed(0)
np.random.seed(0)

tf = transforms.ToTensor()
train = datasets.MNIST(DATA, train=True, download=True, transform=tf)
test = datasets.MNIST(DATA, train=False, download=True, transform=tf)
fashion = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
train_dl = DataLoader(train, batch_size=128, shuffle=True)


class VAE(nn.Module):
    def __init__(self, zdim=2, h=400, deterministic=False):
        super().__init__()
        self.det = deterministic
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, h), nn.ReLU())
        self.mu = nn.Linear(h, zdim)
        self.logvar = nn.Linear(h, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, h), nn.ReLU(), nn.Linear(h, 784))

    def encode(self, x):
        hdn = self.enc(x)
        return self.mu(hdn), self.logvar(hdn)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu if self.det else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z), mu, logvar


def loss_fn(logits, x, mu, logvar, beta=1.0, det=False):
    rec = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum") / x.size(0)
    kl = torch.zeros(()) if det else (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
    return rec + beta * kl, rec, kl


def fit(model, epochs, beta=1.0, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        tot = np.zeros(3)
        n = 0
        for x, _ in train_dl:
            logits, mu, logvar = model(x)
            loss, rec, kl = loss_fn(logits, x, mu, logvar, beta, model.det)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += [loss.item(), rec.item(), kl.item()]
            n += 1
        if log is not None:
            log.append((tot / n).tolist())
        print(f"  epoch {ep + 1}: loss {tot[0] / n:.1f}  rec {tot[1] / n:.1f}  kl {tot[2] / n:.2f}")
    return model


def decode(model, z):
    with torch.no_grad():
        return torch.sigmoid(model.dec(torch.as_tensor(z, dtype=torch.float32))).view(-1, 28, 28).numpy()


metrics = {}
EPOCHS = 10

print("VAE (z=2)")
hist = []
vae2 = fit(VAE(2), EPOCHS, log=hist)
print("Autoencoder (z=2)")
ae2 = fit(VAE(2, deterministic=True), EPOCHS)
print("VAE (z=16)")
vae16 = fit(VAE(16), EPOCHS)

# --- training curves
h = np.array(hist)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, EPOCHS + 1), h[:, 1], marker="o", color=PRIMARY)
ax[0].set_title("Reconstruction loss (BCE per image)", loc="left")
ax[1].plot(range(1, EPOCHS + 1), h[:, 2], marker="o", color=ACCENT)
ax[1].set_title("KL divergence per image (nats)", loc="left")
for a in ax:
    a.set_xlabel("epoch")
save(fig, OUT / "training_curves.png")
metrics["vae2_final"] = {"loss": h[-1, 0], "rec": h[-1, 1], "kl": h[-1, 2]}

# --- latent scatter: AE vs VAE
xs, ys = next(iter(DataLoader(Subset(test, range(5000)), batch_size=5000)))
with torch.no_grad():
    z_ae, _ = ae2.encode(xs)
    z_vae, _ = vae2.encode(xs)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, z, t in ((ax[0], z_ae, "Plain autoencoder: codes are points,\nscattered with gaps"), (ax[1], z_vae, "VAE: codes pulled towards N(0, I),\none continuous space")):
    sc = a.scatter(z[:, 0], z[:, 1], c=ys, cmap="tab10", s=4)
    a.set_title(t, loc="left")
    a.set_xlabel("z₁")
    a.set_ylabel("z₂")
cb = fig.colorbar(sc, ax=ax, ticks=range(10), shrink=0.85)
cb.set_label("digit")
save(fig, OUT / "latent_ae_vs_vae.png")

# --- random samples from the prior: AE vs VAE
rng = np.random.default_rng(1)
zs = rng.normal(size=(32, 2))
fig, ax = plt.subplots(2, 1, figsize=(11, 3.6))
for a, m, t in ((ax[0], ae2, "Autoencoder decoder fed random z ~ N(0, I)"), (ax[1], vae2, "VAE decoder fed random z ~ N(0, I)")):
    imgs = decode(m, zs)
    a.imshow(np.hstack(imgs[:16]), cmap="gray_r")
    a.set_title(t, loc="left", fontsize=13)
    a.axis("off")
save(fig, OUT / "samples_ae_vs_vae.png")

# --- reconstructions (z=16)
with torch.no_grad():
    logits, _, _ = vae16(xs[:10])
rec = torch.sigmoid(logits).view(-1, 28, 28).numpy()
fig, ax = plt.subplots(2, 1, figsize=(11, 2.8))
ax[0].imshow(np.hstack(xs[:10, 0].numpy()), cmap="gray_r")
ax[0].set_title("Test images", loc="left", fontsize=13)
ax[1].imshow(np.hstack(rec), cmap="gray_r")
ax[1].set_title("VAE reconstructions (16-dimensional latent)", loc="left", fontsize=13)
for a in ax:
    a.axis("off")
save(fig, OUT / "reconstructions.png")

# --- 2-D manifold
from scipy.stats import norm  # noqa: E402

n = 15
grid = norm.ppf(np.linspace(0.03, 0.97, n))
canvas = np.zeros((28 * n, 28 * n))
for i, yi in enumerate(grid[::-1]):
    for j, xi in enumerate(grid):
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = decode(vae2, [[xi, yi]])[0]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.imshow(canvas, cmap="gray_r")
ax.set_title("Decoding a grid of latent points (z = 2)", loc="left")
ax.axis("off")
save(fig, OUT / "manifold.png")

# --- interpolation between a 1 and a 7 (z=16)
with torch.no_grad():
    mu, _ = vae16.encode(xs)
i1 = int((ys == 1).nonzero()[0])
i7 = int((ys == 7).nonzero()[0])
i3 = int((ys == 3).nonzero()[0])
i8 = int((ys == 8).nonzero()[0])
fig, ax = plt.subplots(2, 1, figsize=(11, 2.8))
for a, (p, q) in zip(ax, ((i1, i7), (i3, i8))):
    ts = np.linspace(0, 1, 10)
    zz = np.stack([(1 - t) * mu[p].numpy() + t * mu[q].numpy() for t in ts])
    a.imshow(np.hstack(decode(vae16, zz)), cmap="gray_r")
    a.axis("off")
ax[0].set_title("Straight-line interpolation in latent space: 1 → 7 and 3 → 8", loc="left", fontsize=13)
save(fig, OUT / "interpolation.png")

# --- beta comparison (z=16)
betas = [0.1, 1.0, 4.0]
rows = []
for b in betas:
    print("beta", b)
    lg = []
    m = fit(VAE(16), 5, beta=b, log=lg)
    zs16 = np.random.default_rng(2).normal(size=(12, 16))
    rows.append((b, decode(m, zs16), lg[-1]))
fig, ax = plt.subplots(len(betas), 1, figsize=(11, 4.2))
for a, (b, imgs, lg) in zip(ax, rows):
    a.imshow(np.hstack(imgs), cmap="gray_r")
    a.set_title(f"β = {b}:  reconstruction {lg[1]:.0f}, KL {lg[2]:.1f} nats", loc="left", fontsize=12.5)
    a.axis("off")
save(fig, OUT / "beta_samples.png")
metrics["beta"] = {str(b): {"rec": lg[1], "kl": lg[2]} for b, _, lg in rows}

# --- anomaly detection: reconstruction error MNIST vs Fashion-MNIST (z=16)
def rec_err(ds, k=2000):
    x, _ = next(iter(DataLoader(Subset(ds, range(k)), batch_size=k)))
    with torch.no_grad():
        lg, _, _ = vae16(x)
        return F.binary_cross_entropy_with_logits(lg, x.view(-1, 784), reduction="none").sum(1).numpy()


e_in, e_out = rec_err(test), rec_err(fashion)
from sklearn.metrics import roc_auc_score  # noqa: E402

auc = roc_auc_score(np.r_[np.zeros_like(e_in), np.ones_like(e_out)], np.r_[e_in, e_out])
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.linspace(0, max(e_in.max(), np.percentile(e_out, 99)), 60)
ax.hist(e_in, bins=bins, alpha=0.7, color=PRIMARY, label="MNIST digits (seen type)")
ax.hist(e_out, bins=bins, alpha=0.7, color=ACCENT, label="Fashion-MNIST items (unseen type)")
ax.set_xlabel("reconstruction error (BCE per image)")
ax.set_ylabel("count")
ax.set_title(f"Anomaly detection by reconstruction error (ROC AUC = {auc:.3f})", loc="left")
ax.legend()
save(fig, OUT / "anomaly.png")
metrics["anomaly_auc"] = auc

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
