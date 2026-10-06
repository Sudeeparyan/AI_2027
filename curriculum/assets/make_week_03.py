"""Train a DCGAN on MNIST and small GANs on 2-D toy data; save REAL result images for week 3.

Run once with the lab environment (CPU ~10 minutes):
    .venv-labs/Scripts/python.exe curriculum/assets/make_week_03.py
Outputs: curriculum/assets/week_03/*.png and metrics.json
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, save  # noqa: E402

OUT = Path(__file__).resolve().parent / "week_03"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parents[2] / "build" / "data"
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))
metrics = {}

# ------------------------------------------------------------------ DCGAN on MNIST
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # [-1, 1]
train = datasets.MNIST(DATA, train=True, download=True, transform=tf)
dl = DataLoader(train, batch_size=128, shuffle=True, drop_last=True)
ZD = 64


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ZD, 128, 7, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),   # 7x7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),    # 14x14
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh())                              # 28x28

    def forward(self, z):
        return self.net(z.view(-1, ZD, 1, 1))


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),                                     # 14x14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),   # 7x7
            nn.Conv2d(128, 1, 7, 1, 0))                                                             # logit

    def forward(self, x):
        return self.net(x).view(-1)


def init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


g, d = G(), D()
g.apply(init)
d.apply(init)
opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
z_fixed = torch.randn(32, ZD)
EPOCHS = 8
snap_epochs = [0, 1, 2, 4, 8]
snaps = {0: None}
lg, ld, dx, dgz = [], [], [], []
step = 0
with torch.no_grad():
    snaps[0] = g(z_fixed)
for ep in range(1, EPOCHS + 1):
    for x, _ in dl:
        b = x.size(0)
        ones, zeros = torch.ones(b), torch.zeros(b)
        # --- discriminator: maximise log D(x) + log(1 - D(G(z)))
        fake = g(torch.randn(b, ZD))
        out_real, out_fake = d(x), d(fake.detach())
        loss_d = bce(out_real, ones) + bce(out_fake, zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        # --- generator: non-saturating loss, maximise log D(G(z))
        out = d(fake)
        loss_g = bce(out, ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if step % 20 == 0:
            lg.append(loss_g.item())
            ld.append(loss_d.item())
            dx.append(torch.sigmoid(out_real).mean().item())
            dgz.append(torch.sigmoid(out_fake).mean().item())
        step += 1
    with torch.no_grad():
        g.eval()
        snaps[ep] = g(z_fixed)
        g.train()
    print(f"epoch {ep}: loss_D {loss_d.item():.3f} loss_G {loss_g.item():.3f}")


def grid(t, n=16):
    return np.hstack((t[:n, 0].numpy() + 1) / 2)


fig, axes = plt.subplots(len(snap_epochs), 1, figsize=(11, 1.15 * len(snap_epochs) + 0.3))
for a, e in zip(axes, snap_epochs):
    a.imshow(grid(snaps[e]), cmap="gray_r", vmin=0, vmax=1)
    a.set_title("before training" if e == 0 else f"after {e} epoch{'s' if e > 1 else ''}", loc="left", fontsize=12)
    a.axis("off")
save(fig, OUT / "epochs.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
steps = np.arange(len(lg)) * 20
ax[0].plot(steps, ld, color=PRIMARY, lw=1, label="discriminator loss")
ax[0].plot(steps, lg, color=ACCENT, lw=1, label="generator loss")
ax[0].set_xlabel("training step")
ax[0].set_title("Losses do not simply go down", loc="left")
ax[0].legend()
ax[1].plot(steps, dx, color=TEAL, lw=1, label="D(real)")
ax[1].plot(steps, dgz, color=ACCENT, lw=1, label="D(fake)")
ax[1].axhline(0.5, color=MUTED, ls="--", lw=1)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel("training step")
ax[1].set_title("Discriminator's average output", loc="left")
ax[1].legend()
save(fig, OUT / "losses.png")

# interpolation
g.eval()
with torch.no_grad():
    za, zb = torch.randn(ZD, generator=torch.Generator().manual_seed(3)), torch.randn(ZD, generator=torch.Generator().manual_seed(8))
    zc, zd_ = torch.randn(ZD, generator=torch.Generator().manual_seed(11)), torch.randn(ZD, generator=torch.Generator().manual_seed(21))
    ts = torch.linspace(0, 1, 10).unsqueeze(1)
    r1 = g((1 - ts) * za + ts * zb)
    r2 = g((1 - ts) * zc + ts * zd_)
fig, axes = plt.subplots(2, 1, figsize=(11, 2.6))
for a, r in zip(axes, (r1, r2)):
    a.imshow(grid(r, 10), cmap="gray_r", vmin=0, vmax=1)
    a.axis("off")
axes[0].set_title("Walking in a straight line between two random latent vectors", loc="left", fontsize=12)
save(fig, OUT / "interpolation.png")

# final sample sheet
with torch.no_grad():
    s = g(torch.randn(64, ZD, generator=torch.Generator().manual_seed(5)))
sheet = np.vstack([np.hstack((s[i * 8:(i + 1) * 8, 0].numpy() + 1) / 2) for i in range(8)])
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(sheet, cmap="gray_r", vmin=0, vmax=1)
ax.axis("off")
save(fig, OUT / "samples.png")
metrics["dcgan"] = {"epochs": EPOCHS, "final_loss_D": ld[-1], "final_loss_G": lg[-1], "final_D_real": dx[-1], "final_D_fake": dgz[-1]}

# ------------------------------------------------------------------ 2-D ring: mode collapse vs coverage
def ring(n, k=8, r=2.0, s=0.05):
    c = np.random.randint(0, k, n)
    ang = 2 * np.pi * c / k
    return torch.tensor(np.c_[r * np.cos(ang), r * np.sin(ang)] + s * np.random.randn(n, 2), dtype=torch.float32)


def mlp(i, o, h=128, out_act=None):
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)]
    return nn.Sequential(*layers)


def train_2d(steps, d_steps=1, lr_g=1e-3, lr_d=1e-3, wgan_gp=False, seed=0, snaps=(0, 500, 2000, 5000)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    G2, D2 = mlp(2, 2), mlp(2, 1)
    og = torch.optim.Adam(G2.parameters(), lr=lr_g, betas=(0.5, 0.9))
    od = torch.optim.Adam(D2.parameters(), lr=lr_d, betas=(0.5, 0.9))
    zf = torch.randn(1500, 2)
    out = {}
    for t in range(steps + 1):
        if t in snaps:
            with torch.no_grad():
                out[t] = G2(zf).numpy()
        for _ in range(d_steps):
            x = ring(256)
            f = G2(torch.randn(256, 2)).detach()
            if wgan_gp:
                eps = torch.rand(256, 1)
                xi = (eps * x + (1 - eps) * f).requires_grad_(True)
                gr = torch.autograd.grad(D2(xi).sum(), xi, create_graph=True)[0]
                ldd = D2(f).mean() - D2(x).mean() + 10 * ((gr.norm(dim=1) - 1) ** 2).mean()
            else:
                ldd = bce(D2(x).view(-1), torch.ones(256)) + bce(D2(f).view(-1), torch.zeros(256))
            od.zero_grad()
            ldd.backward()
            od.step()
        f = G2(torch.randn(256, 2))
        lgg = -D2(f).mean() if wgan_gp else bce(D2(f).view(-1), torch.ones(256))
        og.zero_grad()
        lgg.backward()
        og.step()
    return out


def modes_covered(pts, k=8, r=2.0, tol=0.3):
    centres = np.c_[r * np.cos(2 * np.pi * np.arange(k) / k), r * np.sin(2 * np.pi * np.arange(k) / k)]
    dist = np.linalg.norm(pts[:, None, :] - centres[None], axis=2)
    near = dist.min(1) < tol
    counts = np.bincount(dist.argmin(1)[near], minlength=k)
    return int((counts > 0.02 * len(pts)).sum()), float(near.mean())


snaps2 = (0, 500, 2000, 5000)
vanilla = train_2d(5000, seed=1, snaps=snaps2)
wgan = train_2d(5000, d_steps=5, lr_g=1e-4, lr_d=1e-4, wgan_gp=True, seed=1, snaps=snaps2)
real = ring(1500).numpy()
fig, axes = plt.subplots(2, len(snaps2), figsize=(13, 6.4))
for row, (name, res) in enumerate((("Standard GAN", vanilla), ("WGAN-GP", wgan))):
    for col, t in enumerate(snaps2):
        a = axes[row, col]
        a.scatter(real[:, 0], real[:, 1], s=2, color=MUTED, alpha=0.3)
        a.scatter(res[t][:, 0], res[t][:, 1], s=2, color=PRIMARY if row == 0 else TEAL, alpha=0.5)
        m, q = modes_covered(res[t])
        a.set_title(f"{name}, step {t}\nmodes covered: {m}/8", fontsize=11.5, loc="left")
        a.set_xlim(-3, 3)
        a.set_ylim(-3, 3)
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect("equal")
save(fig, OUT / "mode_collapse.png")
metrics["ring"] = {"vanilla_final_modes": modes_covered(vanilla[5000])[0], "wgan_final_modes": modes_covered(wgan[5000])[0]}

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
