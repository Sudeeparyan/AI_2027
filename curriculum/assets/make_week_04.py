"""REAL results for week 4 (diffusion models).

Part A (CPU is fine): forward noising of real digits, a DDPM trained on 2-D data, a tiny DDPM trained on MNIST.
Part B (GPU recommended; CPU works but is slow): Stable Diffusion v1.5 sweeps over guidance scale, steps,
seeds and negative prompts, plus SD-Turbo 1-step generation with timings.

Run:  .venv-gpu/Scripts/python.exe curriculum/assets/make_week_04.py [--part A|B|all]
Outputs: curriculum/assets/week_04/*.png and metrics.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, save  # noqa: E402

OUT = Path(__file__).resolve().parent / "week_04"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parents[2] / "build" / "data"
# ASSET_DEVICE=cpu forces CPU (GTX 16xx cards give black images in fp16 and cannot hold SD 1.5 in fp32).
DEV = __import__("os").environ.get("ASSET_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MPATH = OUT / "metrics.json"
metrics = json.loads(MPATH.read_text()) if MPATH.exists() else {}


def schedule(T=1000, b0=1e-4, b1=0.02):
    betas = torch.linspace(b0, b1, T)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, 0)
    return betas, alphas, abar


# ============================================================ PART A
def part_a():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    np.random.seed(0)
    betas, alphas, abar = schedule()

    # --- A1 forward noising of real digits
    mn = datasets.MNIST(DATA, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    x0 = torch.stack([mn[i][0] for i in (0, 1, 2)])
    ts = [0, 50, 100, 250, 500, 750, 999]
    fig, axes = plt.subplots(3, len(ts), figsize=(12, 5.4))
    for r in range(3):
        for c, t in enumerate(ts):
            eps = torch.randn_like(x0[r])
            xt = abar[t].sqrt() * x0[r] + (1 - abar[t]).sqrt() * eps
            axes[r, c].imshow(xt[0].numpy(), cmap="gray", vmin=-2.5, vmax=2.5)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"t = {t}\nᾱ = {abar[t]:.3f}", fontsize=12)
    save(fig, OUT / "forward_mnist.png")

    # --- A2 DDPM on 2-D swiss roll
    from sklearn.datasets import make_swiss_roll

    data = make_swiss_roll(20000, noise=0.4, random_state=0)[0][:, [0, 2]] / 10.0
    data = torch.tensor(data, dtype=torch.float32)
    T2 = 200
    b2, a2, ab2 = schedule(T2, 1e-4, 0.05)

    class Eps2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

        def forward(self, x, t):
            return self.net(torch.cat([x, (t.float() / T2).unsqueeze(1)], 1))

    m2 = Eps2D()
    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    losses = []
    for step in range(6000):
        x = data[torch.randint(0, len(data), (512,))]
        t = torch.randint(0, T2, (512,))
        eps = torch.randn_like(x)
        xt = ab2[t].sqrt().unsqueeze(1) * x + (1 - ab2[t]).sqrt().unsqueeze(1) * eps
        loss = F.mse_loss(m2(xt, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    snaps = {}
    with torch.no_grad():
        x = torch.randn(2000, 2)
        for t in reversed(range(T2)):
            tt = torch.full((2000,), t)
            e = m2(x, tt)
            mean = (x - b2[t] / (1 - ab2[t]).sqrt() * e) / a2[t].sqrt()
            x = mean + (b2[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
            if t in (199, 150, 100, 50, 20, 0):
                snaps[t] = x.clone()
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.9))
    for a, t in zip(axes, (199, 150, 100, 50, 20, 0)):
        a.scatter(data[:2000, 0], data[:2000, 1], s=1, color=MUTED, alpha=0.25)
        a.scatter(snaps[t][:, 0], snaps[t][:, 1], s=1.5, color=PRIMARY, alpha=0.6)
        a.set_title(f"reverse step t = {t}", fontsize=12)
        a.set_xlim(-1.7, 1.7)
        a.set_ylim(-1.7, 1.7)
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect("equal")
    save(fig, OUT / "reverse_2d.png")
    metrics["ddpm2d_final_loss"] = float(np.mean(losses[-200:]))

    # --- A3 tiny DDPM on MNIST
    class Block(nn.Module):
        def __init__(self, cin, cout, tdim):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.t = nn.Linear(tdim, cout)
            self.n1 = nn.GroupNorm(8, cout)
            self.n2 = nn.GroupNorm(8, cout)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, temb):
            h = F.silu(self.n1(self.c1(x)))
            h = h + self.t(temb)[:, :, None, None]
            h = F.silu(self.n2(self.c2(h)))
            return h + self.skip(x)

    def temb_fn(t, dim=64):
        half = dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], 1)

    class TinyUNet(nn.Module):
        def __init__(self, c=32, tdim=64):
            super().__init__()
            self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
            self.inc = nn.Conv2d(1, c, 3, padding=1)
            self.d1 = Block(c, c, tdim)
            self.d2 = Block(c, 2 * c, tdim)
            self.mid = Block(2 * c, 2 * c, tdim)
            self.u2 = Block(4 * c, c, tdim)
            self.u1 = Block(2 * c, c, tdim)
            self.out = nn.Conv2d(c, 1, 3, padding=1)

        def forward(self, x, t):
            te = self.tmlp(temb_fn(t))
            h0 = self.inc(x)
            h1 = self.d1(h0, te)                      # 28
            h2 = self.d2(F.avg_pool2d(h1, 2), te)     # 14
            m = self.mid(F.avg_pool2d(h2, 2), te)     # 7
            u = F.interpolate(m, scale_factor=2)      # 14
            u = self.u2(torch.cat([u, h2], 1), te)
            u = F.interpolate(u, scale_factor=2)      # 28
            u = self.u1(torch.cat([u, h1], 1), te)
            return self.out(u)

    dev = DEV
    b, a, ab = [v.to(dev) for v in schedule()]
    net = TinyUNet().to(dev)
    print("tiny UNet parameters:", sum(p.numel() for p in net.parameters()))
    dl = DataLoader(mn, batch_size=128, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    EPOCHS = 8 if dev == "cuda" else 3
    t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0
        for i, (x, _) in enumerate(dl):
            x = x.to(dev)
            t = torch.randint(0, 1000, (x.size(0),), device=dev)
            eps = torch.randn_like(x)
            xt = ab[t].sqrt()[:, None, None, None] * x + (1 - ab[t]).sqrt()[:, None, None, None] * eps
            loss = F.mse_loss(net(xt, t), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  MNIST DDPM epoch {ep + 1}: loss {tot / (i + 1):.4f}")
    metrics["mnist_ddpm"] = {"epochs": EPOCHS, "device": dev + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""),
                             "train_minutes": round((time.time() - t0) / 60, 1), "final_loss": tot / (i + 1),
                             "unet_params": sum(p.numel() for p in net.parameters())}

    @torch.no_grad()
    def sample(n, steps_keep=(999, 800, 600, 400, 200, 100, 0)):
        x = torch.randn(n, 1, 28, 28, device=dev)
        keep = {}
        for t in reversed(range(1000)):
            tt = torch.full((n,), t, device=dev)
            e = net(x, tt)
            x = (x - b[t] / (1 - ab[t]).sqrt() * e) / a[t].sqrt()
            if t > 0:
                x = x + b[t].sqrt() * torch.randn_like(x)
            if t in steps_keep:
                keep[t] = x.clamp(-1, 1).cpu()
        return keep

    net.eval()
    keep = sample(8)
    order = sorted(keep, reverse=True)
    np.savez_compressed(OUT / "reverse_mnist.npz", **{str(t): keep[t].numpy() for t in order})  # re-plot without retraining
    fig, axes = plt.subplots(len(order), 1, figsize=(9.8, 1.05 * len(order) + 0.2))
    for ax, t in zip(axes, order):
        ax.imshow(np.hstack(keep[t][:, 0].numpy()), cmap="gray_r", vmin=-1, vmax=1)
        ax.text(-8, 14, f"t = {t}", ha="right", va="center", fontsize=12, color="#1E1B4B")  # label left of the row
        ax.axis("off")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.01, hspace=0.12)
    save(fig, OUT / "reverse_mnist.png")
    final = sample(48, steps_keep=(0,))[0]
    sheet = np.vstack([np.hstack(final[i * 12:(i + 1) * 12, 0].numpy()) for i in range(4)])
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.imshow(sheet, cmap="gray_r", vmin=-1, vmax=1)
    ax.axis("off")
    save(fig, OUT / "mnist_samples.png")


# ============================================================ PART B
def part_b():
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, StableDiffusionPipeline

    # GTX 16xx cards produce NaN (black) images in float16, so test once and fall back to float32 + offload.
    state = {"dtype": torch.float16 if DEV == "cuda" else torch.float32}

    def load(model_id):
        dt = state["dtype"]
        if "v1-5" in model_id:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dt, safety_checker=None, requires_safety_checker=False)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dt)
        if DEV == "cuda":
            if dt == torch.float16:
                pipe.enable_model_cpu_offload()
            else:
                pipe.enable_attention_slicing()
                pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run(pipe, **kw):
        imgs = pipe(**kw).images
        arr = np.array(imgs[0]).astype(np.float32)
        if np.isnan(arr).any() or arr.max() < 5:
            raise RuntimeError("black/NaN image")
        return imgs[0]

    prompt = "a watercolour painting of a lighthouse on a cliff at sunset, seagulls, soft light"
    sd = load("stable-diffusion-v1-5/stable-diffusion-v1-5")
    sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
    try:
        run(sd, prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0))
    except RuntimeError as e:
        print("float16 gives black/NaN images on this GPU; switching to float32:", e)
        del sd
        torch.cuda.empty_cache()
        state["dtype"] = torch.float32
        sd = load("stable-diffusion-v1-5/stable-diffusion-v1-5")
        sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
    metrics["sd_dtype"] = str(state["dtype"])

    def row(images, titles, name, size=3.0):
        fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size + 0.5))
        for ax, im, t in zip(axes, images, titles):
            ax.imshow(im)
            ax.set_title(t, fontsize=13)
            ax.axis("off")
        save(fig, OUT / name)
        MPATH.write_text(json.dumps(metrics, indent=2))
        print("saved", name, flush=True)

    gen = lambda s: torch.Generator("cpu").manual_seed(s)  # noqa: E731
    times = {}
    metrics["sd_times_seconds"] = times
    # guidance sweep
    ims, tl = [], []
    for g in (1.0, 3.0, 7.5, 15.0):
        t0 = time.time()
        ims.append(run(sd, prompt=prompt, guidance_scale=g, num_inference_steps=25, generator=gen(42)))
        times[f"sd15_25steps_g{g}"] = round(time.time() - t0, 1)
        tl.append(f"guidance = {g}")
    row(ims, tl, "sd_guidance.png")
    # steps sweep
    ims, tl = [], []
    for s in (2, 5, 10, 25):
        t0 = time.time()
        ims.append(run(sd, prompt=prompt, guidance_scale=7.5, num_inference_steps=s, generator=gen(42)))
        times[f"sd15_{s}steps"] = round(time.time() - t0, 1)
        tl.append(f"{s} steps")
    row(ims, tl, "sd_steps.png")
    # seeds
    ims = [run(sd, prompt=prompt, guidance_scale=7.5, num_inference_steps=25, generator=gen(s)) for s in (1, 2, 3, 4)]
    row(ims, [f"seed {s}" for s in (1, 2, 3, 4)], "sd_seeds.png")
    # negative prompt
    p2 = "a cosy reading corner with a green armchair and a lamp, photograph"
    a1 = run(sd, prompt=p2, guidance_scale=7.5, num_inference_steps=25, generator=gen(7))
    a2 = run(sd, prompt=p2, negative_prompt="lamp, blurry, low quality", guidance_scale=7.5, num_inference_steps=25, generator=gen(7))
    row([a1, a2], ["no negative prompt", "negative: 'lamp, blurry, low quality'"], "sd_negative.png", size=3.6)
    del sd
    if DEV == "cuda":
        torch.cuda.empty_cache()
    # SD-Turbo: 1 step
    turbo = load("stabilityai/sd-turbo")
    ims, tl = [], []
    for s in (1, 2, 4):
        t0 = time.time()
        ims.append(run(turbo, prompt=prompt, guidance_scale=0.0, num_inference_steps=s, generator=gen(42)))
        times[f"sdturbo_{s}steps"] = round(time.time() - t0, 1)
        tl.append(f"SD-Turbo, {s} step{'s' if s > 1 else ''}")
    row(ims, tl, "sd_turbo.png")
    metrics["sd_times_seconds"] = times
    metrics["sd_device"] = DEV + (" " + torch.cuda.get_device_name(0) if DEV == "cuda" else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    if a.part in ("A", "all"):
        part_a()
    if a.part in ("B", "all"):
        part_b()
    MPATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
