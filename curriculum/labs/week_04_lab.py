# %% [markdown]
# # Week 4 Lab: Train a diffusion model, then use Stable Diffusion
#
# **Module:** Generative AI (MSc in Artificial Intelligence) · **Time:** 2 hours · **Learning outcomes:** MIMLO 1, 2, 3, 4
#
# Part A — build it yourself:
# 1. implement the **forward (noising) process** in closed form;
# 2. train a **DDPM** on 2-D data and write the **sampling loop**; optionally train a tiny **U-Net DDPM** on MNIST;
# 3. write a **DDIM** sampler using selected levels across the full schedule.
#
# Part B — inspect stored image evidence; optionally use a pretrained model (**Diffusers**):
# 4. generate images with **Stable Diffusion 1.5**; sweep **steps, guidance scale, seeds**; try **negative prompts** and **schedulers**;
# 5. compare with the distilled **SD-Turbo** (1–4 steps) and measure time;
# 6. evaluate prompt alignment with a **CLIP score** you implement.
#
# **CPU core:** train a real small diffusion model on synthetic 2-D points;
# implement and check the noising, loss, reverse update and DDIM jump;
# inspect stored image experiments and calculate similarity on small vectors.
# This path needs no image-model or dataset download. The teaching notes give
# the measured CPU time on the course laptop; well under the 15-minute budget.
#
# **Optional stretches:** enable MNIST training or pretrained image generation
# only after the core. A GPU is optional. Stored lecture figures show separate
# measured MNIST and Stable Diffusion runs, not results from your new point model.
#
# Use the injected setup's `DEVICE`, `SMOKE`, `check()` and `ask_model()`.
# It selects CPU by default. Keep `USE_GEMINI = False` and `USE_OPENAI = False` for the local core.
# For optional feedback, enable exactly one switch in the shared setup.
# Stop after each TODO until its targeted check prints a check mark.
#
# **Image-model use:** use synthetic scenes, record the checkpoint, read its
# model card and licence, and retain the pipeline's supplied safety components.

# %% [markdown]
# ## Before coding: training and generation use different loops
#
# Training corrupts known data and teaches a network to predict the added noise.
# Generation starts from new noise and reuses that network at decreasing noise
# levels. The **denoiser** learns; the **sampler** decides how each state changes.
#
# ![Train a network to predict noise](fig:beginner_overview)
#
# Follow `diffusion_loss` → `q_sample` → `EpsMLP` or `TinyUNet` → the noise
# comparison → `opt.step`. The sampled noise is a separate target input to the
# loss, not an input to the network. Only the denoiser weights are updated.
#
# ![Build diffusion, then inspect a pipeline](fig:beginner_lab)
#
# | Diagram block | Code to find | First observation |
# |---|---|---|
# | Visualise noising | `q_sample`, `abar` | Larger timestep indices usually retain less signal. |
# | Learn on points | `EpsMLP`, `diffusion_loss` | The network predicts two noise values per point. |
# | Learn on digits | `TinyUNet` | Predicted noise has the same shape as the image batch. |
# | Compare samplers | `sample_ddpm`, `sample_ddim` | DDIM selects timesteps across the full schedule. |
# | Generate from text | `pipe`, `generate` | A pretrained system adds text encoding and latent decoding. |
# | Evaluate controlled runs | `grid`, `clip_score` | Record the active model, controls, timing, and visual evidence. |
#
# **Pause and predict:** does a training target contain the clean image or the
# noise that was drawn? Trace the target path before implementing the loss.

# %% [markdown]
# ## Read, run, change, check
#
# **Read:** begin with small arrays and stored pictures showing added noise.
# **Run:** complete `q_sample`, then the loss and reverse-step TODOs before
# training. The same loss is used for points and for digit images.
# **Change:** compare DDPM and DDIM using the same trained denoiser. In the
# pretrained pipeline, vary one control while keeping the prompt and seed fixed.
# **Check:** name the model and sampler in each evidence table. The core
# point model, stored lecture images and optional pretrained pipelines differ.
#
# A **denoiser** predicts the noise in an input. A **noise schedule** records
# how noisy each step is. A **sampler** uses these predictions and schedule
# values to move towards an image. **MSE** is mean squared error: square each
# prediction error, then average. Smaller MSE means closer noise predictions.

# %% [markdown]
# **Purpose:** Install pretrained-pipeline libraries.
# **Why now:** Later cells use Diffusers and Transformers.
# **Expected observation:** Installation completes or reports a dependency error.
#

# %% tags=["colab-install"]
%pip install -q diffusers transformers accelerate scikit-learn matplotlib

# %% [markdown]
# **Purpose:** Import lab tools and choose optional stretches.
# **Why now:** The shared setup has already selected seeds, device and provider access.
# **Expected observation:** CPU core settings and three readiness checks.
#

# %%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
# Keep long training and downloaded image models outside the required path.
TRAIN_MNIST = False
RUN_IMAGE_STRETCH = False
POINT_UPDATES = 100 if SMOKE else 10000
POINT_BATCH = 256
CORE_SAMPLES = 128 if SMOKE else 512
core_started = time.perf_counter()

ready = check(torch.tensor([[1.0, 2.0]]).shape == (1, 2),
              "An array shape tells us rows and entries per row.", "Read [1, 2] as one point with two coordinates.")
ready &= check(abs(0.8 * 1.0 + 0.6 * (-0.5) - 0.5) < 1e-8,
               "The scalar noising example gives 0.5.", "Multiply each coefficient before adding the two terms.")
ready &= check(list(reversed(range(3))) == [2, 1, 0],
               "The reverse loop visits decreasing integer indices.", "range(3) contains 0, 1 and 2.")
assert ready, "Review arrays and the scalar calculation before continuing."
def image_lab_device(force_cpu=False):
    """Keep full image pipelines on GPUs with at least 8 GB, or use CPU."""
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda" if torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3 else "cpu"


# Shared DEVICE stays unchanged; the memory helper is used only by the optional image pipeline.
print("Core device:", DEVICE, "| point updates:", POINT_UPDATES)

# %% [markdown]
# ## Part A1 · The forward process
#
# > **Core:** implement direct noising for points and image-shaped arrays.
# > **Change one thing:** move the noise-level slider with the clean data and noise fixed.
# > **Optional stretch:** inspect the stored MNIST corruption strip below.
# > **Common errors:** reshape one coefficient per example, not one per coordinate.
#
# **Predict:** if the signal coefficient is 1, what contribution does noise make?
#
# With a noise schedule $\beta_1, \ldots, \beta_T$, define $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s \le t} \alpha_s$. Then
#
# $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$
#
# **TODO 1:** complete `q_sample(x0, t, eps)`. `t` contains one index per example.
# **Hint:** multiply `x0` by `a.sqrt()` and `eps` by `(1-a).sqrt()`, then add them.
#
# **Read the shapes:** a point batch is `[B, 2]`; an image batch is
# `[B, 1, 28, 28]`. There is one timestep coefficient per batch item. The
# extra singleton dimensions let that coefficient apply to all its entries.
# Index 0 already adds a small amount of noise: the clean input is `x0`.
#
# Read the formula as "scaled clean data + scaled random noise". `x0` is clean
# data, `xt` is its noisy version, and `eps` has independent standard Gaussian
# values. `beta` sets a noise increment; `alpha` is `1 - beta`; `abar` multiplies
# the alpha values up to a step. Larger `abar` keeps more of the clean signal.
# This lab indexes the first noisy step as 0, whereas the written schedule
# starts at 1. `T` is the number of diffusion steps, not a decoding temperature.

# %% [markdown]
# **Purpose:** Implement direct noising.
# **Why now:** A chosen timestep can be reached without earlier steps.
# **Expected observation:** Correct point/image shapes, a passed noising check and a signal coefficient.
#

# %%
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, dim=0)


def q_sample(x0, t, eps, abar=abar):
    a = abar.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
    ### BEGIN SOLUTION
    return a.sqrt() * x0 + (1 - a).sqrt() * eps
    ### END SOLUTION
    ### STUB return x0  # TODO 1


def check_q_sample(fn):
    try:
        clean = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        noise = torch.full_like(clean, -0.5)
        levels = torch.tensor([1.0, 0.64])
        got = fn(clean, torch.tensor([0, 1]), noise, levels)
        expected = torch.tensor([[1.0, -1.0], [0.5, -1.1]])
        images = clean.view(2, 1, 1, 2)
        image_got = fn(images, torch.tensor([0, 1]), noise.view_as(images), levels)
        ok = got.shape == clean.shape and torch.allclose(got, expected)
        ok = bool(ok and image_got.shape == images.shape and torch.allclose(image_got.view_as(clean), expected))
    except (TypeError, ValueError, RuntimeError, AttributeError):
        ok = False
    return check(ok, "TODO 1 mixes signal and noise, with one coefficient per example.",
                 "Use both square roots and broadcast across every non-batch dimension.")

assert check_q_sample(q_sample), "Complete TODO 1 before continuing."
print("Point shape:", (2, 2), "| image test shape:", (2, 1, 1, 2))
print("Signal variance coefficient at index 999:", float(abar[-1]))

# %% [markdown]
# **Purpose:** Explore corruption with fixed synthetic points and fixed noise.
# **Why now:** Holding those arrays fixed isolates the effect of the noise level.
# **Expected observation:** The point cloud changes as the index increases, without training.
# If widgets are unavailable, call `show_noise_level(500)` in an ordinary code cell.

# %% tags=["skip-test"]
from sklearn.datasets import make_swiss_roll

slider_clean = torch.tensor(make_swiss_roll(300, noise=0.4, random_state=SEED)[0][:, [0, 2]] / 10,
                            dtype=torch.float32)
slider_noise = torch.randn(slider_clean.shape, generator=torch.Generator().manual_seed(SEED))

def show_noise_level(index=500):
    levels = torch.full((len(slider_clean),), int(index), dtype=torch.long)
    noisy = q_sample(slider_clean, levels, slider_noise)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(*slider_clean.T, s=8, color="lightgray", label="clean points")
    ax.scatter(*noisy.T, s=8, label="noisy points")
    ax.set_title(f"index {index}, signal variance fraction {float(abar[index]):.3f}")
    ax.legend(); ax.set_aspect("equal"); plt.show()

try:
    import ipywidgets as widgets
    from IPython.display import display
    control = widgets.IntSlider(value=500, min=0, max=999, step=25, description="Noise index", continuous_update=False)
    display(control, widgets.interactive_output(show_noise_level, {"index": control}))
except ImportError:
    show_noise_level(500)
    print("Change the index in show_noise_level(index) to explore without widgets.")

# %% [markdown]
# **Stored image example:** the MNIST strip uses the 1000-step linear schedule.
# These are recorded digit corruptions, not outputs from the 2-D model below.
#
# ![Stored MNIST digits at selected noise levels](fig:forward_mnist_slide)

# %% [markdown]
# ## Part A2 · A DDPM on 2-D data
#
# > **Core:** train the existing small MLP and inspect full 200-step sampling.
# > **Change one thing:** compare one longer training budget after saving the first outputs.
# > **Optional stretch:** adapt the training target to flow matching at the end.
# > **Common errors:** the known noise is a loss target, not a network input.
#
# **Predict:** does one training update require all 200 reverse steps?
#
# **TODO 2 (training loss):** draw timesteps and noise, make $x_t$, then compare predicted and known noise.
# **Hint:** use `torch.randint`, `torch.randn_like`, `q_sample`, and `F.mse_loss` in that order.
#
# **TODO 3 (reverse step):** implement one DDPM ancestral step.
# **Hint:** compute the displayed mean; add fresh noise only when `t > 0`:
#
# $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\hat\epsilon\right) + \sqrt{\beta_t}\,z, \qquad z \sim \mathcal{N}(0, I) \text{ (no noise when } t = 0)$$
#
# **Trace training:** keep the sampled `eps` as the known answer, give the
# network only `xt` and `t`, then compare its prediction with `eps`.
# **Trace generation:** there is no known clean target; predict noise from
# the current state, take one reverse step, and repeat. A hat, as in
# `eps_hat`, means "the model's estimate". This teaching DDPM uses `beta_t`
# as the reverse variance; other DDPM implementations can use a different one.

# %% [markdown]
# **Purpose:** Train a point noise predictor.
# **Why now:** Two coordinates make the reverse trajectory visible.
# **Expected observation:** Loss and three reverse snapshots; coverage must be inspected.
#
# ![Create one noise-prediction training example](fig:beginner_training)
#

# %%
from sklearn.datasets import make_swiss_roll

T2 = 200
b2 = torch.linspace(1e-4, 0.05, T2)
a2 = 1 - b2
ab2 = torch.cumprod(a2, 0)
data2d = torch.tensor(make_swiss_roll(20000, noise=0.4, random_state=SEED)[0][:, [0, 2]] / 10.0, dtype=torch.float32)


class EpsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T2).unsqueeze(1)], dim=1))


def diffusion_loss(model, x0, T_, abar_):
    # The model sees noisy data and t; the known noise follows the target path.
    ### BEGIN SOLUTION
    t = torch.randint(0, T_, (x0.size(0),), device=x0.device)
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, eps, abar_)
    return F.mse_loss(model(xt, t), eps)
    ### END SOLUTION


def reverse_step(model, x, t, betas_, alphas_, abar_):
    tt = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
    e = model(x, tt)
    ### BEGIN SOLUTION
    mean = (x - betas_[t] / (1 - abar_[t]).sqrt() * e) / alphas_[t].sqrt()
    return mean + (betas_[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    ### END SOLUTION
    ### STUB return x  # TODO 3


def check_diffusion_loss(fn):
    class RecordZero(nn.Module):
        def forward(self, x, t):
            self.noisy, self.times = x.detach(), t.detach()
            return torch.zeros_like(x)
    try:
        probe = RecordZero()
        clean = torch.arange(32, dtype=torch.float32).view(16, 2) / 32
        levels = torch.tensor([0.9, 0.7, 0.5, 0.3])
        with torch.random.fork_rng():
            torch.manual_seed(123)
            got = fn(probe, clean, 4, levels)
        a = levels[probe.times].view(-1, 1)
        known_noise = (probe.noisy - a.sqrt() * clean) / (1 - a).sqrt()
        expected = known_noise.square().mean()
        ok = (probe.times.shape == (16,) and probe.times.dtype == torch.long
              and probe.times.min() >= 0 and probe.times.max() < 4
              and isinstance(got, torch.Tensor) and got.ndim == 0
              and torch.isfinite(got) and torch.allclose(got, expected, atol=1e-6))
    except (TypeError, ValueError, RuntimeError, AttributeError, IndexError):
        ok = False
    return check(bool(ok), "TODO 2 compares the prediction with the actual sampled noise.",
                 "Pass only noisy data and timesteps to the model; average squared error against eps.")

def check_reverse_step(fn):
    class ConstantNoise(nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 0.2)
    try:
        probe, x = ConstantNoise(), torch.tensor([[0.5, -0.25], [0.1, 0.8]])
        bs = torch.tensor([0.1, 0.2]); aa = 1 - bs; ab = aa.cumprod(0)
        final_mean = (x - bs[0] / (1 - ab[0]).sqrt() * 0.2) / aa[0].sqrt()
        with torch.random.fork_rng():
            torch.manual_seed(123)
            final = fn(probe, x, 0, bs, aa, ab)
            torch.manual_seed(123)
            noise = torch.randn_like(x)
            expected = (x - bs[1] / (1 - ab[1]).sqrt() * 0.2) / aa[1].sqrt() + bs[1].sqrt() * noise
            torch.manual_seed(123)
            earlier = fn(probe, x, 1, bs, aa, ab)
        ok = final.shape == x.shape and torch.allclose(final, final_mean, atol=1e-6)
        ok = bool(ok and torch.allclose(earlier, expected, atol=1e-6))
    except (TypeError, ValueError, RuntimeError, AttributeError, IndexError):
        ok = False
    return check(ok, "TODO 3 uses predicted noise and adds random noise only before the final step.",
                 "Check the schedule indices, both square roots and the t > 0 condition.")

assert check_diffusion_loss(diffusion_loss), "Complete TODO 2 before training."
assert check_reverse_step(reverse_step), "Complete TODO 3 before sampling."
print("Training points:", tuple(data2d.shape), "| batch:", (POINT_BATCH, 2))

m2 = EpsMLP().to(DEVICE)
point_data = data2d.to(DEVICE)
b2, a2, ab2 = b2.to(DEVICE), a2.to(DEVICE), ab2.to(DEVICE)
opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
point_losses = []
for step in range(POINT_UPDATES):
    indices = torch.randint(0, len(point_data), (POINT_BATCH,), device=DEVICE)
    loss = diffusion_loss(m2, point_data[indices], T2, ab2)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == POINT_UPDATES - 1:
        point_losses.append(float(loss.detach().cpu()))
print("Final 2-D loss:", round(loss.item(), 4), "| output shape:", (POINT_BATCH, 2))
assert check(np.isfinite(point_losses).all(), "Recorded training losses are finite.",
             "Check the noising and loss TODOs before changing learning rate."), "Loss became non-finite."

with torch.no_grad():
    m2.eval()
    point_sample_start = time.perf_counter()
    x = torch.randn(CORE_SAMPLES, 2, device=DEVICE)
    shots = {}
    for t in reversed(range(T2)):
        x = reverse_step(m2, x, t, b2, a2, ab2)
        if t in (150, 50, 0):
            shots[t] = x.detach().cpu().clone()
point_ddpm_time = time.perf_counter() - point_sample_start
point_ddpm = shots[0]
print("Generated point shape:", tuple(point_ddpm.shape), "| DDPM seconds:", round(point_ddpm_time, 3))
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, t in zip(axes, (150, 50, 0)):
    ax.scatter(*data2d[:2000].T, s=1, color="lightgray")
    ax.scatter(*shots[t].T, s=1)
    ax.set_title(f"reverse step t = {t}"); ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Part A3 · Optional stretch: a tiny U-Net DDPM on MNIST
#
# > **Core:** trace the network shapes and inspect the stored digit figure.
# > **Change one thing:** predict what reducing channels changes before opting in.
# > **Optional stretch:** set `TRAIN_MNIST = True`, then run training and image sampling.
# > **Common errors:** a download failure concerns data access, not the diffusion equations.
#
# The stored image run is separate from the new CPU point run. The stretch
# keeps the original image network and full DDPM mechanism; it can take longer.
#
# The denoiser must output a tensor of the same shape as its input. A small **U-Net** (down-sample, up-sample, skip connections) with a **sinusoidal timestep embedding** does the job. Training uses exactly your `diffusion_loss`.
# The embedding turns a timestep number into features. Skip connections pass
# saved spatial information from the shrinking path to the expanding path.
# They join features; they do not supply the clean training image to the model.

# %% [markdown]
# **Purpose:** Train and sample the tiny image U-Net.
# **Why now:** One model receives all timestep values.
# **Expected observation:** Recorded losses and a full DDPM sample row; poor samples are possible.
#
# ![Generate by repeating reverse steps](fig:beginner_mechanism)
#

# %%
def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim=64):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.t = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x))) + self.t(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(h))) + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.inc = nn.Conv2d(1, c, 3, padding=1)
        self.d1, self.d2, self.mid = Block(c, c), Block(c, 2 * c), Block(2 * c, 2 * c)
        self.u2, self.u1 = Block(4 * c, c), Block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x, t):
        te = self.tmlp(timestep_embedding(t))
        h1 = self.d1(self.inc(x), te)                        # 28x28
        h2 = self.d2(F.avg_pool2d(h1, 2), te)                # 14x14
        m = self.mid(F.avg_pool2d(h2, 2), te)                # 7x7
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)
        return self.out(u)


unet = None
if TRAIN_MNIST and not SMOKE:
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mnist = datasets.MNIST("data", train=True, download=True, transform=tf)
    unet = TinyUNet().to(DEVICE)
    print(f"TinyUNet: {sum(p.numel() for p in unet.parameters()):,} parameters")
    dl = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(unet.parameters(), lr=2e-3)
    EPOCHS = 6 if DEVICE == "cuda" else 2
    t0 = time.time()
    for epoch in range(EPOCHS):
        for x, _ in dl:
            loss = diffusion_loss(unet, x.to(DEVICE), T, abar)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.4f} ({time.time() - t0:.0f} s)")
else:
    print("MNIST training skipped. Inspect the stored, separately measured digit run.")

betas_d, alphas_d, abar_d = betas.to(DEVICE), alphas.to(DEVICE), abar.to(DEVICE)


@torch.no_grad()
def sample_ddpm(model, n=16, steps=T):
    # Use the full T-step schedule here. Lowering steps only truncates this chain;
    # it does not make valid large jumps from index T-1. For skipping, use DDIM.
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=DEVICE)
    for t in reversed(range(steps)):
        x = reverse_step(model, x, t, betas_d, alphas_d, abar_d)
    return x.clamp(-1, 1).cpu()


def show_row(imgs, title):
    plt.figure(figsize=(12, 1.2)); plt.imshow(np.hstack(imgs[:, 0].numpy()), cmap="gray_r", vmin=-1, vmax=1)
    plt.axis("off"); plt.title(title, loc="left"); plt.show()


if unet is not None:
    t0 = time.time()
    ddpm_imgs = sample_ddpm(unet, n=16)
    ddpm_time = time.time() - t0
    show_row(ddpm_imgs, f"DDPM, 1000 steps ({ddpm_time:.1f} s)")

# %% [markdown]
# ### A4 · A faster sampler: DDIM
#
# > **Core:** complete the deterministic jump and compare 40 selected point-model steps with 200 DDPM steps.
# > **Change one thing:** try 20 selected steps using the same trained point denoiser.
# > **Optional stretch:** compare 50-step image DDIM after enabling MNIST training.
# > **Common errors:** truncating a DDPM loop does not span the full schedule.
#
# DDIM (Song, Meng & Ermon, 2020) takes large **deterministic** steps along a sub-sequence of timesteps. From the noise prediction $\hat\epsilon$ at step $t$:
#
# $$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{x}_0 + \sqrt{1-\bar\alpha_{t'}}\,\hat\epsilon$$
#
# where $t' < t$ is the next selected timestep. **TODO 4:** implement the DDIM update.
# **Hint:** first solve for `x0_hat`; then use the next signal and noise coefficients.
# Unlike shortening the DDPM loop, `seq` spans index `T - 1` down to 0.
# Each DDIM update explicitly uses both the current and next schedule values.
# Read the two equations in order: first estimate the clean image, then mix
# that estimate with the predicted noise at a lower noise level. The last step
# uses `a_next=1`, so its remaining-noise coefficient is zero.

# %% [markdown]
# **Purpose:** Implement schedule-spanning DDIM.
# **Why now:** Compatible larger jumps reduce calls without retraining.
# **Expected observation:** 40 selected point-model levels and measured time; coverage may change.
#

# %%
@torch.no_grad()
def sample_ddim(model, n=16, steps=50, shape=(1, 28, 28), abar_=None):
    # Original image calls still work; passing shape=(2,) reuses the equation on points.
    levels = abar_d if abar_ is None else abar_.to(DEVICE)
    if not 2 <= steps <= len(levels):
        raise ValueError("Choose between 2 and the number of schedule entries.")
    model.eval()
    x = torch.randn(n, *shape, device=DEVICE)
    seq = torch.linspace(len(levels) - 1, 0, steps).long().tolist()
    for i, t in enumerate(seq):
        t_next = seq[i + 1] if i + 1 < len(seq) else -1
        eps = model(x, torch.full((n,), t, device=DEVICE, dtype=torch.long))
        a_t = levels[t]
        a_next = levels[t_next] if t_next >= 0 else levels.new_tensor(1.0)
        ### BEGIN SOLUTION
        x0_hat = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x = a_next.sqrt() * x0_hat + (1 - a_next).sqrt() * eps
        ### END SOLUTION
    return (x.clamp(-1, 1) if len(shape) == 3 else x).cpu()

def check_ddim_sampler(fn):
    class RecordZero(nn.Module):
        def __init__(self):
            super().__init__(); self.seen = []
        def forward(self, x, t):
            self.seen.append(int(t[0])); return torch.zeros_like(x)
    class RecordNonzero(RecordZero):
        def forward(self, x, t):
            self.seen.append(int(t[0])); return torch.full_like(x, 0.2)
    try:
        probe = RecordZero()
        levels = torch.tensor([0.9, 0.7, 0.4, 0.2], device=DEVICE)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE == "cuda" else []):
            torch.manual_seed(123)
            initial = torch.randn(2, 2, device=DEVICE)
            torch.manual_seed(123)
            got = fn(probe, n=2, steps=3, shape=(2,), abar_=levels)
        expected = (initial / levels[-1].sqrt()).cpu()
        ok = got.shape == (2, 2) and torch.isfinite(got).all()
        ok = bool(ok and probe.seen == [3, 1, 0] and torch.allclose(got, expected, atol=1e-5))
        nonzero_probe = RecordNonzero()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE == "cuda" else []):
            torch.manual_seed(123)
            nonzero = fn(nonzero_probe, n=2, steps=3, shape=(2,), abar_=levels)
        # Constant predicted noise makes the selected updates telescope to this clean estimate.
        nonzero_expected = ((initial - (1 - levels[-1]).sqrt() * 0.2) / levels[-1].sqrt()).cpu()
        ok = bool(ok and nonzero_probe.seen == [3, 1, 0]
                  and torch.allclose(nonzero, nonzero_expected, atol=1e-5))
    except (TypeError, ValueError, RuntimeError, AttributeError, IndexError):
        ok = False
    return check(ok, "TODO 4 spans the schedule and uses nonzero predicted noise in its clean estimate.",
                 "Use eps in both formulas and both schedule coefficients; final a_next is 1.")

assert check_ddim_sampler(sample_ddim), "Complete TODO 4 before comparing samplers."
t0 = time.perf_counter()
point_ddim = sample_ddim(m2, n=CORE_SAMPLES, steps=40, shape=(2,), abar_=ab2)
point_ddim_time = time.perf_counter() - t0
print("DDIM point shape:", tuple(point_ddim.shape), "| seconds:", round(point_ddim_time, 3))
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, points, title in zip(axes, (point_ddpm, point_ddim), ("DDPM: 200 calls", "DDIM: 40 calls")):
    ax.scatter(*data2d[:2000].T, s=1, color="lightgray")
    ax.scatter(*points.T, s=2)
    ax.set_title(title); ax.set_aspect("equal")
plt.show()
if unet is not None:
    t0 = time.time()
    ddim_imgs = sample_ddim(unet, n=16, steps=50)
    show_row(ddim_imgs, f"DDIM, 50 steps ({time.time() - t0:.1f} s)")

# %% [markdown]
# ✍️ **Question 1.** Compare the observed core DDPM (200 calls) with DDIM (40 calls). Why can DDIM skip levels without retraining?
#
# <!-- BEGIN ANSWER -->
# The core compares 40 with 200 calls. The optional digit stretch compares 50 with 1000. Measure actual time and describe coverage and visible failures rather than assuming a matching wall-clock ratio. This DDIM implementation is deterministic given its starting noise and fixed model; DDPM injects fresh noise at intermediate reverse steps. DDIM does not need retraining because it uses the same trained noise predictor at selected noise levels. Its prediction also gives an estimate of the clean image x̂₀. DDIM combines x̂₀ and ε̂ with the next timestep's coefficients to jump to that lower noise level. The sampling procedure changes, not the trained weights. Simply setting sample_ddpm(steps=50) would instead truncate the chain and would not implement these jumps.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part B · Stored evidence and optional Stable Diffusion stretch
#
# > **Core:** trace the components, complete the synthetic negative-prompt call check, and inspect stored images.
# > **Change one thing:** compare stored guidance settings with the seed and prompt fixed.
# > **Optional stretch:** set `RUN_IMAGE_STRETCH = True` after the core; expect downloads and extra waiting.
# > **Common errors:** SD-Turbo ignores guidance and negative prompts; it cannot test those controls.
#
# A `DiffusionPipeline` bundles the **text encoder**, the **denoiser** (U-Net), the **VAE** and a **scheduler** (sampler). On a T4 GPU we load it in half precision (float16).
# The stretch uses SD 1.5 only after GPU opt-in with enough memory. CPU uses
# SD-Turbo if you explicitly enable this stretch. The required core loads neither.
# Historical SD 1.5 teaches ordinary guidance; Turbo teaches model-specific distillation.
# Read [SD-Turbo settings](https://huggingface.co/stabilityai/sd-turbo) before changing them.
# CPU uses float32 and SD-Turbo; its controls differ from the SD 1.5 GPU experiment below.
# The text encoder supplies prompt features. The denoiser operates on a
# compressed image representation, and the VAE decoder turns the final
# representation into pixels. **Float16** stores each floating-point value
# with 16 bits; it can reduce memory use on supported hardware.

# %% [markdown]
# **Purpose:** Load and inspect the active image pipeline.
# **Why now:** The pretrained system adds text encoding and latent decoding.
# **Expected observation:** Actual model path, scheduler and component sizes.
#
# ![From a text prompt to image pixels](fig:beginner_inference)
#

# %%
pipe = None
PIPELINE_DEVICE = image_lab_device(force_cpu=DEVICE != "cuda")
USE_SD15 = RUN_IMAGE_STRETCH and PIPELINE_DEVICE == "cuda" and not SMOKE
dtype = torch.float16 if PIPELINE_DEVICE == "cuda" else torch.float32
SIZE, STEPS = 512, 25 if USE_SD15 else 2
if RUN_IMAGE_STRETCH and not SMOKE:
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, StableDiffusionPipeline
    if USE_SD15:
        pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dtype)
    pipe = pipe.to(PIPELINE_DEVICE)
    pipe.set_progress_bar_config(disable=True)
    print(type(pipe).__name__, "| scheduler:", type(pipe.scheduler).__name__)
    for name in ("text_encoder", "unet", "vae"):
        mod = getattr(pipe, name)
        print(f"{name}: {sum(p.numel() for p in mod.parameters()) / 1e6:.1f} M parameters")
else:
    print("Image-model downloads skipped. Use the stored, labelled comparisons below.")


def generate(prompt, seed=42, steps=STEPS, guidance=7.5, negative=None):
    # This is inference through pretrained components; it performs no training.
    if pipe is None:
        raise RuntimeError("Enable RUN_IMAGE_STRETCH to generate; the CPU core uses stored comparisons.")
    g = torch.Generator(device="cpu").manual_seed(seed)
    if not USE_SD15:
        guidance = 0.0  # SD-Turbo is trained without classifier-free guidance
        negative = None
    return pipe(prompt, negative_prompt=negative, num_inference_steps=steps, guidance_scale=guidance,
                generator=g, height=SIZE, width=SIZE).images[0]


def grid(images, titles, size=3):
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size + 0.4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im); ax.set_title(t); ax.axis("off")
    plt.show()


PROMPT = "a watercolour painting of a lighthouse on a cliff at sunset, seagulls, soft light"
img = None
if pipe is not None:
    t0 = time.time()
    img = generate(PROMPT)
    print(f"one image in {time.time() - t0:.1f} s")
    plt.imshow(img); plt.axis("off"); plt.show()

# %% [markdown]
# ### B1 · Sweep the controls
# Change **one thing at a time** and keep the seed fixed. SD-Turbo disables guidance.
# **Stored evidence:** the steps and guidance grids below are separate measured
# Stable Diffusion 1.5 runs, not outputs from the CPU point model or SD-Turbo.
# They hold seed 42 fixed. The step grid uses guidance 7.5 and DPM-Solver++.
# The guidance grid uses 25 steps. Inspect prompt fit and artefacts yourself.
#
# ![Stored SD 1.5 step comparison](fig:sd_steps)
#
# ![Stored SD 1.5 guidance comparison](fig:sd_guidance)

# %% [markdown]
# **Purpose:** Sweep one setting at a time.
# **Why now:** Control comparisons require a fixed seed and prompt.
# **Expected observation:** Step/guidance/seed grids; Turbo guidance is disabled.
#

# %%
guidance_values, guidance_images = [], []
if pipe is not None:
    step_values = [2, 5, 10, 25] if USE_SD15 else [1, 2, 3, 4]
    grid([generate(PROMPT, steps=s) for s in step_values], [f"{s} steps" for s in step_values])
    if USE_SD15:
        guidance_values = [1.0, 3.0, 7.5, 15.0]
        guidance_images = [generate(PROMPT, guidance=g) for g in guidance_values]
        grid(guidance_images, [f"guidance {g}" for g in guidance_values])
    grid([generate(PROMPT, seed=s) for s in (1, 2, 3, 4)], [f"seed {s}" for s in (1, 2, 3, 4)])

# %% [markdown]
# ### B2 · Negative prompts and schedulers
#
# **TODO 5:** write `negative_prompt_pair` using the same prompt and seed twice, adding a negative condition only on the second call.
# **Hint:** pass `negative="lamp, blurry, low quality"` to the second `generate` call. The check records calls without downloading a model.
# **Check the active model first:** this negative-prompt experiment is useful
# with SD 1.5's guidance. The SD-Turbo fallback sets guidance to zero, so a
# negative prompt is not an effective control there. Record that limitation
# instead of interpreting unchanged output as evidence about negative prompts.

# %% [markdown]
# **Purpose:** Test guidance references and samplers.
# **Why now:** A negative prompt depends on active guidance.
# **Expected observation:** Two prompt variants and two schedulers; unchanged Turbo negatives are expected.
#

# %%
P2 = "a cosy reading corner with a green armchair and a lamp, photograph"

def negative_prompt_pair(prompt, seed=7):
    ### BEGIN SOLUTION
    a = generate(prompt, seed=seed)
    b = generate(prompt, seed=seed, negative="lamp, blurry, low quality")
    ### END SOLUTION
    ### STUB a = b = generate(prompt, seed=seed)  # TODO 5: add a negative prompt to b
    return a, b

def check_negative_prompt_pair(fn):
    global generate
    original, calls = generate, []
    def record(prompt, seed=42, steps=STEPS, guidance=7.5, negative=None):
        calls.append((prompt, seed, negative)); return object()
    try:
        generate = record
        pair = fn("synthetic reading corner", seed=7)
        ok = len(pair) == 2 and len(calls) == 2
        ok = ok and calls[0][:2] == calls[1][:2] == ("synthetic reading corner", 7)
        ok = bool(ok and calls[0][2] is None and isinstance(calls[1][2], str) and calls[1][2].strip())
    except (TypeError, ValueError, AttributeError):
        ok = False
    finally:
        generate = original
    return check(ok, "TODO 5 holds prompt and seed fixed and changes only the negative condition.",
                 "Call generate twice with the same seed; add a nonempty negative argument only to b.")

assert check_negative_prompt_pair(negative_prompt_pair), "Complete TODO 5 before the optional image experiment."
if USE_SD15:
    a, b = negative_prompt_pair(P2)
    grid([a, b], ["no negative prompt", "with negative prompt"], size=4)
    original_scheduler = pipe.scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(original_scheduler.config)
    c = generate(P2, seed=7)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(original_scheduler.config)
    d = generate(P2, seed=7)
    pipe.scheduler = original_scheduler
    grid([c, d], ["Euler ancestral", "DPM-Solver++"], size=4)
else:
    print("The synthetic call check passed; it does not measure a visual negative-prompt effect.")

# %% [markdown]
# ### B3 · A distilled few-step model: SD-Turbo
# Compare the stored 1-, 2- and 4-step outputs with the separately recorded
# SD 1.5 experiment. Optional live comparisons must also record checkpoint and settings.
#
# ![Stored SD-Turbo with guidance disabled](fig:sd_turbo)

# %% [markdown]
# **Purpose:** Optionally compare a distilled model.
# **Why now:** A shorter trained student can reduce denoiser calls.
# **Expected observation:** Turbo times and images when a full SD 1.5 run is active.
#

# %%
if USE_SD15:
    turbo = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dtype).to(PIPELINE_DEVICE)
    turbo.set_progress_bar_config(disable=True)
    for s in (1, 4):
        t0 = time.time()
        im = turbo(PROMPT, num_inference_steps=s, guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(42)).images[0]
        print(f"SD-Turbo {s} step(s): {time.time() - t0:.2f} s")
        plt.imshow(im); plt.axis("off"); plt.title(f"SD-Turbo, {s} step(s)"); plt.show()
    del turbo
    torch.cuda.empty_cache()
else:
    print("Live Turbo comparison skipped; inspect the stored, separately measured output grid.")

# %% [markdown]
# ## Part C · Similarity and optional CLIP evaluation
#
# > **Core:** implement cosine similarity on small synthetic embedding vectors.
# > **Change one thing:** scale a vector, then change its direction; predict each score.
# > **Optional stretch:** load CLIP only after enabling the image-model stretch.
# > **Common errors:** a matching tensor shape does not prove prompt fit or image quality.
#
# CLIP (week 9) maps images and text to numerical vectors in one space. A
# **CLIP score** compares their directions using cosine similarity, often ×100.
# **TODO 6:** implement `cosine_score` from two embedding tensors.
# **Hint:** use `F.cosine_similarity(img_emb, txt_emb).item()` for one vector pair.
# Synthetic vectors demonstrate the calculation; they are not CLIP-derived image-quality scores.

# %% [markdown]
# **Purpose:** Check direction similarity; optionally evaluate real image/text embeddings.
# **Why now:** The arithmetic needs a check before interpreting a model-derived score.
# **Expected observation:** Four synthetic checks; real CLIP scores appear only after the image stretch.
#

# %%
def cosine_score(img_emb, txt_emb):
    ### BEGIN SOLUTION
    return F.cosine_similarity(img_emb, txt_emb).item()
    ### END SOLUTION
    ### STUB return 0.0  # TODO 6

def check_cosine_score(fn):
    try:
        v = torch.tensor([[1.0, 0.0]])
        got = [fn(v, other) for other in (v, 3 * v, torch.tensor([[0.0, 1.0]]), -v)]
        ok = np.allclose(got, [1.0, 1.0, 0.0, -1.0], atol=1e-6)
    except (TypeError, ValueError, RuntimeError, AttributeError):
        ok = False
    return check(bool(ok), "TODO 6 measures direction: parallel 1, orthogonal 0, opposite -1.",
                 "Normalise by both vector lengths; a raw dot product changes when a vector is scaled.")

assert check_cosine_score(cosine_score), "Complete TODO 6 before interpreting scores."
print("Synthetic vector shape:", (1, 2), "| parallel score:", cosine_score(torch.tensor([[1., 0.]]), torch.tensor([[3., 0.]])))

clip, proc = None, None
if pipe is not None:
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(PIPELINE_DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def clip_score(image, text):
    # This checks similarity in CLIP's feature space, not every visible detail.
    if clip is None:
        raise RuntimeError("CLIP is part of the optional image stretch; the core uses synthetic vectors.")
    inputs = proc(text=[text], images=image, return_tensors="pt", padding=True).to(PIPELINE_DEVICE)
    out = clip(**inputs)
    print("CLIP image/text shapes:", tuple(out.image_embeds.shape), tuple(out.text_embeds.shape))
    return cosine_score(out.image_embeds.float(), out.text_embeds.float())

rows = []
if USE_SD15:
    for gscale, im in zip(guidance_values, guidance_images):
        rows.append((f"guidance {gscale}", clip_score(im, PROMPT)))
if img is not None:
    rows.append(("unrelated prompt (control)", clip_score(img, "a plate of spaghetti on a kitchen table")))
for name, score in rows:
    print(f"{name:28s} CLIP score = {100 * score:5.1f}")

# %% [markdown]
# ✍️ **Question 2.** Give two limitations of CLIP similarity. If you ran the image stretch, compare your actual scores with your preferred image; otherwise say the core only checked the cosine calculation.
#
# <!-- BEGIN ANSWER -->
# A core-only run does not supply new CLIP image scores; its synthetic-vector checks verify the calculation. If you ran the optional pipeline, describe your actual scores and pictures. Guidance can affect alignment and artefacts, but no ordering is guaranteed; the unrelated-prompt control is a check of this experiment. The highest score is not always the image a person prefers: very high guidance can score well while looking over-saturated. Limitations: (1) CLIP measures semantic similarity in its own embedding space and can be fooled by images containing the right keywords or text, while ignoring counting, spatial relations or attribute binding (e.g. "red cube on a blue sphere"); (2) it says nothing about realism, artefacts, diversity or safety, and it inherits CLIP's biases and English-centric training. Evaluation should combine it with FID/realism metrics and human judgement using a rubric.
# <!-- END ANSWER -->
#
# ✍️ **Question 3.** Your project team wants to generate marketing images with Stable Diffusion. List three settings you would record for reproducibility and two responsible-use checks.
#
# <!-- BEGIN ANSWER -->
# Reproducibility: model name and version (e.g. stable-diffusion-v1-5), scheduler, number of steps, guidance scale, seed, image size, negative prompt, library versions and hardware. Responsible use: (1) check the model licence and the organisation's policy for commercial use and avoid generating identifiable people, trademarks or artists' styles without permission; (2) review outputs for bias and harmful content (audit prompts across demographics), keep the safety checker on, and label images as AI-generated (content credentials / EU AI Act Art. 50 disclosure where relevant).
# <!-- END ANSWER -->

# %% [markdown]
# ## Optional extension · From DDPM to flow matching
#
# Adapt both the training target and the time input: the 2-D EpsMLP divides integer t by T2, so either pass continuous t × T2 into it or change its time normalisation to accept t in [0, 1].
# Replace the training target: with $x_t = (1-t)\,x_0 + t\,\epsilon$ for $t \in [0,1]$, train the network to predict the velocity $v = \epsilon - x_0$, then sample by Euler integration from $t=1$ to $t=0$: $x \leftarrow x - \Delta t \cdot v_\theta(x, t)$. Try it on the 2-D swiss roll with 20 Euler steps.
#
# | Experiment | Setting | Measure | Result | Interpretation |
# |---|---|---|---|---|
# | Core point model | 200 DDPM vs 40 DDIM calls | time, coverage | | |
# | Optional MNIST model | 1000 vs 50 calls | time, visual quality | | |
# | Active image model | guidance sweep if supported; otherwise seed sweep | CLIP similarity, visual | | |
# | SD-Turbo vs SD 1.5 | 1–4 vs 25 steps | seconds per image | | |
#
# **Before next week:** watch 3Blue1Brown's *Transformers* and *Attention in transformers* (chapters 5–6).


# %% [markdown]
# ## Explain, compare and check
#
# > **Core:** explain why fewer sampling calls need not improve every output.
# > **Change one thing:** revise your explanation after checking the actual point clouds.
# > **Optional stretch:** choose Gemini or the OpenAI API in shared setup if your account and region permit it.
# > **Common errors:** feedback on text is not a measurement of generated images.
#
# **Purpose:** Compare a synthetic explanation with a local rubric or optional Gemini or OpenAI API feedback.
# **Why now:** The explanation should connect the observed experiment to the mechanism.
# **Expected observation:** A local checklist always appears; an optional response may also appear.
# Send only the supplied fictional explanation. API prompts may be used by the provider.
# Check the selected provider's account, regional and cost terms before opting in; the local core is complete.

# %%
synthetic_explanation = "Fewer DDIM steps always produce better images because the model learns faster during generation."
local_feedback = "Check: weights stay fixed during generation; DDIM changes the numerical updates; inspect coverage and quality, and measure time."
feedback = ask_model("Review this fictional student explanation in two sentences. Identify its mechanism error: " + synthetic_explanation,
                      max_output_tokens=128)
print("Local rubric:", local_feedback)
print("Optional feedback:", feedback if feedback is not None else "Use the local rubric; no provider response was obtained.")
print("Elapsed seconds since lab imports (includes optional work and pauses):", round(time.perf_counter() - core_started, 3))

# %% [markdown]
# ## Five final self-checks
#
# 1. For $x_0=1$, $\bar\alpha_t=0.64$ and $\epsilon=-0.5$, what noisy value and training target do we use?
# <!-- BEGIN ANSWER -->
# The noisy value is 0.8 × 1 + 0.6 × (−0.5) = 0.5. The target is the added noise −0.5.
# <!-- END ANSWER -->
#
# 2. Does training need to run every earlier reverse step for its random timestep?
# <!-- BEGIN ANSWER -->
# No. Direct noising constructs the selected training state. The reverse loop is for generation.
# <!-- END ANSWER -->
#
# 3. Does calling the trained denoiser at 40 selected DDIM levels update its weights?
# <!-- BEGIN ANSWER -->
# No. Sampling runs the fixed network repeatedly; the DDIM rule uses current and next schedule coefficients.
# <!-- END ANSWER -->
#
# 4. Why is a guidance sweep unsuitable for SD-Turbo?
# <!-- BEGIN ANSWER -->
# Its published usage disables classifier-free guidance. Use SD 1.5 stored evidence or the appropriate optional guided model.
# <!-- END ANSWER -->
#
# 5. What does a cosine score of 1 between synthetic vectors tell us about image quality?
# <!-- BEGIN ANSWER -->
# Only that the vectors point in the same direction. Synthetic vectors are not image evidence; CLIP similarity is also incomplete.
# <!-- END ANSWER -->
