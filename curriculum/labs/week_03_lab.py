# %% [markdown]
# # Week 3 Lab: Train, diagnose and evaluate a GAN
#
# **Module:** Generative AI (MSc in Artificial Intelligence) · **Time:** 2 hours · **Learning outcomes:** MIMLO 1, 3, 4
#
# You will implement a **DCGAN** in PyTorch, train it on MNIST, and then act like a researcher:
#
# 1. implement the **discriminator** and **non-saturating generator** losses;
# 2. watch samples evolve across epochs and read the training diagnostics;
# 3. explore the latent space with linear and **spherical** interpolation;
# 4. **evaluate** the generator: mode coverage with a digit classifier and a Fréchet distance you implement;
# 5. inspect **mode collapse** on a 2-D toy problem and compare a **WGAN-GP** configuration.
#
# | Part | Topic | Suggested time |
# |---|---|---|
# | 0–1 | Setup, data, models | 15 min |
# | 2 | Losses and training | 30 min |
# | 3 | Latent space | 10 min |
# | 4 | Evaluation (coverage + Fréchet distance) | 30 min |
# | 5 | Mode collapse and WGAN-GP | 25 min |
# | 6 | Reflection | 10 min |

# %% [markdown]
# **Core route and longer archive.** The default CPU route trains a real GAN for four epochs on 25,600 MNIST images (800 updates), a classifier on 8,000 images, and two ring configurations of 1,000 steps. The teaching notes give the measured waiting time on the course laptop; the last cell prints your own. Download time depends on the connection. Smoke mode is a smaller engineering check, not the teaching measurement.
#
# `RUN_STRETCH = True` chooses eight epochs, the full training set, 2,000 evaluation samples and 5,000 ring steps. This is optional longer training. The lecture's stored images came from that longer archived experiment; the shorter core is expected to look different. Discuss poor samples honestly rather than claiming the core reproduces the archive.
#
# **Common errors:** If CPU tensors and model tensors use different devices, use the shared `DEVICE`. Do not create a second device selector or reset the shared seed. A download error needs a working connection or an existing MNIST cache.
#
# **Read → predict → run → change → check:** read each part's purpose, predict its output, run once, change one setting and explain the check. Keep a note of what stayed fixed.

# %% [markdown]
# ## Before coding: two networks, two training roles
#
# A **generator** turns random codes into images. A **discriminator** learns to
# distinguish those images from real data. They improve through alternating
# updates; an individual code has no corresponding target training digit.
#
# ![A GAN learns from a discriminator](fig:beginner_overview)
#
# Follow the noise path into `G` and the real-image path directly into `D`.
# `d_loss_fn` trains `D` to distinguish sources. `g_loss_fn` sends feedback
# through `D` to `G`. After training, generation uses `G` alone.
#
# ![Check predicted class coverage](fig:beginner_lab)
# This flow covers one evaluation route only. Interpolation, feature distance and the ring are separate experiments.
#
# | Independent experiment | Code to find | First observation |
# |---|---|---|
# | Prepare and build | `Generator`, `Discriminator` | Noise has 64 values; each image is 28 by 28 pixels. |
# | Alternate training | `d_loss_fn`, `g_loss_fn`, `log` | The same fakes have different target roles in the two updates. |
# | Explore codes | `lerp`, `slerp` | Compare images and code norms along each path. |
# | Check class coverage | `DigitCNN`, `counts` | Classifier errors affect the estimated histogram. |
# | Compare features | `feats`, `frechet_distance` | This score uses the lab's classifier features. |
# | Test collapse | `train_toy`, `modes_covered` | Count coverage as well as inspecting the scatter plots. |
#
# **Pause and predict:** should the generator receive gradients when the
# discriminator learns to reject its fakes? Trace each update separately.

# %% [markdown]
# **Purpose:** Install GAN and matrix libraries.
# **Why now:** The lab uses PyTorch and SciPy.
# **Expected observation:** Installation or an explicit environment error.
#

# %% tags=["colab-install"]
%pip install -q torch torchvision scipy matplotlib

# %% [markdown]
# **Purpose:** Set device and random state.
# **Why now:** All later tensors and sampling share this setup.
# **Expected observation:** Printed CPU/CUDA and training epoch count.
#

# %%
import copy
import os
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Shared setup supplies DEVICE, SMOKE, SEED and the optional API switches.
# CPU is the default; enable the shared USE_GPU switch only if you choose.
RUN_STRETCH = False
EPOCHS = 1 if SMOKE else (8 if RUN_STRETCH else 4)
GAN_IMAGES = 512 if SMOKE else (60000 if RUN_STRETCH else 25600)
CLASSIFIER_IMAGES = 1024 if SMOKE else (60000 if RUN_STRETCH else 8000)
EVAL_N = 128 if SMOKE else (2000 if RUN_STRETCH else 512)
TOY_STEPS = 20 if SMOKE else (5000 if RUN_STRETCH else 1000)
Z_DIM = 64
core_started = time.perf_counter()
print("device:", DEVICE, "| epochs:", EPOCHS, "| GAN images:", GAN_IMAGES)
print("mode:", "optional longer training" if RUN_STRETCH else "bounded CPU core")

# %% [markdown]
# ## Part 1 · Data and models
#
# **Core:** Predict the G and D batch shapes before running.
# **Change one thing:** Change only the plotted batch size; explain which dimensions change.
# **Optional stretch:** Trace each convolution size on paper.
# **Common errors:** B is batch count, not image width; logits need matching target shape.
# **Check:** explain the printed result and the relevant check_ function.
# GAN generators usually end with `tanh`, so we scale pixels to **[−1, 1]**.
#
# **Read:** the generator creates images and the discriminator scores them.
# A **parameter** is a stored number changed during training; a **gradient**
# tells the optimiser how changing it affects the current loss.
# **Run:** implement the two losses before starting the training loop.
# **Change:** inspect snapshots made from the same `z_fixed` after each epoch.
# **Check:** compare sample quality and class coverage together. Lower GAN
# loss alone does not mean better pictures.

# %% [markdown]
# **Purpose:** Inspect the real training images.
# **Why now:** G’s tanh output uses the same pixel range.
# **Expected observation:** Normalised digits lie in [−1, 1].
#

# %%
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
train_set = Subset(train_set, range(GAN_IMAGES))
train_dl = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)


def show(imgs, n=16, title=None):
    imgs = imgs[:n].detach().cpu()
    plt.figure(figsize=(n * 0.6, 0.9))
    plt.imshow(np.hstack(((imgs[:, 0] + 1) / 2).numpy()), cmap="gray_r", vmin=0, vmax=1)
    plt.axis("off")
    if title:
        plt.title(title, loc="left")
    plt.show()


show(next(iter(train_dl))[0], title="Real MNIST digits")

# %% [markdown]
# The DCGAN generator upsamples a noise vector with **transposed convolutions**; the discriminator is a small CNN that outputs one **logit** (real vs. fake).
#
# **Read the shapes:** G maps `[B, 64]` codes to `[B, 1, 28, 28]` images.
# D maps those images to `[B]` scores. A logit is a score, not a probability;
# the diagnostics apply sigmoid when showing D(real) and D(fake).

# %% [markdown]
# **Purpose:** Build G and D.
# **Why now:** Their inputs and outputs define the two roles.
# **Expected observation:** Parameter counts and matching image/logit batch shapes.
#

# %%
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 128, 7, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),  # 7x7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),     # 14x14
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh())                               # 28x28

    def forward(self, z):
        return self.net(z.view(-1, Z_DIM, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 7, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):  # DCGAN initialisation
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
G.apply(init_weights)
D.apply(init_weights)
print(f"G: {sum(p.numel() for p in G.parameters()):,} parameters | D: {sum(p.numel() for p in D.parameters()):,} parameters")
with torch.no_grad():
    shape_fake = G(torch.randn(4, Z_DIM, device=DEVICE))
    print("codes [4,64] -> images", tuple(shape_fake.shape), "-> logits", tuple(D(shape_fake).shape))

# %% [markdown]
# ## Part 2 · Losses and training
#
# **Core:** Predict whether D-loss backward reaches G.
# **Change one thing:** Change only one learning rate and compare fixed-code outputs.
# **Optional stretch:** Enable longer training after recording the core observations.
# **Common errors:** Do not detach the generator loss; recompute D after stepping its weights.
# **Check:** explain the printed result and the relevant check_ function.
#
# `bce = nn.BCEWithLogitsLoss()` expects logits and targets (1 = real, 0 = fake).
#
# **TODO 1 (discriminator loss):** real images should be classified as 1 and generated images as 0. Remember to **detach** the fakes so this step does not update G.
#
# **TODO 2 (generator loss, non-saturating):** the generator wants D to call its fakes **real**.
#
# `detach` keeps image values while disconnecting their gradient path to G
# for the discriminator update. The generator update needs the original
# connected fakes so its own weights can receive feedback through D.
#
# **Trace one batch:** first score real and detached fake images, then update
# D. Next score the connected fakes with the updated D, then update G.
# The generator uses target 1 because it wants its images to pass as real.
# `zero_grad` clears old gradients, `backward` computes new ones, and `step`
# changes only the parameters owned by that optimiser.

# %% [markdown]
# **Purpose:** Implement the two losses.
# **Why now:** The same fake has different targets in each update.
# **Expected observation:** D loss separates sources; G loss keeps its gradient connection.
#
# ![One alternating GAN update](fig:beginner_mechanism)
#

# %%
bce = nn.BCEWithLogitsLoss()


def d_loss_fn(D, x_real, x_fake):
    ones = torch.ones(x_real.size(0), device=DEVICE)
    zeros = torch.zeros(x_fake.size(0), device=DEVICE)
    ### BEGIN SOLUTION
    return bce(D(x_real), ones) + bce(D(x_fake.detach()), zeros)
    ### END SOLUTION
    ### STUB return torch.zeros((), device=DEVICE, requires_grad=True)  # TODO 1: sum both source losses


def g_loss_fn(D, x_fake):
    ones = torch.ones(x_fake.size(0), device=DEVICE)
    ### BEGIN SOLUTION
    return bce(D(x_fake), ones)
    ### END SOLUTION
    ### STUB return torch.zeros((), device=DEVICE, requires_grad=True)  # TODO 2: target one; keep fake linked

# %% [markdown]
# **Purpose:** Check both loss targets and gradient routes.
# **Why now:** A plausible scalar can still train the wrong network.
# **Expected output:** Two ✅ messages, or a hint identifying the incorrect route.

# %%
class _ProbeD(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    def forward(self, x):
        return self.scale * x.flatten(1).mean(1)


def check_d_loss():
    try:
        probe = _ProbeD()
        real = torch.ones(2, 1, 2, 2, device=DEVICE)
        fake = torch.full_like(real, 0.5, requires_grad=True)
        value = d_loss_fn(probe, real, fake)
        expected = bce(probe(real), torch.ones(2, device=DEVICE)) + bce(probe(fake.detach()), torch.zeros(2, device=DEVICE))
        route = torch.autograd.grad(value, fake, allow_unused=True)[0]
        return check(torch.allclose(value, expected) and route is None,
                     "D loss: source labels correct; G route detached.",
                     "Use real target one, fake target zero and fake.detach().")
    except (RuntimeError, TypeError, AttributeError) as exc:
        return check(False, "D loss checked.", "Check scalar loss, target shapes and detach: " + str(exc))


def check_g_loss():
    try:
        probe = _ProbeD()
        fake = torch.full((2, 1, 2, 2), 0.5, device=DEVICE, requires_grad=True)
        value = g_loss_fn(probe, fake)
        expected = bce(probe(fake), torch.ones(2, device=DEVICE))
        route = torch.autograd.grad(value, fake, allow_unused=True)[0]
        linked = route is not None and bool(route.abs().sum() > 0)
        return check(torch.allclose(value, expected) and linked,
                     "G loss: target one; gradient route connected.",
                     "Pass the original fake to D and use target one.")
    except (RuntimeError, TypeError, AttributeError) as exc:
        return check(False, "G loss checked.", "Check scalar loss and the connected fake: " + str(exc))


d_ready = check_d_loss()
g_ready = check_g_loss()
losses_ready = d_ready and g_ready

# %% [markdown]
# **Purpose:** Alternate network updates.
# **Why now:** Fixed codes separate training progress from new noise.
# **Expected observation:** Epoch snapshots and recorded losses; assess actual images.
#
# ![Where the GAN training feedback comes from](fig:beginner_training)
#

# %%
if not losses_ready:
    raise ValueError("Fix the two loss checks before training.")
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
z_fixed = torch.randn(16, Z_DIM, device=DEVICE)
log = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": []}
snapshots = {}
G_after_epoch1 = None

for epoch in range(1, EPOCHS + 1):
    for step, (x, _) in enumerate(train_dl):
        x = x.to(DEVICE)
        fake = G(torch.randn(x.size(0), Z_DIM, device=DEVICE))
        # 1) discriminator step
        loss_D = d_loss_fn(D, x, fake)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        # 2) generator step
        # D was just updated; the original fake tensor is still connected to G.
        # Backward passes through D, but opt_G steps only G's parameters.
        loss_G = g_loss_fn(D, fake)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        if step % 20 == 0:
            with torch.no_grad():
                log["loss_D"].append(loss_D.item()); log["loss_G"].append(loss_G.item())
                log["D_real"].append(torch.sigmoid(D(x)).mean().item())
                log["D_fake"].append(torch.sigmoid(D(fake)).mean().item())
    G.eval()
    with torch.no_grad():
        snapshots[epoch] = G(z_fixed).cpu()
    G.train()
    if epoch == 1:
        G_after_epoch1 = copy.deepcopy(G).eval()
    print(f"epoch {epoch}: loss_D {loss_D.item():.3f}  loss_G {loss_G.item():.3f}")

for e, imgs in snapshots.items():
    show(imgs, title=f"after epoch {e}")

# %% [markdown]
# **Purpose:** Read diagnostic curves.
# **Why now:** The opponent changes each objective.
# **Expected observation:** Fluctuating losses and source scores; no single curve proves quality.
#

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
logged_steps = np.arange(len(log["loss_D"])) * 20  # a value was logged every 20 updates within each epoch
ax[0].plot(logged_steps, log["loss_D"], label="D loss"); ax[0].plot(logged_steps, log["loss_G"], label="G loss")
ax[0].legend(); ax[0].set_title("losses"); ax[0].set_xlabel("logged update")
ax[1].plot(logged_steps, log["D_real"], label="D(real)"); ax[1].plot(logged_steps, log["D_fake"], label="D(fake)")
ax[1].set_xlabel("logged update")
ax[1].axhline(0.5, ls="--", color="gray"); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("discriminator outputs")
plt.show()

# %% [markdown]
# ✍️ **Question 1.** Describe your loss and D-output curves. Why do GAN losses not decrease steadily like a classifier's? What pattern would indicate that the discriminator has "won"?
#
# <!-- BEGIN ANSWER -->
# The networks keep changing each other's task, so their losses may fluctuate. State what your curves show; there is no required range for D(real) or D(fake). If D(real) stays near 1 and D(fake) near 0, D is separating the two sets easily. Its loss is then low and the generator's non-saturating loss can be large. This alone does not prove that G's gradients vanish: the non-saturating loss was chosen to avoid that simple failure. Inspect whether the generated images actually improve and whether they cover several digit classes. Use the fixed-noise snapshots and evaluation metrics alongside the losses.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 3 · Exploring the latent space
#
# **Core:** Predict which midpoint has smaller norm for perpendicular equal-length codes.
# **Change one thing:** Move the interpolation slider while keeping endpoints fixed.
# **Optional stretch:** Compare endpoint norms across several pairs.
# **Common errors:** Move t to the model device if mixing devices; near-collinear codes need a fallback.
# **Check:** explain the printed result and the relevant check_ function.
#
# In high dimensions, samples from $\mathcal{N}(0, I)$ lie close to a sphere of radius about $\sqrt{d}$. A straight line between two samples cuts through the lower-norm interior; **spherical linear interpolation (slerp)** follows the sphere instead:
#
# $$\mathrm{slerp}(a, b; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,a + \frac{\sin(t\Omega)}{\sin\Omega}\,b, \qquad \Omega = \arccos\left(\frac{a \cdot b}{\|a\|\,\|b\|}\right)$$
#
# **TODO 3:** implement `slerp`. **Hint:** normalise directions to find the angle, then use the two sine weights.
#
# `a` and `b` are two noise codes; `t` is how far we move between them.
# `||a||` is the code's length, `a·b` compares its direction with `b`, and
# `Omega` is the angle between directions. Sine supplies curved-path weights.
# This demonstration draws fresh random, nonzero codes. Zero-length or near-collinear codes
# fall back to a straight line. Opposite directions need an extra convention for a unique spherical path.

# %% [markdown]
# **Purpose:** Compare interpolation paths.
# **Why now:** Latent geometry can affect intermediate codes.
# **Expected observation:** Images and midpoint norms; no memorisation guarantee.
#
# ![Use the trained generator on new noise](fig:beginner_inference)
#

# %%
def lerp(a, b, t):
    return (1 - t) * a + t * b


def slerp(a, b, t):
    ### BEGIN SOLUTION
    if a.norm() < 1e-8 or b.norm() < 1e-8:
        return lerp(a, b, t)
    cosine = torch.clamp(torch.dot(a / a.norm(), b / b.norm()), -1.0, 1.0)
    omega = torch.arccos(cosine)
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return lerp(a, b, t)
    return torch.sin((1 - t) * omega) / so * a + torch.sin(t * omega) / so * b
    ### END SOLUTION
    ### STUB return lerp(a, b, t)  # TODO 3: replace with spherical interpolation


def check_slerp():
    left = torch.tensor([1.0, 0.0], device=DEVICE)
    right = torch.tensor([0.0, 1.0], device=DEVICE)
    midpoint = slerp(left, right, 0.5)
    good = torch.allclose(slerp(left, right, 0.0), left, atol=1e-6) and torch.allclose(slerp(left, right, 1.0), right, atol=1e-6)
    good = good and bool(torch.isfinite(midpoint).all()) and torch.allclose(midpoint.norm(), torch.tensor(1.0, device=DEVICE), atol=1e-5)
    return check(good, "Slerp endpoints and equal-radius midpoint checked.",
                 "Use sine weights; a straight-line midpoint has smaller norm here.")


check_slerp()
G.eval()
a, b = torch.randn(Z_DIM, device=DEVICE), torch.randn(Z_DIM, device=DEVICE)
ts = torch.linspace(0, 1, 10)
with torch.no_grad():
    show(G(torch.stack([lerp(a, b, t) for t in ts])), n=10, title="linear interpolation")
    show(G(torch.stack([slerp(a, b, t) for t in ts])), n=10, title="spherical interpolation")
print("norm of midpoint  lerp: %.2f   slerp: %.2f   (typical sample: %.2f)" % (
    lerp(a, b, 0.5).norm(), slerp(a, b, torch.tensor(0.5)).norm(), a.norm()))

# %% [markdown]
# **Purpose:** Explore one interpolation setting without retraining.
# **Why now:** Holding codes fixed isolates t.
# **Expected output:** A slider and one generated image; no optimiser update. This cell is skipped by automated execution.

# %% tags=["skip-test"]
import ipywidgets as widgets
from IPython.display import display


def show_interpolation(t):
    with torch.no_grad():
        code = slerp(a, b, t).unsqueeze(0)
        show(G(code), n=1, title=f"spherical path t={t:.2f}")


display(widgets.interactive(show_interpolation,
                           t=widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="Path t")))

# %% [markdown]
# ## Part 4 · Evaluation
#
# **Core:** Predict whether sharpness alone guarantees all classes.
# **Change one thing:** Change only EVAL_N and discuss sampling uncertainty.
# **Optional stretch:** Compare two seeds using the same trained classifier.
# **Common errors:** Use features, not labels, for distance; custom DigitCNN distance is not standard FID.
# **Check:** explain the printed result and the relevant check_ function.
#
# This GAN does not provide a tractable likelihood calculation, so we evaluate **samples**. Standard FID uses Inception features from ImageNet photos; here we train a **small digit classifier** and use it in two ways:
#
# * **mode coverage:** the class histogram of generated digits (a collapsed GAN draws only a few classes);
# * **custom feature Fréchet distance** between the classifier's features for real and generated digits.
#
# This uses the FID formula with a different feature network. Its values are
# not directly comparable with standard Inception FID. Keep the classifier,
# preprocessing, and sample count fixed within this lab's comparisons.
# A **feature vector** is a list of numbers describing an image. The mean
# describes the centre of many such vectors; the covariance describes their
# spread. The distance compares these two summaries, so it can miss individual
# bad images. Also check the classifier's test accuracy before trusting its
# class histogram: a weak classifier can make generator coverage look misleading.

# %% [markdown]
# **Purpose:** Train an independent digit classifier.
# **Why now:** Its predictions and features support later evaluation.
# **Expected observation:** Held-out accuracy; inspect reliability before scoring G.
#

# %%
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.features(x))


clf = DigitCNN().to(DEVICE)
opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
full_train = datasets.MNIST("data", train=True, download=True, transform=tf)
clf_dl = DataLoader(Subset(full_train, range(CLASSIFIER_IMAGES)), batch_size=256, shuffle=True)
for x, y in clf_dl:
    x, y = x.to(DEVICE), y.to(DEVICE)
    loss = F.cross_entropy(clf(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
clf.eval()
x_te, y_te = next(iter(DataLoader(test_set, batch_size=EVAL_N)))
with torch.no_grad():
    acc = (clf(x_te.to(DEVICE)).argmax(1).cpu() == y_te).float().mean().item()
print(f"digit classifier test accuracy: {acc:.3f}")

# %% [markdown]
# **TODO 4:** generate `EVAL_N` digits, classify them and plot the class histogram. **Hint:** apply `argmax(1)` to ten logits per image. Count classes receiving at least 5% of samples.

# %% [markdown]
# **Purpose:** Count generated classes.
# **Why now:** Coverage differs from sharpness.
# **Expected observation:** Histogram and classes above the stated 5% threshold.
#

# %%
coverage_codes = torch.randn(EVAL_N, Z_DIM, device=DEVICE)
with torch.no_grad():
    coverage_images = G(coverage_codes)
    gen = coverage_images
    coverage_logits = clf(coverage_images)
    ### BEGIN SOLUTION
    pred = coverage_logits.argmax(1).cpu().numpy()
    ### END SOLUTION
    ### STUB pred = np.zeros(EVAL_N, dtype=int)  # TODO 4: predicted class of each generated digit
def check_predictions():
    expected = coverage_logits.argmax(1).cpu().numpy()
    good = pred.shape == (EVAL_N,) and np.issubdtype(pred.dtype, np.integer) and np.array_equal(pred, expected)
    return check(good, "One classifier label per generated image.",
                 "Use argmax over the ten class logits, not over the batch.")


check_predictions()
counts = np.bincount(pred, minlength=10)
print("coverage images:", tuple(coverage_images.shape), "labels:", pred.shape, "counts:", counts.shape)
plt.bar(range(10), counts / counts.sum())
plt.axhline(0.1, ls="--", color="gray")
plt.xticks(range(10)); plt.xlabel("predicted digit"); plt.ylabel("fraction of samples"); plt.title("Mode coverage of the generator")
plt.show()
print("classes with at least 5% of samples:", int((counts / counts.sum() >= 0.05).sum()), "of 10")

# %% [markdown]
# **TODO 5:** implement the Fréchet distance between two sets of feature vectors:
#
# $$d^2 = \|\mu_1 - \mu_2\|^2 + \mathrm{Tr}\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2}\right)$$
#
# Use np.cov(feats, rowvar=False) and scipy.linalg.sqrtm. The teaching implementation drops imaginary round-off; inspect large imaginary values or nonfinite results before interpreting a distance.
# Here `mu` is a mean vector, `Sigma` is a covariance matrix, `Tr` adds a
# matrix's diagonal entries, and the square-root term compares the spreads.
# Each input has shape `[number of images, 128 features]`, not `[images, 10 labels]`.

# %% [markdown]
# **Purpose:** Compare feature statistics.
# **Why now:** A custom metric makes the network and sample choices explicit.
# **Expected observation:** Four distances; rankings are empirical, not guaranteed.
#

# %%
from scipy import linalg


def frechet_distance(f1, f2):
    # Inputs are feature vectors, not raw pixels or predicted class labels.
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    ### BEGIN SOLUTION
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        imaginary_size = np.max(np.abs(covmean.imag))
        reference_size = max(1.0, float(np.max(np.abs(covmean.real))))
        if imaginary_size > 1e-5 * reference_size:
            raise ValueError("Large imaginary covariance root: inspect finite features and use more samples.")
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))
    ### END SOLUTION
    ### STUB return 0.0  # TODO 5: mean difference plus covariance trace


def check_frechet_distance():
    points = np.zeros((12, 2))
    shifted = points + np.array([2.0, 3.0])
    same = frechet_distance(points, points)
    moved = frechet_distance(points, shifted)
    good = np.isfinite(same) and np.isfinite(moved) and abs(same) < 1e-6 and abs(moved - 13.0) < 1e-6
    # Both means are zero; only the covariance term distinguishes these clouds.
    spread = np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])
    spread_distance = frechet_distance(spread, 2 * spread)
    good = good and np.isfinite(spread_distance) and abs(spread_distance - 4 / 3) < 1e-6
    return check(good, "Feature-distance identity, mean-shift and covariance checks passed.",
                 "Keep both terms: shifting the mean adds 13; doubling this zero-mean cloud adds 4/3.")


check_frechet_distance()


@torch.no_grad()
def feats(x):
    # Small batches keep CPU memory bounded while preserving the feature contract.
    return np.concatenate([clf.features(chunk.to(DEVICE)).cpu().numpy() for chunk in x.split(64)])


x_tr, _ = next(iter(DataLoader(full_train, batch_size=EVAL_N, shuffle=True)))
real_a, real_b = feats(x_te), feats(x_tr)
with torch.no_grad():
    fake_final = feats(G(torch.randn(EVAL_N, Z_DIM, device=DEVICE)))
    fake_ep1 = feats(G_after_epoch1(torch.randn(EVAL_N, Z_DIM, device=DEVICE)))
    noise = feats(torch.rand(EVAL_N, 1, 28, 28) * 2 - 1)
for name, f in [("real test vs real train (floor)", real_b), ("GAN after epoch 1", fake_ep1),
                ("GAN after training", fake_final), ("uniform noise", noise)]:
    print(f"{name:32s} Fréchet distance = {frechet_distance(real_a, f):9.2f}")

# %% [markdown]
# ✍️ **Question 2.** Interpret the four Fréchet distances. Why is the "real vs real" value not zero? Why might this classifier-based score disagree with the standard Inception-based FID, and why must you use the same feature network and sample size when comparing models?
#
# <!-- BEGIN ANSWER -->
# Two finite real-image samples can have different means and covariances, so real-vs-real is a reference rather than a guaranteed zero or universal floor. Describe the ordering you actually observe: a short core need not improve monotonically or reproduce the archive. This custom distance uses DigitCNN features, whereas standard FID uses Inception features. Keep extractor, preprocessing and sample count fixed within a comparison, report seeds and classifier accuracy, and inspect coverage and actual images as well.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 5 · Mode collapse on a 2-D toy problem
#
# **Core:** Predict which configuration reaches more clusters after 1,000 generator steps.
# **Change one thing:** Change only TOY_STEPS before changing losses or rates.
# **Optional stretch:** Run the archived 5,000-step configurations.
# **Common errors:** The critic is not a probability; five critic steps add work; no coverage outcome is guaranteed.
# **Check:** explain the printed result and the relevant check_ function.
#
# The target distribution is 8 small Gaussian clusters arranged on a ring. We compare a standard GAN with a **Wasserstein GAN with gradient penalty (WGAN-GP)**.
#
# **TODO 6:** implement the gradient penalty. **Hint:** differentiate the summed scores with respect to `x_hat`, then measure each input-gradient norm. For random interpolates $\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}$, penalise $(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2$.
#
# The two configurations also differ in learning rates and one versus five critic updates. Compare the whole configurations; a single run cannot isolate the effect of the loss.
# On this toy problem, each example is only two coordinates. WGAN's **critic**
# gives an unrestricted score rather than a real/fake probability. The penalty
# measures how strongly that score changes as an input point moves, and favours
# a gradient length near 1 at the sampled points. It is a training aid, not a
# promise that every cluster will be covered.

# %% [markdown]
# **Purpose:** Compare two toy GAN configurations.
# **Why now:** Eight visible clusters make missing modes easier to inspect.
# **Expected observation:** Scatter plots and mode counts; WGAN-GP also costs extra critic updates.
#

# %%
def ring(n, k=8, r=2.0, s=0.05):
    c = np.random.randint(0, k, n)
    ang = 2 * np.pi * c / k
    return torch.tensor(np.c_[r * np.cos(ang), r * np.sin(ang)] + s * np.random.randn(n, 2), dtype=torch.float32, device=DEVICE)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)).to(DEVICE)


def gradient_penalty(D, real, fake):
    # This regularises the critic's input-gradient norm on sampled interpolates.
    # It encourages smooth feedback; it does not enforce a global guarantee.
    eps = torch.rand(real.size(0), 1, device=DEVICE)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    ### BEGIN SOLUTION
    grad = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)[0]
    return ((grad.norm(dim=1) - 1) ** 2).mean()
    ### END SOLUTION
    ### STUB return torch.tensor(0.0, device=DEVICE)  # TODO 6


def check_gradient_penalty():
    real = torch.randn(4, 2, device=DEVICE)
    fake = torch.randn(4, 2, device=DEVICE)
    unit = gradient_penalty(lambda x: x[:, :1], real, fake)
    doubled = gradient_penalty(lambda x: 2 * x[:, :1], real, fake)
    good = torch.isfinite(unit) and torch.isfinite(doubled) and abs(unit.item()) < 1e-6 and abs(doubled.item() - 1.0) < 1e-6
    class TrainableSlope(nn.Module):
        def __init__(self):
            super().__init__()
            self.slope = nn.Parameter(torch.tensor(2.0, device=DEVICE))
        def forward(self, x):
            return self.slope * x[:, :1]
    try:
        # The extra fixture must not consume random draws used by the toy experiment.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE == "cuda" else []):
            critic = TrainableSlope()
            penalty = gradient_penalty(critic, real, fake)
            route = torch.autograd.grad(penalty, critic.slope, allow_unused=True)[0] if penalty.requires_grad else None
        good = bool(good and route is not None and torch.allclose(route, torch.tensor(2.0, device=DEVICE), atol=1e-6))
    except (RuntimeError, TypeError, AttributeError):
        good = False
    return check(bool(good), "Known slope penalties and the critic-weight gradient are correct.",
                 "Use create_graph=True: at slope two, penalty is one and its slope gradient must be two.")


check_gradient_penalty()


def modes_covered(pts, k=8, r=2.0, tol=0.3):
    centres = np.c_[r * np.cos(2 * np.pi * np.arange(k) / k), r * np.sin(2 * np.pi * np.arange(k) / k)]
    d = np.linalg.norm(pts[:, None] - centres[None], axis=2)
    counts = np.bincount(d.argmin(1)[d.min(1) < tol], minlength=k)
    return int((counts > 0.02 * len(pts)).sum())


def train_toy(kind, steps):
    torch.manual_seed(1); np.random.seed(1)
    g2, d2 = mlp(2, 2), mlp(2, 1)
    wgan = kind == "wgan-gp"
    og = torch.optim.Adam(g2.parameters(), lr=1e-4 if wgan else 1e-3, betas=(0.5, 0.9))
    od = torch.optim.Adam(d2.parameters(), lr=1e-4 if wgan else 1e-3, betas=(0.5, 0.9))
    for _ in range(steps):
        for _ in range(5 if wgan else 1):
            x, f = ring(256), g2(torch.randn(256, 2, device=DEVICE)).detach()
            if wgan:
                ld = d2(f).mean() - d2(x).mean() + 10 * gradient_penalty(d2, x, f)
            else:
                ld = bce(d2(x).view(-1), torch.ones(256, device=DEVICE)) + bce(d2(f).view(-1), torch.zeros(256, device=DEVICE))
            od.zero_grad(); ld.backward(); od.step()
        f = g2(torch.randn(256, 2, device=DEVICE))
        lg = -d2(f).mean() if wgan else bce(d2(f).view(-1), torch.ones(256, device=DEVICE))
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        return g2(torch.randn(1500, 2, device=DEVICE)).cpu().numpy()


steps = TOY_STEPS
print("toy training steps:", steps, "| WGAN-GP uses five critic updates per step")
ring_results = {}
real_pts = ring(1500).cpu().numpy()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, kind in zip(axes, ["standard", "wgan-gp"]):
    pts = train_toy(kind, steps)
    ax.scatter(*real_pts.T, s=3, alpha=0.3, color="gray", label="real")
    ax.scatter(*pts.T, s=3, alpha=0.5, label="generated")
    ring_results[kind] = modes_covered(pts)
    ax.set_title(f"{kind}: {ring_results[kind]}/8 modes covered"); ax.set_aspect("equal"); ax.legend()
plt.show()

# %% [markdown]
# ✍️ **Question 3.** Compare your two counts with the archived 5,000-step run on lecture slide 23. Why can a standard GAN reach clusters early yet later hop or collapse? Why can WGAN-GP, with a smaller learning rate and five critic updates, need more steps before it spreads? Name one cost of WGAN-GP.
#
# <!-- BEGIN ANSWER -->
# A standard generator is rewarded for samples the current discriminator accepts. With a larger learning rate it can spread quickly, but as D adapts, G can jump between clusters or concentrate on a few (mode hopping/collapse); the archive shows 8/8 at step 500 but 7/8 at step 5,000. The WGAN-GP configuration uses a ten-times smaller learning rate and a critic that must first learn useful slopes, so after a short budget it may still sit between clusters; in the archive it reached 8/8 only by step 5,000. The gradient penalty encourages an input-gradient norm near 1 on sampled interpolates; it does not guarantee a globally 1-Lipschitz critic or complete coverage. One short run cannot rank the methods: compare budgets and seeds. Costs: five critic updates per generator update and differentiating through the penalty make each step slower; λ and learning rates still matter. Toy results do not guarantee the same behaviour on images.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 6 · Reflection: responsible use
#
# **Core:** Predict why a detector can misclassify real media.
# **Change one thing:** Revise one disclosure or provenance action for a synthetic example.
# **Optional stretch:** Consult the official Article 50 text for provider and deployer roles.
# **Common errors:** A missing credential does not establish that an image is generated.
# **Check:** explain the printed result and the relevant check_ function.
#
# ✍️ **Question 4.** GAN-based face generators made realistic fake profile photos trivial to produce. Propose two technical and one organisational measure a social-media platform could use, and one limitation of each.
#
# <!-- BEGIN ANSWER -->
# Technical: (1) detectors for synthetic images, with the limitation that they often fail on new generators (arms race) and produce false positives; (2) provenance: require or display C2PA content credentials and check invisible watermarks from cooperating generators, with the limitation that credentials can be stripped and open models may not watermark. Organisational: a policy requiring disclosure of AI-generated profile images, backed by reporting and review (EU Article 50 separates provider marking and deployer disclosure, with role-specific exceptions), with the limitation that enforcement depends on detection and human review capacity. Good answers also mention user education and rate limits on account creation.
# <!-- END ANSWER -->
#
# | Experiment | Setting | Metric | Result | Interpretation |
# |---|---|---|---|---|
# | DCGAN | actual core epochs and sample count | coverage (classes ≥ 5%) | | |
# | DCGAN | epoch 1 vs final | Fréchet distance | | |
# | Ring | standard vs WGAN-GP | modes covered | | |
#
# **Before next week:** read Foster (2023) chapter 8 (diffusion models) and watch the 3Blue1Brown / Welch Labs video on how AI images work.

# %% [markdown]
# **Purpose:** Critique one synthetic claim with local evidence, optionally using Gemini.
# **Why now:** Explanation checking is distinct from training or measuring a model.
# **Expected output:** A local checklist, or a critique from the optional selected Gemini or OpenAI API provider.
#
# The core needs no API key. The injected setup defaults USE_GEMINI and USE_OPENAI to False. Enable exactly one to opt in; enabling both returns the local fallback with guidance. Keys come from Colab Secrets or the provider's environment variable and are never printed. Send no personal data. The shared helper bounds calls and retries HTTP 429 with backoff. Gemini free-tier availability and terms vary by region; EEA/UK/Swiss API-client distribution has paid-service conditions. Keep the local route if the selected API is unavailable.

# %%
synthetic_claim = "Sharp digits and low GAN losses prove that every digit class is covered."
local_critique = "Check held-out classifier accuracy, class counts and samples. Loss and sharpness do not prove coverage."
critique = ask_model("Critique this synthetic teaching claim in two sentences: " + synthetic_claim,
                      max_output_tokens=120)
print(critique if critique else local_critique)

# %% [markdown]
# **Purpose:** Record the results of this execution.
# **Why now:** Exported metrics need mode, settings and elapsed-time provenance.
# **Expected output:** Settings and measured elapsed seconds; JSON is written only when GENAI_RESULTS_PATH is supplied by the teacher.

# %%
elapsed_seconds = time.perf_counter() - core_started
run_results = {
    "dcgan": {"epochs": EPOCHS, "final_loss_D": float(loss_D.item()), "final_loss_G": float(loss_G.item()),
              "final_D_real": float(log["D_real"][-1]), "final_D_fake": float(log["D_fake"][-1])},
    "ring": {"vanilla_final_modes": ring_results["standard"], "wgan_final_modes": ring_results["wgan-gp"]},
    "run": {"seed": SEED, "device": str(DEVICE), "smoke": bool(SMOKE), "stretch": bool(RUN_STRETCH),
            "gan_images": GAN_IMAGES, "classifier_images": CLASSIFIER_IMAGES, "evaluation_images": EVAL_N,
            "toy_steps": TOY_STEPS, "elapsed_seconds": elapsed_seconds},
    "evaluation": {"classifier_accuracy": float(acc), "predicted_class_counts": counts.tolist()}
}
print("Measured code-path elapsed seconds:", round(elapsed_seconds, 2), "| mode:", run_results["run"])
results_target = os.environ.get("GENAI_RESULTS_PATH")
if results_target:
    results_path = Path(results_target)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(run_results, indent=2), encoding="utf-8")
    print("Results exported to:", results_path)

# %% [markdown]
# ## Exit self-checks
#
# 1. Why is fake target zero for D but one for G?
# 2. Which gradient connection does detach cut?
# 3. What shapes enter G and leave D for batch size four?
# 4. Why can custom feature distance and class coverage disagree?
# 5. Why must you report the actual core outcome rather than copy the archive?
#
# <!-- BEGIN ANSWER -->
# 1. D identifies the generated source; G aims to make D accept its sample.
# 2. The route back through the fake tensor into G for that operation; its values are unchanged.
# 3. G receives [4,64], returns [4,1,28,28], and D returns [4].
# 4. Feature summaries and predicted class counts measure different properties; both depend on a fallible classifier.
# 5. Budgets, seeds and environments affect observations. Stored longer-run images are evidence for that run only.
# <!-- END ANSWER -->

