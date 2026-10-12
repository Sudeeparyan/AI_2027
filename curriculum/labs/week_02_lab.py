# %% [markdown]
# # Week 2 lab: build, train and inspect a variational autoencoder
#
# **Generative AI, MSc · 2 hours · MIMLO 1, 3, 4**
#
# A digit is an image of 784 intensities. Can a model compress it into two
# numbers, rebuild it and draw a new digit without an input image?
# You will implement a real VAE, train it with gradients and inspect its limits.
#
# | Part | Core task | Suggested class time |
# |---|---|---|
# | 0 | Read images and predict shapes | 10 min |
# | 1 | Complete and check the five TODOs | 25 min |
# | 2 | Train and inspect both costs | 20 min |
# | 3 | Decode a grid and interpolate | 25 min |
# | 4 | Change beta; distinguish weighting from training | 15 min |
# | 5 | Score unusual inputs and discuss limitations | 15 min |
# | 6 | Record evidence and answer self-checks | 10 min |
#
# Class time includes discussion and coding; it is not a measured compute time.
# CPU is the default. The core trains small models on a bounded subset.
# Longer beta comparisons and a conditional VAE are behind `RUN_STRETCH = False`.
# Lecture figures come from archived, longer experiments; your core results can differ.

# %% [markdown]
# ## Before coding: reconstruction and generation are different routes
#
# A **latent code** is a short hidden numerical representation. Its coordinates
# need not mean named properties such as digit identity, slant or stroke width.
# Reconstruction starts with an image. Generation starts with a fresh prior code.
# A **prior** is the chosen code distribution before seeing an image.
#
# ![Train a VAE to rebuild digits](fig:beginner_overview)
#
# Follow pixels → encoder parameters → sampled code → decoder logits → loss → update.
# Original pixels also enter the reconstruction loss directly. Encoder parameters
# enter KL directly. These are separate loss inputs, even though the costs train together.
#
# ![Build, train, then explore the VAE](fig:beginner_lab)
#
# | Operation | Code | Shape or observation |
# |---|---|---|
# | Prepare | `train_dl`, `x_test` | Images `[B,1,28,28]` |
# | Encode | `VAE.encode` | Mean and log-variance `[B,d]` |
# | Sample | `VAE.reparameterise` | Code `[B,d]` |
# | Decode | `VAE.dec`, `VAE.decode` | Logits `[B,784]`; images `[B,28,28]` |
# | Train | `train`, `vae_loss` | Reconstruction and KL separately |
# | Evaluate | `recon_error` | One score per image |
#
# **Predict:** which operations can be skipped when generating a fresh prior sample?
# Record your prediction before running the generation cell.

# %% [markdown]
# ## Part 0 · Read the data contract
#
# **Read:** a batch groups examples; an epoch is one pass through the training subset.
# **Predict:** what does flattening change in `[8,1,28,28]`?
# **Core:** inspect eight images and verify their values and dimensions.
# **Change one thing:** display six images rather than twelve.
# **Stretch:** compare the image and flattened views using `reshape`.
# **Common errors:** an import error usually means the wrong kernel or an incomplete
# installation. A failed dataset download needs a working connection; rerun that cell.

# %% [markdown]
# **What:** install the image, maths and plotting libraries.
# **Why:** PyTorch trains the VAE; torchvision supplies public teaching datasets.
# **Expect:** installation succeeds or reports a package/environment issue.
# The builder supplies the dated pins and one shared setup cell after installation.

# %% tags=["colab-install"]
%pip install -q torch torchvision scikit-learn matplotlib scipy

# %% [markdown]
# **What:** load a bounded digit subset and inspect the first batch.
# **Why:** tensor shapes and intensity ranges define the network's input contract.
# **Expect:** images `[8,1,28,28]`, flattened pixels `[8,784]`, and values within `[0,1]`.
# Use the shared `DEVICE`, `SEED`, `SMOKE`, `check()` and `ask_model()` definitions.
# Training limits below are settings to measure, not a promise about runtime or quality.

# %%
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# This explicit switch keeps the longer experiments outside the default core.
RUN_STRETCH = False
TRAIN_LIMIT = 1024 if SMOKE else 12000
TEST_LIMIT = 256 if SMOKE else 1000
EPOCHS = 1 if SMOKE else 3
HIDDEN = 128
BATCH_SIZE = 128

tf = transforms.ToTensor()  # Convert each grayscale image to values between zero and one.
all_train = datasets.MNIST("data", train=True, download=True, transform=tf)
all_test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_set = Subset(all_train, range(TRAIN_LIMIT))
train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))
x_test, y_test = next(iter(DataLoader(Subset(all_test, range(TEST_LIMIT)),
                                     batch_size=TEST_LIMIT)))
print("Core settings:", {"device": DEVICE, "train_images": TRAIN_LIMIT,
                          "test_images": TEST_LIMIT, "epochs": EPOCHS, "hidden": HIDDEN})
print("Image batch:", tuple(x_test[:8].shape), "| flattened:", tuple(x_test[:8].flatten(1).shape))
check(tuple(x_test[:8].shape) == (8, 1, 28, 28), "Input shapes match", "Inspect batch, channel, row and column dimensions.")
check(bool((x_test >= 0).all() and (x_test <= 1).all()), "Pixels lie in [0,1]", "Use ToTensor for grayscale intensities.")
plt.figure(figsize=(8, 1.3))
plt.imshow(np.hstack(x_test[:12, 0].numpy()), cmap="gray_r")
plt.axis("off")
plt.title("Held-out MNIST digits")
plt.show()

# %% [markdown]
# ## Part 1 · Build the distribution and the loss
#
# **Read:** the encoder predicts **mean** `mu` and **log-variance** `logvar`.
# Standard deviation is `exp(0.5 * logvar)`, so `logvar=0` gives spread one.
# A **logit** is a raw score; sigmoid converts it to a probability.
# **Predict:** both encoder outputs are `[B,d]`; decoder logits are `[B,784]`.
# **Core:** implement TODOs 1–3, then use the checks before training.
# **Change one thing:** inspect `zdim=4` in the shape check.
# **Stretch:** explain why the KL term needs `mu` and `logvar`, rather than sampled `z`.
# **Common errors:** `None` outputs mean TODO 1 is unfinished. Reusing `mu` as the sample
# removes variation. Using variance where standard deviation is needed changes the Gaussian.
#
# **TODO 1 hint:** apply the two output heads to the same hidden representation `h`.
# **TODO 2 hint:** draw `torch.randn_like(std)`, then calculate centre + spread × noise.
#
# ![Reconstruction and prior generation share one decoder](fig:lecture_flow_14)
#
# The two sampled routes meet at the decoder; only reconstruction supplies an input image.
#
# ![Sample a latent code while training](fig:beginner_mechanism)

# %% [markdown]
# **What:** define the VAE and run checks for its encoder and sampling method.
# **Why:** small contract checks catch wrong outputs before a training loop hides the error.
# **Expect:** two green checks and printed `[8,2]` parameter shapes.
# The checks explain unfinished TODOs without requiring you to train first.

# %%
class VAE(nn.Module):
    def __init__(self, zdim=2, hidden=HIDDEN):
        super().__init__()
        self.zdim = zdim
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.mu_head = nn.Linear(hidden, zdim)
        self.logvar_head = nn.Linear(hidden, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Linear(hidden, 784))

    def encode(self, x):
        h = self.enc(x)
        ### BEGIN SOLUTION
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        ### END SOLUTION
        ### STUB mu, logvar = None, None  # TODO 1: apply both output heads to h.
        return mu, logvar

    def reparameterise(self, mu, logvar):
        ### BEGIN SOLUTION
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        ### END SOLUTION
        ### STUB return mu  # TODO 2: centre + positive spread times fresh noise.

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.dec(z), mu, logvar

    @torch.no_grad()
    def decode(self, z):
        # Display predicted intensities; this does not sample individual binary pixels.
        return torch.sigmoid(self.dec(z)).view(-1, 28, 28)


def check_encoder(model):
    try:
        mu, logvar = model.encode(x_test[:8].to(DEVICE))
        good = (isinstance(mu, torch.Tensor) and isinstance(logvar, torch.Tensor)
                and tuple(mu.shape) == (8, model.zdim) and mu.shape == logvar.shape)
        if good:
            print("mu:", tuple(mu.shape), "| logvar:", tuple(logvar.shape))
        return check(good, "TODO 1: both encoder heads have the right shape",
                     "Apply mu_head and logvar_head to h; each should return [B,zdim].")
    except Exception:
        return check(False, "", "TODO 1: inspect the two head calls and their tensor shapes.")


def check_sampling(model):
    try:
        mu = torch.zeros(2048, model.zdim, device=DEVICE)
        logvar = torch.zeros_like(mu)
        z1 = model.reparameterise(mu, logvar)
        z2 = model.reparameterise(mu, logvar)
        good = (z1.shape == mu.shape and not torch.equal(z1, z2)
                and 0.85 < float(z1.std()) < 1.15)
        # Additional checks preserve the random state reserved for real training.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE == "cuda" else []):
            shifted_mu = torch.full((8, model.zdim), 1.25, device=DEVICE)
            nonunit_logvar = torch.full_like(shifted_mu, math.log(4))
            torch.manual_seed(123)
            known_noise = torch.randn_like(shifted_mu)
            expected = shifted_mu + 2 * known_noise  # variance four means standard deviation two.
            torch.manual_seed(123)
            observed = model.reparameterise(shifted_mu, nonunit_logvar)
        good = good and observed.shape == expected.shape and torch.allclose(observed, expected, atol=1e-6)
        return check(good, "TODO 2: fresh samples preserve the mean and convert variance to spread",
                     "logvar=0 needs std=1; logvar=ln(4) needs std=2. Return mu + exp(0.5*logvar) * eps.")
    except Exception:
        return check(False, "", "TODO 2: match the shapes of mu, std and fresh noise.")


model_check = VAE().to(DEVICE)
encoder_ready = check_encoder(model_check)
sampling_ready = check_sampling(model_check)

# %% [markdown]
# **TODO 3:** implement reconstruction + beta × KL, averaged over images.
# **Hint:** use BCE with logits and `reduction="sum"`; sum KL over images and
# latent coordinates, then divide each term by the batch size.
# At beta one, binary targets give the negative ELBO. Greyscale targets in this
# lab use a conventional soft-target BCE surrogate, not a literal Bernoulli likelihood.
#
# ![How the two VAE costs train the model](fig:beginner_training)
#
# **What:** define the loss and check a one-image numerical case and gradient path.
# **Why:** summing pixels but averaging images keeps reconstruction and KL on compatible scales.
# **Expect:** reconstruction `784 × ln(2)`, KL `0.5`, and finite gradients in this constructed example.

# %%
def vae_loss(logits, x, mu, logvar, beta=1.0):
    ### BEGIN SOLUTION
    rec = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ### END SOLUTION
    ### STUB rec, kl = torch.tensor(0.0), torch.tensor(0.0)  # TODO 3: summed pixel and Gaussian costs.
    n = x.size(0)
    return (rec + beta * kl) / n, rec / n, kl / n


def check_loss():
    try:
        logits = torch.zeros(1, 784, device=DEVICE, requires_grad=True)
        x = torch.zeros(1, 1, 28, 28, device=DEVICE)
        mu = torch.tensor([[1.0, 0.0]], device=DEVICE, requires_grad=True)
        logvar = torch.zeros_like(mu)
        total, rec, kl = vae_loss(logits, x, mu, logvar, beta=4.0)
        expected_rec = 784 * math.log(2)
        values_ok = (abs(float(rec.detach()) - expected_rec) < 0.01
                     and abs(float(kl.detach()) - 0.5) < 1e-5
                     and abs(float(total.detach()) - expected_rec - 2) < 0.01)
        if total.requires_grad:
            total.backward()
        gradients_ok = (logits.grad is not None and mu.grad is not None
                        and bool(torch.isfinite(logits.grad).all()))
        # A second case checks nonunit variance, image averaging and its gradient route.
        batch_logits = torch.zeros(2, 784, device=DEVICE, requires_grad=True)
        batch_x = torch.zeros(2, 1, 28, 28, device=DEVICE)
        batch_mu = torch.tensor([[1.0,0.0],[0.0,1.0]], device=DEVICE, requires_grad=True)
        batch_logvar = torch.tensor([[math.log(4),0.0],[0.0,math.log(0.25)]], device=DEVICE, requires_grad=True)
        batch_total, batch_rec, batch_kl = vae_loss(batch_logits, batch_x, batch_mu, batch_logvar, beta=2.0)
        # Summed KL is 2.125, hence mean KL is 1.0625 for these two images.
        values_ok = values_ok and abs(float(batch_rec.detach()) - expected_rec) < 0.01
        values_ok = values_ok and abs(float(batch_kl.detach()) - 1.0625) < 1e-5
        values_ok = values_ok and abs(float(batch_total.detach()) - expected_rec - 2.125) < 0.01
        if batch_total.requires_grad:
            batch_total.backward()
        expected_variance_grad = torch.tensor([[1.5,0.0],[0.0,-0.375]], device=DEVICE)
        gradients_ok = gradients_ok and batch_logvar.grad is not None
        gradients_ok = gradients_ok and torch.allclose(batch_logvar.grad, expected_variance_grad, atol=1e-6)
        print("Constructed loss case:", {"reconstruction": round(float(rec.detach()), 3),
                                         "KL": round(float(kl.detach()), 3)})
        return check(values_ok and gradients_ok, "TODO 3: loss values and gradients match",
                     "Include mean and variance in KL, average over images, and keep the log-variance gradient connected.")
    except Exception:
        return check(False, "", "TODO 3: check tensor devices, reductions and the sign of Gaussian KL.")


loss_ready = check_loss()

# %% [markdown]
# ## Part 2 · Train and monitor
#
# **Read:** `zero_grad` clears old gradients; `backward` computes new gradients;
# `step` changes weights. Displaying an image does none of these updates.
# **Predict:** must reconstruction and KL both fall every epoch?
# **Core:** train real two-dimensional and sixteen-dimensional VAEs on the bounded subset.
# **Change one thing:** use two epochs rather than three, and describe a visible limitation.
# **Stretch:** increase training data or epochs after timing the core.
# **Common errors:** unfinished checks should be fixed before training. A missing gradient
# often means a detached loss. A growing KL alone does not show failure; inspect both costs.

# %% [markdown]
# **What:** train two small VAEs with Adam and record per-image reconstruction and KL.
# **Why:** one model makes a two-dimensional map; the other provides a larger code for interpolation.
# **Expect:** finite epoch records and two loss curves. Quality depends on your actual run.
# Each call uses the same seeded batch order; these short runs differ from the archived lecture models.

# %%
if not (encoder_ready and sampling_ready and loss_ready):
    raise ValueError("Finish TODOs 1–3 and rerun their checks before training.")


def train(model, epochs=EPOCHS, beta=1.0, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    for ep in range(epochs):
        model.train()
        sums = np.zeros(3)
        images = 0
        for x, _ in loader:
            x = x.to(DEVICE)
            logits, mu, logvar = model(x)
            total, rec, kl = vae_loss(logits, x, mu, logvar, beta)
            opt.zero_grad()
            total.backward()
            opt.step()
            # Weight the final partial batch by its number of images.
            sums += np.array([total.item(), rec.item(), kl.item()]) * len(x)
            images += len(x)
        values = sums / images
        history.append(values)
        if verbose:
            print(f"epoch {ep+1}: total {values[0]:.2f} | reconstruction {values[1]:.2f} | KL {values[2]:.2f}")
    model.eval()
    return np.asarray(history)


vae2 = VAE(zdim=2).to(DEVICE)
hist = train(vae2)
vae16 = VAE(zdim=16).to(DEVICE)
hist16 = train(vae16)
check(bool(np.isfinite(hist).all() and np.isfinite(hist16).all()),
      "Training costs are finite", "Inspect loss scaling, learning rate and log-variance.")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
epochs = np.arange(1, len(hist) + 1)  # count epochs from 1, as printed during training
axes[0].plot(epochs, hist[:, 1], marker="o", label="2-D")
axes[0].plot(epochs, hist16[:, 1], marker="s", label="16-D")
axes[0].set_title("Reconstruction per image")
axes[1].plot(epochs, hist[:, 2], marker="o", label="2-D")
axes[1].plot(epochs, hist16[:, 2], marker="s", label="16-D")
axes[1].set_title("KL per image")
for ax in axes:
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 3 · Explore the code space
#
# **Read:** digit labels colour the scatter; they were not used in this VAE's training loss.
# A plotted encoder mean is different from a random code sampled during training.
# **Predict:** will every point in a decoded grid look like a valid digit?
# **Core:** inspect encoder means, prior grids, reconstruction and latent interpolation.
# **Change one thing:** move one code coordinate using the slider, then name one changed feature.
# **Stretch:** repeat an interpolation with another pair, reporting an unsuccessful path too.
# **Common errors:** the decoder code size must match `model.zdim`. A CPU NumPy conversion
# needs `.cpu()`. A smooth numerical change is not proof of a meaningful or realistic transition.

# %% [markdown]
# **What:** encode held-out images and decode a fixed grid of two-number codes.
# **Why:** the scatter inspects representation, while the grid tests generation without input images.
# **Expect:** coloured means `[TEST_LIMIT,2]` and a grid with both recognisable and weak samples.
# The grid is vectorised into one decoder call, avoiding hundreds of small calls.

# %%
with torch.no_grad():
    mu_test, _ = vae2.encode(x_test.to(DEVICE))
means = mu_test.cpu().numpy()
print("Encoder means:", means.shape)
plt.figure(figsize=(6.5, 4.8))
sc = plt.scatter(means[:, 0], means[:, 1], c=y_test.numpy(), cmap="tab10", s=8)
plt.colorbar(sc, ticks=range(10), label="digit label (inspection only)")
plt.xlabel("First code coordinate")
plt.ylabel("Second code coordinate")
plt.title("Held-out encoder means")
plt.show()

from scipy.stats import norm
n = 10
axis_codes = norm.ppf(np.linspace(0.03, 0.97, n))
codes = torch.tensor([[zx, zy] for zy in axis_codes[::-1] for zx in axis_codes],
                      dtype=torch.float32, device=DEVICE)
grid_images = vae2.decode(codes).cpu().numpy()
canvas = np.vstack([np.hstack(grid_images[i*n:(i+1)*n]) for i in range(n)])
print("Prior grid codes:", tuple(codes.shape), "| decoded images:", grid_images.shape)
plt.figure(figsize=(6, 6))
plt.imshow(canvas, cmap="gray_r")
plt.axis("off")
plt.title("Chosen prior codes: inspect successes and failures")
plt.show()

# %% [markdown]
# **What:** vary two latent coordinates with sliders while keeping decoder weights fixed.
# **Why:** one small decoder call shows what this trained two-dimensional representation changes.
# **Expect:** a single decoded image. Dragging stops before computation (`continuous_update=False`).
# This optional display cell is tagged `skip-test`; the core grid above also works without widgets.
# **Common errors:** if controls do not appear, check ipywidgets in the selected kernel and
# display the static grid instead. Do not rerun training on each slider movement.

# %% tags=["skip-test"]
try:
    import ipywidgets as widgets
    from IPython.display import display

    def show_latent(z1=0.0, z2=0.0):
        code = torch.tensor([[z1, z2]], dtype=torch.float32, device=DEVICE)
        image = vae2.decode(code)[0].cpu().numpy()
        plt.figure(figsize=(2.8, 2.8))
        plt.imshow(image, cmap="gray_r", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Code ({z1:.1f}, {z2:.1f}); fixed weights")
        plt.show()

    widgets.interact(show_latent,
                     z1=widgets.FloatSlider(min=-2, max=2, step=0.2, value=0, continuous_update=False),
                     z2=widgets.FloatSlider(min=-2, max=2, step=0.2, value=0, continuous_update=False))
except ImportError:
    print("Widget library unavailable. Use the static grid above; all core tasks still work.")

# %% [markdown]
# **TODO 4:** return a straight line of codes from `z1` to `z2`.
# **Hint:** the column `ts` broadcasts over coordinates in `(1-ts)*z1 + ts*z2`.
# At `t=0`, return the first endpoint; at `t=1`, return the second.
#
# **What:** implement interpolation, check its endpoints and compare decoded paths with pixel blending.
# **Why:** latent blending and direct image blending are different computations.
# **Expect:** five toy codes with a midpoint `[1,2]`, then ten decoded images on the real path.

# %%
def interpolate(z1, z2, steps=10):
    ts = torch.linspace(0, 1, steps, device=z1.device).unsqueeze(1)
    ### BEGIN SOLUTION
    return (1 - ts) * z1 + ts * z2
    ### END SOLUTION
    ### STUB return z1.repeat(steps, 1)  # TODO 4: blend both endpoint codes.


def check_interpolation():
    try:
        a = torch.tensor([0.0, 0.0], device=DEVICE)
        b = torch.tensor([2.0, 4.0], device=DEVICE)
        path = interpolate(a, b, steps=5)
        good = (tuple(path.shape) == (5, 2) and torch.allclose(path[0], a)
                and torch.allclose(path[-1], b) and torch.allclose(path[2], b/2))
        return check(good, "TODO 4: interpolation endpoints and midpoint match",
                     "Use (1-ts)*z1 + ts*z2; preserve both endpoints and all coordinates.")
    except Exception:
        return check(False, "", "TODO 4: ts needs shape [steps,1] for coordinate broadcasting.")


interpolation_ready = check_interpolation()
if not interpolation_ready:
    raise ValueError("Finish TODO 4 and rerun its check before decoding the path.")
with torch.no_grad():
    mu16, _ = vae16.encode(x_test.to(DEVICE))
a_idx = int((y_test == 1).nonzero(as_tuple=True)[0][0])
b_idx = int((y_test == 7).nonzero(as_tuple=True)[0][0])
path = interpolate(mu16[a_idx], mu16[b_idx])
print("Interpolated latent codes:", tuple(path.shape))
decoded_path = vae16.decode(path).cpu().numpy()
pixel_blend = [(1-t)*x_test[a_idx, 0].numpy()+t*x_test[b_idx, 0].numpy()
               for t in np.linspace(0, 1, 10)]
fig, axes = plt.subplots(2, 1, figsize=(10, 3))
axes[0].imshow(np.hstack(decoded_path), cmap="gray_r")
axes[0].set_title("Interpolate codes, then decode")
axes[1].imshow(np.hstack(pixel_blend), cmap="gray_r")
axes[1].set_title("Blend input pixels directly")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# **What:** generate twelve images using only fresh prior codes and the trained decoder.
# **Why:** this separates generation from the encoder route used for reconstruction.
# **Expect:** codes `[12,16]` and displayed pixel probabilities `[12,28,28]`.
# Record one recognisable sample and one failure instead of selecting only attractive examples.
#
# ![Generate a digit without an input image](fig:beginner_inference)

# %%
prior_generator = torch.Generator(device=DEVICE).manual_seed(SEED+1)
z_fixed = torch.randn(12, 16, device=DEVICE, generator=prior_generator)
new_images = vae16.decode(z_fixed).cpu().numpy()
print("Prior codes:", tuple(z_fixed.shape), "| generated images:", new_images.shape)
check(new_images.shape == (12, 28, 28), "Generation needs no input image",
      "Use prior codes matching zdim and run the decoder, then sigmoid and reshape.")
plt.figure(figsize=(10, 1.8))
plt.imshow(np.hstack(new_images), cmap="gray_r")
plt.axis("off")
plt.title("Fresh prior samples: report typical quality and failures")
plt.show()

# %% [markdown]
# ## Part 4 · Beta changes weighting; retraining changes weights
#
# **Read:** beta multiplies KL in `reconstruction + beta * KL`.
# **Predict:** if reconstruction is 120 and KL is 3, what changes when beta rises from 1 to 4?
# **Core:** recompute the objective for one fixed batch without retraining the model.
# **Change one thing:** move the beta slider; explain why the images do not change.
# **Stretch:** enable the explicit switch to train separate beta models and compare fixed-code samples.
# **Common errors:** a cost slider is not a training experiment. Total losses with different
# beta values are different objectives; compare both components and the intended task.
# The measured, longer beta study remains in the lecture and Word notes, with its original settings.

# %% [markdown]
# **What:** compute reconstruction and KL for fixed model outputs, then vary only their weighting.
# **Why:** this isolates the arithmetic role of beta before an optional training comparison.
# **Expect:** unchanged reconstruction and KL, with a different weighted total for each beta.
# These printed values come from this run; they do not replace the archived lecture measurements.
# The injected setup contains `USE_GEMINI = False` and `USE_OPENAI = False`.
# Leave both false for the complete local path. To request an optional explanation,
# enable exactly one switch and provide that provider's key through the setup instructions.
# This cell sends invented arithmetic only; a missing key or failed request uses the local explanation.

# %%
with torch.no_grad():
    batch = x_test[:32].to(DEVICE)
    fixed_logits, fixed_mu, fixed_logvar = vae16(batch)
    _, fixed_rec, fixed_kl = vae_loss(fixed_logits, batch, fixed_mu, fixed_logvar)
fixed_rec = float(fixed_rec)
fixed_kl = float(fixed_kl)
for beta in [0.1, 1.0, 4.0]:
    print(f"Fixed weights: beta {beta:g} | rec {fixed_rec:.2f} | KL {fixed_kl:.2f} | total {fixed_rec+beta*fixed_kl:.2f}")
check(fixed_kl >= -1e-5, "Gaussian KL is nonnegative within rounding",
      "Inspect the KL sign and use log-variance rather than standard deviation.")
local_interpretation = (
    "Beta changes the weight of KL. With fixed outputs, reconstruction and KL stay unchanged; "
    "the total changes. Training separate models can change their outputs, so compare both costs and samples."
)
api_interpretation = ask_model(
    "Explain this invented VAE example in three short sentences: reconstruction cost 120, KL 3; "
    "beta 1 gives total 123 and beta 4 gives total 132. State that recomputing this total "
    "does not retrain a model or establish sample quality."
)
print("Optional explanation:", api_interpretation if api_interpretation is not None else local_interpretation)

# %% [markdown]
# **What:** use a beta slider for the same frozen batch outputs.
# **Why:** see the objective change instantly without expensive training or API calls.
# **Expect:** three costs and an explicit reminder that the model weights stay fixed.
# This `skip-test` display is optional; the printed beta table above is the complete local fallback.

# %% tags=["skip-test"]
try:
    import ipywidgets as widgets

    def show_beta(beta=1.0):
        print(f"Reconstruction {fixed_rec:.2f}; KL {fixed_kl:.2f}; weighted total {fixed_rec+beta*fixed_kl:.2f}")
        print("Weights and outputs are fixed. This slider does not train a new beta-VAE.")

    widgets.interact(show_beta,
                     beta=widgets.FloatSlider(min=0, max=4, step=0.1, value=1, continuous_update=False))
except ImportError:
    print("Use the fixed-output beta table above if widgets are unavailable.")

# %% [markdown]
# ## Part 5 · Reconstruction as an anomaly score
#
# **Read:** a score ranks inputs; choosing a decision threshold is a separate task.
# ROC AUC compares rankings across possible thresholds. It is not classification accuracy.
# **Predict:** why might clothing versus digits be easier than finding a scratch on a digit-like image?
# **Core:** implement TODO 5 and compare held-out digits with clothing.
# **Change one thing:** limit the display to 200 images per dataset; discuss uncertainty.
# **Stretch:** evaluate realistic corruptions with settings chosen on development data.
# **Common errors:** one scalar for the whole batch cannot rank images. Do not change a
# threshold using final test outcomes. High AUC on a different dataset does not establish real-world defect detection.
#
# **TODO 5 hint:** request `reduction="none"`, sum pixel costs with `dim=1`, and preserve the image dimension.
# This score decodes the encoder mean for reproducibility rather than resampling a code on every call.

# %% [markdown]
# **What:** implement a deterministic per-image reconstruction score and check its reduction.
# **Why:** ranking anomalies needs one score per input, with a clear scoring convention.
# **Expect:** three finite costs near `784 × ln(2)` from a constructed constant-logit model.
# The constructed test checks arithmetic; it does not evaluate trained-model quality.

# %%
@torch.no_grad()
def recon_error(model, x):
    model.eval()
    x = x.to(DEVICE)
    mu, _ = model.encode(x)
    logits = model.dec(mu)  # Fix the scoring path: decode the encoder mean.
    ### BEGIN SOLUTION
    return F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="none").sum(dim=1).cpu().numpy()
    ### END SOLUTION
    ### STUB return np.zeros(len(x))  # TODO 5: one summed pixel cost per image.


def check_reconstruction_scores():
    class ConstantModel(nn.Module):
        def encode(self, x):
            mu = torch.zeros(len(x), 2, device=x.device)
            return mu, torch.zeros_like(mu)

        def dec(self, z):
            return torch.zeros(len(z), 784, device=z.device)

    try:
        scores = recon_error(ConstantModel(), torch.zeros(3, 1, 28, 28))
        good = scores.shape == (3,) and np.allclose(scores, 784*math.log(2), atol=0.01)
        return check(good, "TODO 5: one correctly summed cost per image",
                     "Use reduction='none', sum dim=1, and return [number_of_images] scores.")
    except Exception:
        return check(False, "", "TODO 5: keep the image axis while summing only pixels.")


scores_ready = check_reconstruction_scores()

# %% [markdown]
# **What:** download the public clothing test set, score both datasets and plot their distributions.
# **Why:** an easy dataset-shift demonstration exposes the limits of reconstruction-based anomaly detection.
# **Expect:** one measured AUC for this core model, plus overlapping or separated score histograms.
# Report the actual number and failures; it need not match the longer archived run.

# %%
from sklearn.metrics import roc_auc_score
if not scores_ready:
    raise ValueError("Finish TODO 5 and rerun its score check first.")
fashion = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
x_fashion, _ = next(iter(DataLoader(Subset(fashion, range(TEST_LIMIT)), batch_size=TEST_LIMIT)))
e_digits = recon_error(vae16, x_test)
e_fashion = recon_error(vae16, x_fashion)
labels = np.r_[np.zeros(len(e_digits)), np.ones(len(e_fashion))]
auc = roc_auc_score(labels, np.r_[e_digits, e_fashion])
print("Score shapes:", e_digits.shape, e_fashion.shape, "| core-run ROC AUC:", round(float(auc), 4))
check(bool(np.isfinite(e_digits).all() and np.isfinite(e_fashion).all()),
      "Anomaly scores are finite", "Inspect logits and the per-image BCE reduction.")
plt.figure(figsize=(8, 3.8))
bins = np.linspace(0, np.percentile(np.r_[e_digits, e_fashion], 99), 40)
plt.hist(e_digits, bins=bins, alpha=0.7, label="digits")
plt.hist(e_fashion, bins=bins, alpha=0.7, label="clothing")
plt.xlabel("Mean-code reconstruction BCE per image")
plt.legend()
plt.title(f"Core-run AUC = {auc:.4f}; artificial anomaly task")
plt.show()

# %% [markdown]
# ## Optional stretch · train beta models and add a class request
#
# **Read:** a conditional VAE gives a digit label to both networks. A one-hot label
# has ten entries; one is one and the rest zero.
# **Predict:** does asking for a seven guarantee a convincing seven?
# **Core:** leave `RUN_STRETCH = False`; explain how conditioning changes the input sizes.
# **Change one thing:** after completing the core, enable the switch for this optional training.
# **Stretch:** compare beta models using identical initial weights, batch order and prior codes;
# then repeat several seeds before claiming a consistent effect.
# **Common errors:** a beta comparison needs separately trained models, not only a new cost
# multiplier. Conditional decoding must concatenate a label with the latent code. This longer
# route is outside the measured core budget; record its own settings and elapsed time.

# %% [markdown]
# **What:** define a conditional VAE and optionally train the longer comparisons.
# **Why:** extra label inputs let the decoder receive a requested class alongside a random code.
# **Expect:** the default prints a skipped-stretch message. Opt-in training prints beta costs
# and displays requested digit rows; inspect errors as well as successful samples.

# %%
class CVAE(VAE):
    def __init__(self, zdim=16, hidden=HIDDEN):
        super().__init__(zdim, hidden)
        self.enc = nn.Sequential(nn.Linear(784+10, hidden), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(zdim+10, hidden), nn.ReLU(), nn.Linear(hidden, 784))

    def forward(self, x, y):
        label = F.one_hot(y, 10).float()
        h = self.enc(torch.cat([x.view(-1, 784), label], dim=1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterise(mu, logvar)
        return self.dec(torch.cat([z, label], dim=1)), mu, logvar

    @torch.no_grad()
    def generate(self, digit, n=8):
        z = torch.randn(n, self.zdim, device=DEVICE)
        label = F.one_hot(torch.full((n,), digit, device=DEVICE), 10).float()
        return torch.sigmoid(self.dec(torch.cat([z, label], dim=1))).view(-1, 28, 28)


if RUN_STRETCH and not SMOKE:
    results = {}
    for beta in [0.1, 1.0, 4.0]:
        # Reset initial weights and sampling seed for a clearer controlled comparison.
        torch.manual_seed(SEED)
        model = VAE(zdim=16).to(DEVICE)
        h = train(model, epochs=5, beta=beta)
        results[beta] = {"reconstruction": float(h[-1, 1]), "KL": float(h[-1, 2]),
                         "samples": model.decode(z_fixed).cpu().numpy()}
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.5))
    for ax, (beta, result) in zip(axes, results.items()):
        ax.imshow(np.hstack(result["samples"]), cmap="gray_r")
        ax.set_title(f"beta {beta:g}: rec {result['reconstruction']:.2f}, KL {result['KL']:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    cvae = CVAE().to(DEVICE)
    opt = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    for epoch in range(5):
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, mu, logvar = cvae(x, y)
            total, _, _ = vae_loss(logits, x, mu, logvar)
            opt.zero_grad()
            total.backward()
            opt.step()
    cvae.eval()
    rows = [np.hstack(cvae.generate(d).cpu().numpy()) for d in range(10)]
    plt.figure(figsize=(6, 7))
    plt.imshow(np.vstack(rows), cmap="gray_r")
    plt.axis("off")
    plt.title("Conditional VAE: each row requests a digit")
    plt.show()
else:
    print("Optional stretch skipped. The CPU core is complete without this training.")

# %% [markdown]
# ## Part 6 · Evidence and reflection
#
# **Read:** a result needs its settings, a measured observation and a limitation.
# **Predict:** will your shorter core reproduce the archived lecture figures exactly?
# **Core:** fill the table using your outputs, then answer the five self-checks.
# **Change one thing:** name one next experiment and the evidence that would support it.
# **Stretch:** compare another seed with the original seed while keeping other settings fixed.
# **Common errors:** do not copy archived metrics into this run's table. Do not report an
# unmeasured runtime or claim that changing beta alone guarantees better generated images.
#
# | Experiment | Actual settings | Measured observation | Limitation |
# |---|---|---|---|
# | VAE training | subset, epochs, hidden size, seed | reconstruction and KL | |
# | Generation | code size, fixed prior codes | typical success and failure | |
# | Beta weighting | fixed outputs, beta values | weighted totals | no retraining |
# | Anomaly score | mean-code BCE, test sizes | ROC AUC | artificial dataset shift |
#
# **AI assistance:** disclose tools and prompts used, the checks you performed and your own explanation.
# Optional Gemini or OpenAI supplies a synthetic arithmetic explanation, not an evaluation of your images.
#
# ## Five self-checks
#
# 1. With mean 2, standard deviation 0.5 and noise −1, what code is sampled?
# 2. What standard deviation does `logvar=0` represent, and why?
# 3. Why sum BCE over pixels before averaging over images?
# 4. Which networks are required for generation from a fresh prior code?
# 5. What does the beta slider establish, and what can clothing-versus-digits AUC fail to establish?
#
# <!-- BEGIN ANSWER -->
# 1. The code is 2 + 0.5 × (−1) = 1.5.
# 2. Spread is one: exp(0.5 × 0) = 1. Zero log-variance does not mean no uncertainty.
# 3. The reconstruction cost should describe one complete image, on a compatible scale with per-image KL.
# 4. Draw the prior code and run the decoder. The encoder is used for supplied-image reconstruction, not this route.
# 5. The slider shows the cost multiplier with fixed outputs. Retraining can change outputs but is a separate experiment.
# A high clothing-versus-digits AUC can fail to establish reliable real-world defect detection, calibrated thresholds or low false alarms.
# <!-- END ANSWER -->
#
# **Before next week:** read Foster chapter 4 and the PyTorch DCGAN tutorial; identify what a learned realism checker adds.
