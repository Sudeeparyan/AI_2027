# Lecture plan at a glance

This week introduces the first deep generative model family of the module. Students should leave able to write the ELBO, explain each term, compute a Gaussian KL divergence, and implement a VAE. All result images in the slides come from real models trained with the lab code (see `curriculum/assets/make_week_02.py`), so students can connect the lab steps to the lecture. Their outputs can differ with the seed and training settings.

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–7 min | 1–4 | Warm-up | "Describe a face with 32 numbers": motivates codes, generation and gaps. |
| 7–27 min | 5–10 | 1 · From autoencoders to VAEs | Autoencoder recap; real latent plots; random codes decode to nonsense; AE vs. VAE. Quiz. |
| 27–60 min | 11–20 | 2 · Model and ELBO | Probabilistic model; architecture; a likelihood that is difficult to calculate exactly; ELBO; KL closed form with worked numbers; reconstruction term. Quiz. |
| 60–67 min | – | Break | |
| 67–84 min | 21–25 | 3 · Reparameterisation | The trick, why it matters, the PyTorch model, training curves and posterior collapse. |
| 84–104 min | 26–31 | 4 · Exploring and evaluating | Reconstructions, latent grid, interpolation, β trade-off, evaluation table. |
| 104–118 min | 32–37 | 5 · Variants and applications | CVAE, β-VAE, VQ-VAE, hierarchical; VAE inside latent diffusion; applications; anomaly demo; responsible AI. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Derive the ELBO on the board (see the derivation below) rather than only showing the slide. Students who see the two-line Jensen derivation remember that the gap is a KL divergence, which is a common exam question.

<!-- pagebreak -->

# Lecture notes

## Begin with two different routes

Show one digit. The reconstruction route is image → encoder mean/spread → sampled code → decoder. Then hide the digit: generation is prior code → decoder. Ask students to draw both routes before introducing q, p, $\theta$ or $\phi$.

**One-number check.** With $\mu$ = 1, $\sigma$ = 0.5 and $\epsilon$ = −2, z = 0. Repeating the same image can give another z because $\epsilon$ changes. `logvar = 0` gives $\sigma$ = 1, not zero.

**Learning loop.** Predict a shape, run a forward pass, compare the result, explain an error, then repeat on a second batch. Track the two losses separately. Before comparing $\beta$ values, predict their effect on the cost formula; use measured samples to test the prediction rather than promising sharper or perfectly complete outputs.



## 1. From autoencoders to VAEs

### Autoencoders

An **autoencoder** learns to rebuild its input using two networks. The **encoder** compresses image $x$ into a short numerical code $z$. The **decoder** rebuilds an image $\hat{x}$ from that code. MNIST digits have 28 × 28 = 784 pixel values. A code might contain 2 or 32 numbers.

Training reduces reconstruction error, using mean squared error or binary cross-entropy. The short code must retain useful image information. A narrow code alone does not guarantee meaningful or separate features.

Autoencoders are useful for **compression**, **denoising** (train to reconstruct clean images from corrupted ones), **representation learning** (use $z$ as features for another model) and **anomaly detection** (unusual inputs reconstruct badly).

![An autoencoder compresses to a code and rebuilds the input](fig:ae_diagram)

### Why an autoencoder is a poor generator

The decoder learns from codes supplied by the encoder. Ordinary reconstruction training does not specify where those codes should lie or what scale they should use. A random code drawn from a standard Gaussian may land in a region the decoder has rarely seen. It can then produce a poor image. The VAE adds a known sampling distribution and a penalty that encourages encoder codes to match it.

![Real latent spaces of an autoencoder and a VAE with 2-D codes](fig:latent_ae_vs_vae)

![The same random codes decoded by each model](fig:samples_ae_vs_vae)

For generation we need a latent space that is:

* **continuous:** nearby codes decode to similar outputs; and
* **useful for prior sampling:** many likely prior codes should produce plausible outputs; verify this empirically.

A **variational autoencoder (VAE)** encourages a more useful sampling space with two changes: the encoder outputs a probability distribution $q(z \mid x)$ rather than a point, and the loss adds a **KL divergence** that pulls every such distribution towards a fixed **prior** $p(z) = \mathcal{N}(0, I)$.

> **Key idea:** A VAE is an autoencoder whose code is a probability distribution, regularised to look like a simple prior, so that sampling from the prior and decoding generates new data.

## 2. The VAE model and the ELBO

A **Gaussian** is a bell-shaped distribution. Its mean is the centre, its standard deviation is the spread, and its variance is spread squared. N(0, I) means zero-centred, unit-variance independent code dimensions. In formulas, θ and φ name learned weights; E means average, and the vertical bar means “given”.

### The probabilistic model

A VAE is a **latent variable model**:

1. **Prior:** $z \sim p(z) = \mathcal{N}(0, I)$.
2. **Decoder (likelihood):** $x \sim p_\theta(x \mid z)$, where a neural network with parameters $\theta$ outputs the parameters of the distribution over $x$ (for MNIST, one Bernoulli probability per pixel).
3. **Encoder (approximate posterior):** $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))$, a neural network with parameters $\phi$ that guesses which codes could have produced $x$.

A **prior** is the code distribution we choose before seeing an image. A **posterior** describes possible codes after seeing that image. The encoder estimates the posterior. Generation needs a prior code and the decoder. Training also uses the encoder to propose codes for each known input.

**Variational inference** means replacing a difficult distribution with an easier approximation. The VAE uses Gaussian approximations described by encoder-predicted means and variances.

![VAE architecture](fig:vae_arch)

### Why the likelihood is too difficult to calculate exactly

The likelihood of an image is an average of the decoder over all codes:

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$

**Read the integral.** Imagine only two possible codes, each with prior probability 0.5. If their likelihoods for x are 0.2 and 0.6, their weighted sum is 0.4. The continuous model uses an integral instead of this sum.

For a neural decoder, the integral has no convenient exact formula. Sampling codes from the prior is inefficient for explaining one specific image: most samples give it little probability. The exact posterior uses the same difficult denominator $p_\theta(x)$. The encoder instead learns a useful approximation, and we train with a computable lower bound.

### Deriving the evidence lower bound (ELBO)

Multiply and divide by the encoder distribution inside the integral, then apply **Jensen's inequality** ($\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$ because log is concave):

$$\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z \mid x)}\left[\frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right] \geq \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right]$$

**Read the inequality.** The expectation means an average under the encoder distribution. Jensen’s inequality moves log inside that average and gives a lower value. This is the mathematical step that makes a usable training bound.

Splitting the logarithm gives the familiar form:

$$\mathrm{ELBO}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - D_{\mathrm{KL}}\left(q_\phi(z \mid x)\,\Vert\,p(z)\right)$$

**Worked example.** If expected reconstruction log-probability is −120 and KL is 3, ELBO is −123. The loss we minimise is 123. These are illustrative values, not a measured run.

An exact identity shows what the bound loses:

$$\log p_\theta(x) = \mathrm{ELBO}(\theta, \phi; x) + D_{\mathrm{KL}}\left(q_\phi(z \mid x)\,\Vert\,p_\theta(z \mid x)\right)$$

**Worked example.** If ELBO = −123 and the posterior KL gap is 2, then log p(x) = −121. Adding a nonnegative gap explains the lower-bound direction.

**KL divergence** measures a difference between distributions. It cannot be negative, although it is not a symmetric distance. The equation shows that log likelihood equals ELBO plus KL to the true posterior. Therefore ELBO cannot exceed log likelihood.

In code, minimise **negative ELBO**, averaged over the image batch. This jointly trains the encoder and decoder. Improving a lower bound is useful, but does not guarantee every optimisation step raises the exact likelihood.

### Reading the ELBO: two forces

* **Reconstruction term** $\mathbb{E}_q[\log p_\theta(x \mid z)]$: codes drawn from the encoder must let the decoder rebuild $x$. Without KL pressure, the model has no reason to match the chosen prior; learned variances may shrink.
* **KL term** $D_{\mathrm{KL}}(q_\phi(z \mid x)\,\Vert\,p(z))$: every image's code distribution should resemble the prior. Alone, it makes the code distribution ignore the input.

Training seeks a compromise between preserving input information and matching the prior. It does not guarantee that every prior code decodes well. Basic pixel losses can favour average intensities where the code leaves image details uncertain, contributing to **blur**.

### The KL divergence between Gaussians

For a one-dimensional Gaussian encoder and a standard normal prior:

$$D_{\mathrm{KL}}\left(\mathcal{N}(\mu, \sigma^2)\,\Vert\,\mathcal{N}(0, 1)\right) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right)$$

**Start with a zero-cost case.** Mean 0 and standard deviation 1 give 0.5 × (0 + 1 − log 1 − 1) = 0. Work the shifted and narrowed cases below next.

For a $d$-dimensional diagonal Gaussian, sum over dimensions. In PyTorch, with the network outputting `logvar` $= \log\sigma^2$:

```python
kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
```

**Worked examples** (natural logs; ln 0.25 = −1.386, ln 4 = 1.386):

| Encoder q | Calculation | KL (nats) |
|---|---|---|
| μ = 0, σ = 1 | ½(0 + 1 − 0 − 1) | 0.00 |
| μ = 1, σ = 1 | ½(1 + 1 − 0 − 1) | 0.50 |
| μ = 0, σ = 0.5 | ½(0 + 0.25 + 1.386 − 1) | 0.32 |
| μ = 2, σ = 0.5 | ½(4 + 0.25 + 1.386 − 1) | 2.32 |
| μ = 1, σ = 2 | ½(1 + 4 − 1.386 − 1) | 1.31 |

Moving the mean away from zero and making the distribution either too narrow or too wide both cost nats. KL is **not symmetric**: $D_{\mathrm{KL}}(q\Vert p) \neq D_{\mathrm{KL}}(p\Vert q)$ in general.

![KL divergence for the worked examples](fig:kl_gaussians)

### The reconstruction term

The reconstruction term becomes a standard loss once we choose the decoder distribution:

* **Bernoulli decoder** (binary pixels): negative log-likelihood is binary cross-entropy summed over pixels. Our lab keeps greyscale MNIST intensities and uses the same formula as a conventional soft-target reconstruction surrogate; the literal Bernoulli ELBO interpretation requires binary observations:

$$-\log p_\theta(x \mid z) = -\sum_{i=1}^{784}\left[x_i \log \hat{x}_i + (1 - x_i)\log(1 - \hat{x}_i)\right]$$

**One-pixel example.** A target pixel of 1 predicted as 0.8 costs −ln(0.8), about 0.223. Predicting 0.2 costs about 1.609. Add the costs over all pixels.

* **Gaussian decoder** with fixed variance $\sigma^2_x$ (real-valued data): negative log-likelihood = $\Vert x - \hat{x}\Vert^2 / (2\sigma^2_x)$ + constant, i.e. **squared error**. The choice of $\sigma^2_x$ implicitly sets the balance with the KL term.

The expectation over $q$ is estimated with **one sampled code** per image per step (a Monte Carlo estimate); across many steps the noise averages out.

> **Common mistake:** averaging the BCE over pixels (PyTorch's default `reduction="mean"`) while summing the KL over dimensions. The KL term then dominates by a factor of 784 and every sample looks like an average blurry digit. Sum over pixels, then divide the total by the batch size.

## 3. Training: the reparameterisation trick

We need gradients of $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$ with respect to the encoder parameters $\phi$. A direct PyTorch `sample()` call disconnects the pathwise autograd route through its draw. Other gradient estimators exist, but the Gaussian **reparameterisation trick** provides a differentiable calculation for fixed noise:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

**One-number example.** Let $\mu$ = 1, $\sigma$ = 0.5 and $\epsilon$ = −2. Then z = 1 + 0.5 × (−2) = 0. A fresh $\epsilon$ changes z while still allowing gradients through $\mu$ and $\sigma$.

The trick samples $\epsilon$ from a fixed standard Gaussian, then shifts and scales it using the encoder outputs. This gives the same distribution for $z$ as direct Gaussian sampling. The noise is now an input to a differentiable calculation. Reconstruction gradients can flow through $z$ to $\mu$ and $\sigma$.

A score-function estimator such as REINFORCE is another option, but typically has higher variance here. Diffusion also uses "clean signal plus scaled noise" in week 4.

![Without and with the reparameterisation trick](fig:reparam)

### A complete PyTorch model

This is the lab's completed model and loss (the lab leaves TODOs in `encode`, `reparameterise` and `vae_loss`).

```python
class VAE(nn.Module):
    def __init__(self, zdim=2, hidden=400):
        super().__init__()
        self.zdim = zdim
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.mu_head = nn.Linear(hidden, zdim)
        self.logvar_head = nn.Linear(hidden, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 784))

    def encode(self, x):
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.dec(z), mu, logvar

    @torch.no_grad()
    def decode(self, z):
        # Display pixel probabilities; this path needs a code, not an input image.
        return torch.sigmoid(self.dec(z)).view(-1, 28, 28)


def vae_loss(logits, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    n = x.size(0)
    return (rec + beta * kl) / n, rec / n, kl / n
```

**Variance** must be positive. The network predicts **log-variance**, which can be any real number, then exponentiates it. For logvar = 0, variance = exp(0) = 1 and standard deviation = 1.

`VAE.encode` predicts mean and log-variance. `VAE.decode` applies sigmoid to decoder logits and returns displayable pixels. These are predicted pixel probabilities. The lab displays them directly rather than drawing each Bernoulli pixel separately.

### Training practice

* Adam with learning rate 10⁻³, batch size 128 and 10 epochs. The model is small enough for a laptop CPU; a GPU is faster. Time one epoch before planning longer runs.
* **Always log reconstruction and KL separately.** Initial KL depends on the network initialisation. It can rise as codes carry more image information; a rising KL alone does not mean training has failed.
* **Posterior collapse:** the KL falls to (almost) zero and the decoder ignores $z$. It is common when the decoder is powerful (e.g. autoregressive). Remedies: **KL warm-up** (increase β from 0 to 1 over the first epochs), **free bits** (no penalty below a minimum KL per dimension), or a less powerful decoder.

![Real training curves for the 2-D VAE](fig:training_curves)

## 4. Exploring and evaluating the latent space

### Reconstructions and blur

The held-out digit reconstructions are recognisable but often softer than the originals. Pixel losses can reward average intensities when details are uncertain. Basic VAE codes also balance image information against the prior. These effects can blur edges. GANs introduce a learned realism checker in week 3. Latent diffusion uses an autoencoder trained with extra visual and adversarial losses in week 4.

![Test images and their reconstructions](fig:reconstructions)

### Latent grid and interpolation

A two-dimensional code can be plotted as a map. Decode a grid of code points and compare neighbouring cells. Smooth changes and many recognisable digits are evidence about this trained model, rather than a guarantee that every possible code works.

**Interpolation** chooses points between codes $z_1$ and $z_2$: $z_t = (1-t)z_1 + tz_2$. At $t = 0.5$, use their midpoint. Decode each point and inspect the path. In high dimensions, spherical interpolation follows a curved path closer to the typical radius of random Gaussian codes.

![Decoded grid of the 2-D latent space](fig:manifold)

![Latent interpolation between digits](fig:interpolation)

### The β trade-off

Weighting KL by β gives the **β-VAE** objective. With β = 1 and a correctly specified decoder likelihood, the two terms form the ELBO. Our greyscale soft-target BCE experiment uses the conventional reconstruction surrogate. Measured on the archived 16-D models after 5 epochs:

| β | Reconstruction (BCE per image) | KL (nats per image) | Samples from the prior |
|---|---|---|---|
| 0.1 | {{W02_REC_B01}} | {{W02_KL_B01}} | sharp details but many malformed digits |
| 1.0 | {{W02_REC_B1}} | {{W02_KL_B1}} | useful balance in this archived run |
| 4.0 | {{W02_REC_B4}} | {{W02_KL_B4}} | well-formed but blurry, generic |

$\beta$ multiplies the KL cost. Increasing it puts more pressure on code distributions to match the prior, often reducing code information and increasing reconstruction error. Lower $\beta$ can improve reconstruction while weakening random prior samples. Actual effects depend on training.

The **rate–distortion** view names KL the information rate and reconstruction error the distortion. A **disentangled** code aims to separate factors such as stroke width and slant. $\beta$ above one can encourage this but cannot guarantee it.

![Samples for three values of β](fig:beta_samples)

![Reconstruction and KL for each β](fig:beta_tradeoff)

### How to evaluate a VAE

| Aspect | Metric | Caveat |
|---|---|---|
| Fit to data | Held-out negative ELBO (nats per image, or bits per dimension) | A bound; only compare models with the same decoder likelihood |
| Reconstruction | BCE or MSE on held-out data; side-by-side pictures | Good reconstructions do not guarantee good samples |
| Sample quality | Fréchet Inception Distance (FID, week 3); human rating with a rubric | FID needs thousands of samples and a relevant feature network |
| Latent space | Interpolations, traversals, disentanglement metrics | Visual inspection is subjective; report seeds and failures |
| Usefulness | Task metric, e.g. anomaly-detection ROC AUC | Needs realistic held-out test cases |

## 5. Variants and applications

* **Conditional VAE (CVAE):** condition both networks on a label or other information $y$ to model $p(x \mid z, y)$; sampling with a chosen $y$ generates that class (lab extension).
* **β-VAE:** β > 1 increases prior pressure and can encourage separated factors; test interpretability rather than assuming it.
* **VQ-VAE:** replaces the Gaussian code with the nearest vector in a learned **codebook**, so images (or audio) become grids of **discrete tokens**. This is the basis of many image and audio tokenizers used by autoregressive and multimodal transformers.
* **Hierarchical VAEs** (e.g. NVAE): several layers of latents; can model detail at different scales.

### VAEs inside latent diffusion

Latent diffusion first learns a smaller image representation. In the Stable Diffusion example, 512 × 512 × 3 = 786,432 pixels become 64 × 64 × 4 = 16,384 hidden values: 48 times fewer numbers. Text-to-image generation starts from random noise in that smaller space; the diffusion model removes the noise step by step while following the prompt, and the VAE decoder turns the final latent into pixels. The encoder is not needed for that path. It turns training photos into latents, and it encodes an existing picture when you edit it (image-to-image).

The autoencoder uses a small KL weight plus visual and adversarial losses to preserve detail. It is more specialised than our basic MNIST VAE. This explains why VAE-style components remain useful even when the main image generator is diffusion.

### Applications

* **Anomaly detection:** reconstruction scores can flag inputs unlike the training data; validate realistic failures because some unusual inputs reconstruct well.
* **Synthetic data:** sample new records or images to augment small datasets (with privacy caveats).
* **Molecule and design search:** improve in a smooth latent space, then decode candidates.
* **Compression and tokens:** learned codecs; latent spaces for diffusion and multimodal models.

In our demonstration, a VAE trained only on digits separated unseen digits from Fashion-MNIST clothing images using reconstruction error with **ROC AUC = {{W02_AUC}}**: a randomly chosen clothing image had a higher error than a randomly chosen digit {{W02_AUC_PCT}} of the time. An AUC of 1 would mean every clothing image scored higher than every digit; {{W02_AUC}} is close but not perfect. Real anomalies are much subtler; always evaluate on realistic, held-out cases.

![Reconstruction error for digits and clothing](fig:anomaly)

# Common misconceptions

| Misconception | Correction |
|---|---|
| "A VAE is just an autoencoder with noise added." | The noise comes from a learned distribution, and the KL term to a prior is what makes the latent space usable for generation. |
| "The ELBO is the log-likelihood." | It is a lower bound; the gap equals the KL between the encoder and the true posterior. |
| "Lower KL is always better." | KL = 0 means the code carries no information about the input; the balance with reconstruction matters. |
| "Blurry samples mean a bug." | Some blur is expected (overlapping codes, pixel-wise likelihood). Severe blur often means the loss terms are mis-scaled (mean vs. sum). |
| "The encoder is needed to generate." | Generation uses only the prior and the decoder; the encoder is used in training and for encoding real images. |
| "Interpolation that looks good proves the model is good." | carefully selected best pairs can mislead; evaluate quantitatively and show failures. |
| "VAEs are obsolete." | Their autoencoders power latent diffusion; VQ-VAE ideas power image and audio tokenizers. |

# Responsible AI lens: VAEs and synthetic data

* **Privacy:** synthetic data is not automatically anonymous. Generative models can memorise rare records. Check for near-duplicates of training data and consider differential privacy for sensitive data.
* **Bias in anomaly detection:** the detector flags whatever differs from its training data. If a group is under-represented, its normal cases look anomalous (for example in fraud or clinical screening). Ask "normal for whom?" and measure error rates across groups.
* **Consent and licences:** MNIST is public; faces, medical images and voices require consent, a lawful basis (GDPR) and appropriate licences.
* **Honest reporting:** report typical samples, failure cases, seeds and metrics, not only the best interpolation.

# Lab guide and answers

**Runtime:** Colab or a laptop CPU. The complete solution notebook ran in about 9 minutes on the laptop CPU used to check this pack (9 October 2026); a GPU is faster. Ask students to record their own timings. MNIST and Fashion-MNIST download through torchvision. **Hand-in:** completed TODOs, figures, the β table and four written answers.

## TODOs and expected results

* **TODO 1** in `VAE.encode`: `mu, logvar = self.mu_head(h), self.logvar_head(h)`.
* **TODO 2** `std = torch.exp(0.5 * logvar); return mu + std * torch.randn_like(std)`.
* **TODO 3** BCE with logits summed over pixels; KL `-0.5 * sum(1 + logvar - mu**2 - exp(logvar))`; divide by batch size.
* **Question 1:** if initial pixel predictions are near 0.5, reconstruction cost is approximately 784 × ln 2 ≈ 543. If μ ≈ 0 and σ ≈ 1, KL is near zero. Check actual initial predictions; these conditions depend on initialisation.
* **Training (2-D model, 10 epochs):** reconstruction falls to about {{W02_REC2}} nats per image; the KL dips after the first epoch, then rises slowly to {{W02_KL2}} nats at epoch 10 (measured with the lecture model). Your values will differ a little.
* **Part 3:** the scatter shows class clusters packed around the origin; the grid decodes to a smooth map of digits. **TODO 4** `(1 - ts) * z1 + ts * z2`. Latent interpolation may change strokes more coherently; inspect the actual intermediate digits and compare with pixel blending.
* **Part 4 (β):** expect the pattern in the table above; exact values vary with seeds and epochs.
* **TODO 5** per-image BCE: `F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="none").sum(dim=1)`. The saved digit-versus-clothing experiment had ROC AUC {{W02_AUC}} (the lab prints four decimals so it is not shown as 1.000); record your measured result and do not generalise it to realistic defects.
* **Extension (CVAE):** each row should show the requested digit in varied styles.

## Troubleshooting

* *Samples all look like the same blurry blob:* the BCE was averaged instead of summed, so the KL dominates.
* *NaN loss:* usually `logvar` exploding; lower the learning rate or clamp `logvar` to [−10, 10].
* *Very slow on CPU:* reduce epochs to 3 for the β study; inspect whether the observed pattern changes with shorter training.

# Practice questions with model answers

## Multiple choice

1. The KL term in the VAE loss: **(a)** keeps each encoder distribution close to the prior; **(b)** measures how well the decoder reconstructs each input; **(c)** is the log-likelihood of the data under the decoder; **(d)** is only used at generation time, to sample new codes. *Answer: (a).*
2. The reparameterisation trick is needed because: **(a)** the decoder is non-linear, so its loss cannot be differentiated; **(b)** MNIST images are binary, which breaks ordinary backpropagation; **(c)** the prior is not Gaussian, so the KL term has no closed form; **(d)** an ordinary sample() call passes no pathwise gradient to μ and σ. *Answer: (d).*
3. Increasing β from 1 to 4 usually: **(a)** sharpens reconstructions, because the KL weight is larger; **(b)** increases the KL, since codes spread further from the prior; **(c)** lowers the KL and increases reconstruction error; **(d)** has no effect, because β only scales the reported loss. *Answer: (c).*
4. In latent diffusion models the VAE is used to: **(a)** classify images so the denoiser knows what to draw; **(b)** compress images into a smaller latent space where diffusion runs; **(c)** encode the text prompt into vectors for cross-attention; **(d)** compute FID between generated and real image batches. *Answer: (b).*

## Short answer

1. **Compute the KL divergence to N(0, 1) for q = N(1.5, 0.5²).** (3 marks) *Model answer:* ½(μ² + σ² − ln σ² − 1) = ½(2.25 + 0.25 − (−1.386) − 1) = ½(2.886) = **1.44 nats**.
2. **Explain why the ELBO is a lower bound on log p(x) and what the gap represents.** (4 marks) *Model answer:* log p(x) = ELBO + KL(q(z|x) ‖ p(z|x)); KL ≥ 0 so ELBO ≤ log p(x) (2). The gap is the KL between the encoder's approximate posterior and the true posterior; for a fixed decoder, a closer encoder approximation makes it smaller (2).
3. **Why are VAE samples often blurry, and name two ways to reduce this.** (4 marks) *Model answer:* overlapping latent codes make the decoder average several compatible images; pixel-wise likelihoods (BCE/MSE) reward the mean prediction under uncertainty (2). Remedies: hierarchical latents or more expressive decoders; perceptual or adversarial losses (as in latent-diffusion autoencoders); lower β; convolutional architectures (2 for any two, justified).
4. **Describe how you would use a VAE for anomaly detection and one limitation.** (4 marks) *Model answer:* train on normal data only; score new inputs by reconstruction error (or negative ELBO); choose a threshold on a validation set; flag inputs above it (3). Limitation: subtle anomalies can reconstruct well, or under-represented normal cases get flagged; needs realistic evaluation (1).

## Exam-style question

**"Derive the evidence lower bound for a variational autoencoder, explain the role of each term, and discuss how the reparameterisation trick enables training. Illustrate with the effect of the β parameter."** (20 marks)

*Marking guide:* latent variable model and a likelihood that is difficult to calculate exactly (3); derivation via Jensen's inequality or the KL identity (5); interpretation of reconstruction and KL terms including their tension (4); reparameterisation trick with equation and gradient argument (4); β trade-off with a concrete consequence for samples or reconstructions (4). Excellent answers mention the gap as a KL to the true posterior, the closed-form Gaussian KL, and posterior collapse.

# Glossary

| Term | Meaning |
|---|---|
| Autoencoder | Encoder–decoder network trained to reconstruct its input through a bottleneck |
| Latent space | The space of codes z; in a VAE, shaped to match the prior |
| Prior p(z) | Fixed distribution over codes, usually N(0, I) |
| Encoder $q_\phi(z \mid x)$ | Network outputting a Gaussian over codes for an input; approximates the posterior |
| Decoder $p_\theta(x \mid z)$ | Network mapping a code to a distribution over data |
| ELBO | Evidence lower bound: reconstruction term minus KL; maximised in training |
| KL divergence | Non-negative, asymmetric measure of difference between distributions |
| Reparameterisation trick | Writing z = μ + σ ⊙ ε so sampling becomes differentiable |
| Posterior collapse | Failure where KL → 0 and the decoder ignores the code |
| β-VAE | VAE with the KL weighted by β, trading reconstruction for structure |
| VQ-VAE | VAE with a discrete codebook of latent vectors (image/audio tokens) |
| Latent interpolation | Decoding points on a path between two codes |
| Latent diffusion | Diffusion model operating in a VAE's latent space |

# Readings, videos and further practice

## Core reading (descriptor 7.9)

* Foster, D. (2023) *Generative Deep Learning*, 2nd ed.: chapter 3, "Variational Autoencoders" (autoencoders, VAEs, latent space exploration; Keras code).

**Papers**

* Kingma & Welling (2013) [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): the original VAE and the reparameterisation trick.
* Higgins et al. (2017) [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl): disentanglement with β > 1.
* van den Oord et al. (2017) [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937).
* Rombach et al. (2022) [Latent Diffusion Models](https://arxiv.org/abs/2112.10752): preview of week 4.

**Tutorials and explainers**

* [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/): derivations and variants.
* [TensorFlow tutorial: Convolutional VAE](https://www.tensorflow.org/tutorials/generative/cvae): the same model in TensorFlow (the descriptor allows either framework).

# Link to the group project

Groups that generate images, detect anomalies or need synthetic data can reuse this week's model directly. Ask students to note (1) whether their project needs a latent space (for search, interpolation or compression), (2) which evaluation from the table above suits their use case, and (3) any privacy risk if they plan to generate synthetic records.


# Detailed explanations behind the short slides

The lecture shows the essential idea first. These fuller explanations retain the supporting comparisons, examples and checks for revision.

## Slide 6: An autoencoder: squeeze, then rebuild

- Encoder: turn image pixels x into a shorter code z.

- Decoder: turn z back into a reconstructed image x-hat.

- Training reduces the difference between original and reconstructed pixels.

- Uses include compression, removing noise and learning useful features.

## Slide 8: Consequence: random codes decode to nonsense in an autoencoder

The same 16 random points z ~ N(0, I) decoded by each model.

A good generator needs its chosen random codes to reach regions the decoder has learned to use.

## Slide 9: Autoencoder vs. variational autoencoder

A prior is a chosen distribution for codes. The VAE learns to reconstruct while keeping codes near it.

**Autoencoder.** 

- Encode each input as one point z

- Train using reconstruction error

- The code locations can have gaps or an arbitrary scale

- Useful for compression and features

- Random codes may decode poorly

**Variational autoencoder.** 

- Encode each input as a Gaussian distribution: a centre $\mu$ and spread $\sigma$

- Train using reconstruction error plus a KL penalty

- Encourage codes to match a known prior, N(0, I)

- Draw a code from the prior and decode a new image

- Sample quality still depends on data, model and training

## Slide 12: The VAE as a probabilistic model

- To generate, sample from the prior and use the decoder. No input image is needed.

- To train, the encoder proposes useful codes for each known image.

- $\theta$ names the decoder weights and $\phi$ the encoder weights. Both learn together.

**Prior.** Draw code z from N(0, I), a Gaussian centred at zero

**Decoder.** Predict pixel probabilities from the code z

**Data.** Observe image x, such as a handwritten digit

**Encoder.** Estimate which codes could explain image x

## Slide 14: Why we cannot maximise the likelihood directly

- To calculate p(x), we would need to combine the decoder's predictions over every possible code z.

- For a neural decoder, this integral and the exact posterior p(z | x) are too difficult to calculate exactly.

- The encoder q(z | x) estimates the posterior. We maximise a quantity below log p(x): a **lower bound**.

## Slide 15: The evidence lower bound (ELBO)

- Reconstruction reward: sampled codes should help the decoder rebuild the input.

- KL penalty: keep encoder distributions near the prior N(0, I). KL measures a difference between distributions.

- The gap is a nonnegative KL, so ELBO cannot exceed log p(x). Minimise negative ELBO to train.

## Slide 16: Reading the ELBO: two forces in tension

The loss balances image detail with useful random codes. $\beta$ sets the KL weight.

**Reconstruction term.** 

- Keep enough image information in the code

- Reward accurate reconstructions

- Use pixel cross-entropy or squared error

**KL term.** 

- Keep each code distribution close to N(0, I)

- Encourage overlap between code distributions

- Too much pressure can make the decoder ignore the input

## Slide 17: KL divergence between two Gaussians has a closed form

- Calculate this cost for each code dimension, then add the costs.

- For $\mu$ = 0 and $\sigma$ = 1, the encoder matches the prior and KL = 0.

- Lab code: `kl = -0.5 * sum(1 + logvar - mu**2 - logvar.exp())`. Here logvar means log variance.

## Slide 18: Worked examples: how far is the encoder from the prior?

- Shift the centre: $\mu$ = 1, $\sigma$ = 1 gives KL = 0.5 × (1 + 1 − 0 − 1) = 0.50.

- Shrink the spread: $\mu$ = 0, $\sigma$ = 0.5 gives KL about 0.32.

- Do both: $\mu$ = 2, $\sigma$ = 0.5 gives KL about 2.32. Add the penalties across dimensions.

## Slide 19: The reconstruction term in practice

- For MNIST intensities in [0, 1], we use binary cross-entropy summed over the 784 pixels.

- Binary observations give a Bernoulli likelihood; greyscale intensities use a soft-target surrogate. A fixed-variance Gaussian likelihood instead gives squared error, up to constants and scale.

- Use one sampled code per image in each step to estimate the reconstruction expectation.

## Slide 22: The reparameterisation trick

- Draw fresh standard-normal noise $\epsilon$, then set z = $\mu$ + $\sigma$ × $\epsilon$.

- For each draw, noise is an input. The calculation remains differentiable with respect to $\mu$ and $\sigma$.

- Reconstruction gradients can train both encoder branches. Draw fresh noise for later examples.

## Slide 24: A complete VAE in PyTorch

- The encoder predicts `mu` and `logvar`, meaning mean and log variance.

- `randn_like` draws noise. Shift and scale it to make code z.

- Loss: summed pixel cross-entropy + $\beta$ × KL, then average over the batch.

- To generate, decode a fresh random-normal code and apply sigmoid.

## Slide 25: Training in practice: watch both terms

- This run uses Adam, learning rate 0.001, batch 128 and 10 epochs. Runtime depends on hardware.

- Track reconstruction and KL separately. The curves show what changed in this run.

- Posterior collapse means the decoder largely ignores z, often with KL near zero.

- Possible remedies: increase KL weight gradually, allow a small free KL amount or reduce decoder capacity.

## Slide 28: Decoding a grid of latent points: a map of the learned space

- Each cell is one decoded point from a 15 × 15 grid of 2-D codes.

- Nearby points often make similar digits: the output changes smoothly.

- Many grid points make digit-like images in this run. This does not guarantee every code is realistic.

- Look for changes in shared strokes: along the bottom row a 0 becomes a 6, then a 5.

## Slide 29: Interpolation: walk in a straight line through latent space

- Encode two images as $z_1$ and $z_2$. Decode $(1 - t) z_1 + t z_2$ as $t$ moves from 0 to 1.

- At t = 0.5, the code is halfway between them. Inspect the resulting image.

- Spherical interpolation (slerp) follows a curved path and can help for high-dimensional random codes.

## Slide 30: β changes the balance between sharpness and structure

- $\beta$ multiplies the KL penalty. The images compare three $\beta$ values.

- Low $\beta$ puts more weight on reconstruction. Prior samples may then be less reliable.

- High $\beta$ pulls codes closer to the prior. Reconstructions may lose detail.

- With $\beta$ = 1 and a decoder likelihood, this is the ELBO objective; our greyscale BCE uses a surrogate. Compare costs and samples.

## Slide 31: How to evaluate a VAE

| What | Metric | Watch out for |
|---|---|---|
| Fit to data | Negative ELBO on held-out data (nats, or bits per dimension) | Only a bound; compare models with the same likelihood setup |
| Reconstruction | BCE or MSE on held-out images; side-by-side pictures | Low error can coexist with poor samples |
| Sample quality | FID against real images (week 3); human rating | FID needs thousands of samples; humans need a rubric |
| Latent space | Interpolations, traversals, disentanglement scores | Visual inspection is subjective; report seeds |
| Usefulness | Downstream task, e.g. anomaly-detection ROC AUC | Test on realistic, held-out anomalies |

## Slide 33: Important VAE variants

**Conditional VAE.** Give the encoder and decoder a class label y, so the model can generate a selected class.

**β-VAE.** Increase $\beta$ above 1 to encourage separate factors in the codes. This can cost image detail.

**VQ-VAE.** Choose codes from a learned table, called a codebook. Images become grids of discrete tokens.

**Hierarchical VAEs.** Use several layers of hidden codes to describe data at different scales.

## Slide 34: Where you meet VAEs every day: inside latent diffusion

- The slide shows text-to-image generation in this Stable Diffusion design. The latent holds 48 times fewer numbers than the image.

- The VAE encoder does not appear in that path. It turns training photos into latents and encodes an existing picture for editing.

- Its autoencoder also uses visual and adversarial losses to help preserve detail. It differs from our simple digit VAE.

**Random latent.** Start from 64 × 64 × 4 random noise values

**Diffusion model.** Remove the noise step by step, guided by the text prompt (week 4)

**VAE decoder.** Turn the final latent into image pixels

**Image.** 512 × 512 × 3 pixel values

## Slide 35: Applications of VAEs

**Anomaly detection.** Use high reconstruction error as a possible warning for unusual inputs. Test real anomalies.

**Synthetic data.** Generate extra tabular, image or sensor examples. Check for copies and unfair patterns.

**Molecule and design search.** Search through hidden codes, then decode candidate molecules or shapes.

**Compression & tokens.** Compress images, audio or video into smaller numerical representations.

## Slide 37: Responsible AI lens: VAEs and synthetic data

- **Privacy**: synthetic data can memorise and leak training records; test for near-copies and consider differential privacy.

- **Bias**: an anomaly detector flags whatever differs from its training data, which can mean under-represented groups. Ask: normal for whom?

- **Consent & licences**: MNIST is public; faces, medical scans and voices need consent and a lawful basis.

- **Honest reporting**: show failure cases and blurry samples, not only the best interpolations.

## Slide 38: Lab 2: implement and explore a VAE in PyTorch

Google Colab or a local CPU; timings vary with hardware

Notebook with completed TODOs, figures, the β comparison table and four written answers

- Load MNIST digit images. Implement the encoder, code sampling and decoder.

- Implement reconstruction cost plus $\beta$ times KL, the usual VAE objective; a literal ELBO requires an appropriate decoder likelihood.

- Train for 10 epochs and plot reconstruction and KL costs separately.

- Plot 2-D codes. Decode a grid and points between two digit codes.

- Train with $\beta$ = 0.1, 1 and 4. Compare fixed-code samples and measured losses.

- Compare digit and clothing reconstruction errors. Compute ROC AUC. Optional: add a class condition.

## Slide 39: Summary

- An autoencoder compresses and rebuilds. Random codes may fall in poorly learned regions.

- A VAE encoder predicts a distribution of codes for each input.

- Loss = reconstruction cost + $\beta$ × KL. At $\beta$ = 1 this is the standard VAE objective; binary targets give the literal negative ELBO.

- Gaussian KL: 0.5 × (μ² + σ² − log σ² − 1) per dimension.

- Sample with z = $\mu$ + $\sigma$ × $\epsilon$, using fresh random-normal noise.

- Inspect reconstructions, random samples and paths. Evaluate on held-out data.

- $\beta$ changes the balance between reconstruction and the prior.

- Conditional VAEs, β-VAEs, VQ-VAEs and hierarchical VAEs extend the basic idea.


## Full implementation from the lecture

The projected slide shows the `encode`, `reparameterise` and `forward` methods. The complete class, including the layers, `decode` and `vae_loss`, is listed in section 3 under "A complete PyTorch model" and matches the lab solution.
