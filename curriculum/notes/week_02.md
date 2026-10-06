# Lecture plan at a glance

This week introduces the first deep generative model family of the module. Students should leave able to write the ELBO, explain each term, compute a Gaussian KL divergence, and implement a VAE. All result images in the slides come from real models trained with the lab code (see `curriculum/assets/make_week_02.py`), so the lab reproduces what students saw in the lecture.

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–7 min | 1–4 | Warm-up | "Describe a face with 32 numbers": motivates codes, generation and gaps. |
| 7–27 min | 5–10 | 1 · From autoencoders to VAEs | Autoencoder recap; real latent plots; random codes decode to nonsense; AE vs. VAE. Quiz. |
| 27–60 min | 11–20 | 2 · Model and ELBO | Probabilistic model; architecture; intractable likelihood; ELBO; KL closed form with worked numbers; reconstruction term. Quiz. |
| 60–67 min | – | Break | |
| 67–84 min | 21–25 | 3 · Reparameterisation | The trick, why it matters, the PyTorch model, training curves and posterior collapse. |
| 84–104 min | 26–31 | 4 · Exploring and evaluating | Reconstructions, latent grid, interpolation, β trade-off, evaluation table. |
| 104–118 min | 32–37 | 5 · Variants and applications | CVAE, β-VAE, VQ-VAE, hierarchical; VAE inside latent diffusion; applications; anomaly demo; responsible AI. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Derive the ELBO on the board (see the derivation below) rather than only showing the slide. Students who see the two-line Jensen derivation remember that the gap is a KL divergence, which is a common exam question.

<!-- pagebreak -->

# Lecture notes

## 1. From autoencoders to VAEs

### Autoencoders

An **autoencoder** is a pair of networks trained together. The **encoder** maps an input $x$ (for MNIST, 784 pixel intensities) to a short **code** $z$ (the bottleneck, perhaps 2–32 numbers); the **decoder** maps $z$ back to a reconstruction $\hat{x}$. Training minimises a reconstruction loss such as mean squared error or binary cross-entropy. Because the code is small, the network must learn the structure of the data rather than copy it.

Autoencoders are useful for **compression**, **denoising** (train to reconstruct clean images from corrupted ones), **representation learning** (use $z$ as features for another model) and **anomaly detection** (unusual inputs reconstruct badly).

![An autoencoder compresses to a code and rebuilds the input](fig:ae_diagram)

### Why an autoencoder is a poor generator

Nothing in the autoencoder's training says where codes should lie. The encoder can place each class anywhere, at any scale, with large empty regions between clusters, because the decoder only ever needs to handle codes the encoder actually produces. If we pick a random code, for example from a standard Gaussian, we often land in an empty region where the decoder was never trained, and it outputs a smudge.

![Real latent spaces of an autoencoder and a VAE with 2-D codes](fig:latent_ae_vs_vae)

![The same random codes decoded by each model](fig:samples_ae_vs_vae)

For generation we need a latent space that is:

* **continuous:** nearby codes decode to similar outputs; and
* **complete:** every code we might sample decodes to something realistic.

A **variational autoencoder (VAE)** gets both properties with two changes: the encoder outputs a probability distribution $q(z \mid x)$ rather than a point, and the loss adds a **KL divergence** that pulls every such distribution towards a fixed **prior** $p(z) = \mathcal{N}(0, I)$.

> **Key idea:** A VAE is an autoencoder whose code is a probability distribution, regularised to look like a simple prior, so that sampling from the prior and decoding generates new data.

## 2. The VAE model and the ELBO

### The probabilistic model

A VAE is a **latent variable model**:

1. **Prior:** $z \sim p(z) = \mathcal{N}(0, I)$.
2. **Decoder (likelihood):** $x \sim p_\theta(x \mid z)$, where a neural network with parameters $\theta$ outputs the parameters of the distribution over $x$ (for MNIST, one Bernoulli probability per pixel).
3. **Encoder (approximate posterior):** $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))$, a neural network with parameters $\phi$ that guesses which codes could have produced $x$.

**Generation** needs only the prior and decoder. **Training** also needs the encoder, because to learn the decoder we must know which codes explain each training image. The word *variational* comes from **variational inference**: approximating an intractable distribution (the true posterior) with a simpler parameterised family (Gaussians whose parameters come from the encoder).

![VAE architecture](fig:vae_arch)

### Why the likelihood is intractable

The likelihood of an image is an average of the decoder over all codes:

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$

With a neural network inside, this integral has no closed form. Estimating it by sampling codes from the prior is hopeless: almost every random code explains a given image extremely badly, so an enormous number of samples would be needed. The true posterior $p_\theta(z \mid x) = p_\theta(x \mid z)\,p(z) / p_\theta(x)$ contains the same integral and is intractable too.

### Deriving the evidence lower bound (ELBO)

Multiply and divide by the encoder distribution inside the integral, then apply **Jensen's inequality** ($\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$ because log is concave):

$$\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z \mid x)}\left[\frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right] \geq \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right]$$

Splitting the logarithm gives the familiar form:

$$\mathrm{ELBO}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - D_{\mathrm{KL}}\left(q_\phi(z \mid x)\,\|\,p(z)\right)$$

An exact identity shows what the bound loses:

$$\log p_\theta(x) = \mathrm{ELBO}(\theta, \phi; x) + D_{\mathrm{KL}}\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)$$

Because KL divergences are never negative, the ELBO is always below $\log p_\theta(x)$, and the gap is exactly how far the encoder's approximation is from the true posterior. Maximising the ELBO over $\theta$ and $\phi$ therefore (i) pushes the likelihood up and (ii) improves the encoder. In practice we **minimise the negative ELBO**, averaged over a mini-batch.

### Reading the ELBO: two forces

* **Reconstruction term** $\mathbb{E}_q[\log p_\theta(x \mid z)]$: codes drawn from the encoder must let the decoder rebuild $x$. Alone, it produces a plain autoencoder (variances shrink to zero, codes spread apart).
* **KL term** $D_{\mathrm{KL}}(q_\phi(z \mid x)\,\|\,p(z))$: every image's code distribution should resemble the prior. Alone, it makes the code ignore the input.

Training finds a compromise: codes carry enough information to reconstruct, but overlap enough to fill the space. The overlap is also a main reason VAE samples are slightly **blurry**: a code is compatible with several similar images and the decoder outputs their average.

### The KL divergence between Gaussians

For a one-dimensional Gaussian encoder and a standard normal prior:

$$D_{\mathrm{KL}}\left(\mathcal{N}(\mu, \sigma^2)\,\|\,\mathcal{N}(0, 1)\right) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right)$$

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

Moving the mean away from zero and making the distribution either too narrow or too wide both cost nats. KL is **not symmetric**: $D_{\mathrm{KL}}(q\|p) \neq D_{\mathrm{KL}}(p\|q)$ in general.

![KL divergence for the worked examples](fig:kl_gaussians)

### The reconstruction term

The reconstruction term becomes a standard loss once we choose the decoder distribution:

* **Bernoulli decoder** (pixels in [0, 1], e.g. MNIST): negative log-likelihood = **binary cross-entropy** summed over the 784 pixels:

$$-\log p_\theta(x \mid z) = -\sum_{i=1}^{784}\left[x_i \log \hat{x}_i + (1 - x_i)\log(1 - \hat{x}_i)\right]$$

* **Gaussian decoder** with fixed variance $\sigma^2_x$ (real-valued data): negative log-likelihood = $\|x - \hat{x}\|^2 / (2\sigma^2_x)$ + constant, i.e. **squared error**. The choice of $\sigma^2_x$ implicitly sets the balance with the KL term.

The expectation over $q$ is estimated with **one sampled code** per image per step (a Monte Carlo estimate); across many steps the noise averages out.

> **Common mistake:** averaging the BCE over pixels (PyTorch's default `reduction="mean"`) while summing the KL over dimensions. The KL term then dominates by a factor of 784 and every sample looks like an average blurry digit. Sum over pixels, then divide the total by the batch size.

## 3. Training: the reparameterisation trick

We need gradients of $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$ with respect to the encoder parameters $\phi$, but the expectation is over a distribution that depends on $\phi$, and a random draw is not differentiable. The **reparameterisation trick** rewrites the sample:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

The distribution of $z$ is unchanged, but the randomness now enters through $\epsilon$, which does not depend on $\phi$. So $z$ is a differentiable function of $\mu$ and $\sigma$, and ordinary backpropagation works. Without the trick, one would need the score-function (REINFORCE) estimator, which is unbiased but has much higher variance. The same "clean value plus scaled noise" form reappears in diffusion models (week 4).

![Without and with the reparameterisation trick](fig:reparam)

### A complete PyTorch model

```python
class VAE(nn.Module):
    def __init__(self, zdim=2, h=400):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h, zdim), nn.Linear(h, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, h), nn.ReLU(), nn.Linear(h, 784))

    def forward(self, x):
        hid = self.enc(x)
        mu, logvar = self.mu(hid), self.logvar(hid)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z), mu, logvar

def vae_loss(logits, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return (rec + beta * kl) / x.size(0)
```

The network outputs the **log-variance** because it can take any real value; exponentiating guarantees a positive variance. To display a generated image: `torch.sigmoid(model.dec(torch.randn(n, zdim)))`. This samples a latent code but displays the decoder's pixel probabilities; it does not draw each pixel from its Bernoulli distribution. In the lab, `VAE.encode` contains the mean/log-variance TODO and `VAE.decode` handles displayable output.

### Training practice

* Adam with learning rate 10⁻³, batch size 128; 10 epochs take about a minute on a Colab T4 GPU and a few minutes on a laptop CPU for this small model.
* **Always log reconstruction and KL separately.** The KL usually rises early in training: an untrained encoder outputs codes near the prior (KL ≈ 0), and as codes begin to carry information the KL grows.
* **Posterior collapse:** the KL falls to (almost) zero and the decoder ignores $z$. It is common when the decoder is powerful (e.g. autoregressive). Remedies: **KL warm-up** (increase β from 0 to 1 over the first epochs), **free bits** (no penalty below a minimum KL per dimension), or a less powerful decoder.

![Real training curves for the 2-D VAE](fig:training_curves)

## 4. Exploring and evaluating the latent space

### Reconstructions and blur

VAE reconstructions of unseen digits are recognisable but softer than the originals. Two causes: overlapping codes mean the decoder averages over compatible images, and pixel-wise likelihoods reward predicting average intensity under uncertainty. GANs (week 3) avoid pixel-wise losses; latent diffusion models (week 4) combine a VAE-style compressor, trained with extra perceptual and adversarial losses, with a diffusion model.

![Test images and their reconstructions](fig:reconstructions)

### Latent grid and interpolation

With a 2-D latent space, decoding a regular grid (spaced with Gaussian quantiles so that it covers the prior evenly) shows a continuous map of the data: neighbouring points decode to similar digits and every point decodes to something digit-like. In higher dimensions, use **interpolation**: encode two images to their means $z_1, z_2$, decode points $z_t = (1 - t) z_1 + t z_2$. Realistic intermediate images show that straight lines in latent space follow the data manifold, unlike pixel blending. For Gaussian latents in high dimension, **spherical linear interpolation (slerp)** often looks better, because random Gaussian vectors concentrate near a sphere and the straight chord passes through low-probability space.

![Decoded grid of the 2-D latent space](fig:manifold)

![Latent interpolation between digits](fig:interpolation)

### The β trade-off

Weighting the KL term by β gives the **β-VAE** objective $\mathbb{E}_q[\log p_\theta(x \mid z)] - \beta\, D_{\mathrm{KL}}(q_\phi(z \mid x)\,\|\,p(z))$. β = 1 is the true ELBO. Measured on our 16-D models after 5 epochs:

| β | Reconstruction (BCE per image) | KL (nats per image) | Samples from the prior |
|---|---|---|---|
| 0.1 | 77.6 | 49.7 | sharp details but many malformed digits |
| 1.0 | 87.9 | 22.8 | good balance (true ELBO) |
| 4.0 | 119.3 | 8.6 | well-formed but blurry, generic |

As β grows the KL (the **rate**: information carried by the code) falls and reconstruction error (the **distortion**) rises. Small β gives sharp reconstructions but a latent distribution that does not match the prior, so prior samples are often malformed; large β gives smooth, generic samples. β > 1 is also used to encourage **disentangled** latent dimensions (each dimension capturing one factor, such as stroke width or slant).

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
* **β-VAE:** β > 1 for disentanglement; trades reconstruction quality for interpretable factors.
* **VQ-VAE:** replaces the Gaussian code with the nearest vector in a learned **codebook**, so images (or audio) become grids of **discrete tokens**. This is the basis of many image and audio tokenizers used by autoregressive and multimodal transformers.
* **Hierarchical VAEs** (e.g. NVAE): several layers of latents; much sharper images than a single code.

### VAEs inside latent diffusion

Latent diffusion models (Rombach et al., 2022), including Stable Diffusion, first train an autoencoder that compresses a 512 × 512 × 3 image (786,432 numbers) into a 64 × 64 × 4 latent (16,384 numbers, 48× fewer). Diffusion then operates in the latent space, guided by the text prompt, and the decoder maps the final latent to pixels. These autoencoders use a small KL weight plus perceptual and adversarial losses to keep reconstructions sharp. The VAE is therefore present in a very large share of today's image and video generators, even though pure VAEs are rarely used as standalone image generators.

### Applications

* **Anomaly detection:** inputs unlike the training data reconstruct badly (manufacturing defects, network intrusions, unusual medical signals).
* **Synthetic data:** sample new records or images to augment small datasets (with privacy caveats).
* **Molecule and design search:** optimise in a smooth latent space, then decode candidates.
* **Compression and tokens:** learned codecs; latent spaces for diffusion and multimodal models.

In our demonstration, a VAE trained only on digits separated unseen digits from Fashion-MNIST clothing images using reconstruction error with **ROC AUC = 1.000**. Real anomalies are much subtler; always evaluate on realistic, held-out cases.

![Reconstruction error for digits and clothing](fig:anomaly)

# Common misconceptions

| Misconception | Correction |
|---|---|
| "A VAE is just an autoencoder with noise added." | The noise comes from a learned distribution, and the KL term to a prior is what makes the latent space usable for generation. |
| "The ELBO is the log-likelihood." | It is a lower bound; the gap equals the KL between the encoder and the true posterior. |
| "Lower KL is always better." | KL = 0 means the code carries no information about the input; the balance with reconstruction matters. |
| "Blurry samples mean a bug." | Some blur is expected (overlapping codes, pixel-wise likelihood). Severe blur often means the loss terms are mis-scaled (mean vs. sum). |
| "The encoder is needed to generate." | Generation uses only the prior and the decoder; the encoder is used in training and for encoding real images. |
| "Interpolation that looks good proves the model is good." | Cherry-picked pairs can mislead; evaluate quantitatively and show failures. |
| "VAEs are obsolete." | Their autoencoders power latent diffusion; VQ-VAE ideas power image and audio tokenizers. |

# Responsible AI lens: VAEs and synthetic data

* **Privacy:** synthetic data is not automatically anonymous. Generative models can memorise rare records. Check for near-duplicates of training data and consider differential privacy for sensitive data.
* **Bias in anomaly detection:** the detector flags whatever differs from its training data. If a group is under-represented, its normal cases look anomalous (for example in fraud or clinical screening). Ask "normal for whom?" and measure error rates across groups.
* **Consent and licences:** MNIST is public; faces, medical images and voices require consent, a lawful basis (GDPR) and appropriate licences.
* **Honest reporting:** report typical samples, failure cases, seeds and metrics, not only the best interpolation.

# Lab guide and answers

**Runtime:** Colab T4 (about 1 minute per 10-epoch run) or CPU (about 5 minutes). MNIST and Fashion-MNIST download automatically through torchvision. **Hand-in:** completed TODOs, figures, the β table and four written answers.

## TODOs and expected results

* **TODO 1** in `VAE.encode`: `mu, logvar = self.mu_head(h), self.logvar_head(h)`.
* **TODO 2** `std = torch.exp(0.5 * logvar); return mu + std * torch.randn_like(std)`.
* **TODO 3** BCE with logits summed over pixels; KL `-0.5 * sum(1 + logvar - mu**2 - exp(logvar))`; divide by batch size.
* **Question 1:** untrained reconstruction ≈ 784 × ln 2 ≈ 543 nats because every pixel is predicted with probability 0.5; KL ≈ 0 because the encoder initially outputs μ ≈ 0, σ ≈ 1.
* **Training (2-D model, 10 epochs):** reconstruction falls to about 147 nats per image and the KL settles near 5.8 nats (measured with the lecture model).
* **Part 3:** the scatter shows class clusters packed around the origin; the grid decodes to a smooth map of digits. **TODO 4** `(1 - ts) * z1 + ts * z2`. Latent interpolation gives realistic intermediate digits; pixel blending gives ghostly overlays.
* **Part 4 (β):** expect the pattern in the table above; exact values vary with seeds and epochs.
* **TODO 5** per-image BCE: `F.binary_cross_entropy_with_logits(logits, x.view(-1, 784), reduction="none").sum(dim=1)`. Expect ROC AUC close to 1.000.
* **Extension (CVAE):** each row should show the requested digit in varied styles.

## Troubleshooting

* *Samples all look like the same blurry blob:* the BCE was averaged instead of summed, so the KL dominates.
* *NaN loss:* usually `logvar` exploding; lower the learning rate or clamp `logvar` to [−10, 10].
* *Very slow on CPU:* reduce epochs to 3 for the β study; results remain qualitatively the same.

# Practice questions with model answers

## Multiple choice

1. The KL term in the VAE loss: **(a)** measures reconstruction quality; **(b)** keeps each encoder distribution close to the prior; **(c)** is the log-likelihood; **(d)** is only used at generation time. *Answer: (b).*
2. The reparameterisation trick is needed because: **(a)** the decoder is non-linear; **(b)** we cannot backpropagate through a random sampling step; **(c)** the prior is not Gaussian; **(d)** MNIST is binary. *Answer: (b).*
3. Increasing β from 1 to 4 usually: **(a)** sharpens reconstructions; **(b)** increases the KL; **(c)** lowers the KL and increases reconstruction error; **(d)** has no effect. *Answer: (c).*
4. In latent diffusion models the VAE is used to: **(a)** classify images; **(b)** compress images into a smaller latent space where diffusion runs; **(c)** encode the text prompt; **(d)** compute FID. *Answer: (b).*

## Short answer

1. **Compute the KL divergence to N(0, 1) for q = N(1.5, 0.5²).** (3 marks) *Model answer:* ½(μ² + σ² − ln σ² − 1) = ½(2.25 + 0.25 − (−1.386) − 1) = ½(2.886) = **1.44 nats**.
2. **Explain why the ELBO is a lower bound on log p(x) and what the gap represents.** (4 marks) *Model answer:* log p(x) = ELBO + KL(q(z|x) ‖ p(z|x)); KL ≥ 0 so ELBO ≤ log p(x) (2). The gap is the KL between the encoder's approximate posterior and the true posterior; it shrinks as the encoder improves (2).
3. **Why are VAE samples often blurry, and name two ways to reduce this.** (4 marks) *Model answer:* overlapping latent codes make the decoder average several compatible images; pixel-wise likelihoods (BCE/MSE) reward the mean prediction under uncertainty (2). Remedies: hierarchical latents or more expressive decoders; perceptual or adversarial losses (as in latent-diffusion autoencoders); lower β; convolutional architectures (2 for any two, justified).
4. **Describe how you would use a VAE for anomaly detection and one limitation.** (4 marks) *Model answer:* train on normal data only; score new inputs by reconstruction error (or negative ELBO); choose a threshold on a validation set; flag inputs above it (3). Limitation: subtle anomalies can reconstruct well, or under-represented normal cases get flagged; needs realistic evaluation (1).

## Exam-style question

**"Derive the evidence lower bound for a variational autoencoder, explain the role of each term, and discuss how the reparameterisation trick enables training. Illustrate with the effect of the β parameter."** (20 marks)

*Marking guide:* latent variable model and intractable likelihood (3); derivation via Jensen's inequality or the KL identity (5); interpretation of reconstruction and KL terms including their tension (4); reparameterisation trick with equation and gradient argument (4); β trade-off with a concrete consequence for samples or reconstructions (4). Excellent answers mention the gap as a KL to the true posterior, the closed-form Gaussian KL, and posterior collapse.

# Glossary

| Term | Meaning |
|---|---|
| Autoencoder | Encoder–decoder network trained to reconstruct its input through a bottleneck |
| Latent space | The space of codes z; in a VAE, shaped to match the prior |
| Prior p(z) | Fixed distribution over codes, usually N(0, I) |
| Encoder q_φ(z \| x) | Network outputting a Gaussian over codes for an input; approximates the posterior |
| Decoder p_θ(x \| z) | Network mapping a code to a distribution over data |
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
