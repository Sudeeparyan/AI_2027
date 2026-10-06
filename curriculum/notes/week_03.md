# Lecture plan at a glance

GANs are the second deep generative family. Students should leave able to write the minimax objective, derive the optimal discriminator, explain the non-saturating loss, recognise the failure modes and explain FID. All sample images in the slides come from real models trained with the lab code (`curriculum/assets/make_week_03.py`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–6 min | 1–4 | Warm-up | Forger and detective; collect three answers (feedback, detective too strong, one repeated trick). |
| 6–21 min | 5–9 | 1 · Why adversarial? | Learned loss for realism; architecture; roles; analogy and its limits. |
| 21–51 min | 10–16 | 2 · The objective | Minimax, optimal D and JSD (derive on the board), non-saturating loss, training step code. Quiz. |
| 51–58 min | – | Break | |
| 58–80 min | 17–23 | 3 · Challenges and fixes | Real curves; mode collapse experiment; failure table; WGAN-GP; stabilisation; quiz. |
| 80–102 min | 24–30 | 4 · Variants | DCGAN architecture and real samples; conditional GAN; pix2pix, CycleGAN, StyleGAN, BigGAN; cycle loss; interpolation. |
| 102–118 min | 31–37 | 5 · Evaluation, applications, risks | FID and metrics table; GANs in 2026; deepfakes; VAE vs GAN summary. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** The derivation of the optimal discriminator takes three lines on the board and is a very common exam question. Do it live rather than only showing the slide.

<!-- pagebreak -->

# Lecture notes

## 1. Why adversarial learning?

VAEs are trained with pixel-wise likelihoods. When the model is uncertain, the likelihood is maximised by predicting the **average** of the plausible images, which looks blurry. "Realism" is hard to write as a formula, but it can be **learned**: train a classifier to distinguish real images from generated ones, and use that classifier as the loss for the generator. This is the idea of a **generative adversarial network (GAN)** (Goodfellow et al., 2014).

* The **generator** $G$ maps noise $z \sim p(z)$ (typically $\mathcal{N}(0, I)$, 64–512 dimensions) to a sample $G(z)$.
* The **discriminator** $D$ maps a sample to the probability that it is real.
* $G$ never sees real data directly; it learns only from gradients passed back **through** $D$.

GANs are **implicit generative models**: we can sample from them, but they give no density $p(x)$. This rules out likelihood-based evaluation and anomaly detection, and is why GAN evaluation uses sample-based metrics such as FID.

![GAN architecture](fig:gan_arch)

> **Key idea:** A GAN replaces a hand-written loss ("be close pixel by pixel") with a learned, adaptive loss ("be indistinguishable from real data"). The price is that training becomes a two-player game rather than the minimisation of a fixed function.

## 2. The GAN objective

### The minimax game

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log\left(1 - D(G(z))\right)\right]$$

For the discriminator this is (the negative of) the binary cross-entropy of a real-vs-fake classifier with labels 1 for real and 0 for fake. The generator appears only in the second term.

### The optimal discriminator (derivation)

Write the expectation over fakes as an expectation over the generator's distribution $p_g$. For a fixed $G$:

$$V(D, G) = \int \left[p_{\mathrm{data}}(x) \log D(x) + p_g(x) \log(1 - D(x))\right] dx$$

For each $x$, the function $a \log y + b \log(1 - y)$ is maximised at $y = a/(a + b)$. Hence

$$D^{*}_G(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + p_g(x)}$$

Substituting back gives

$$V(D^{*}_G, G) = -\log 4 + 2\,\mathrm{JSD}(p_{\mathrm{data}} \,\|\, p_g)$$

where the **Jensen–Shannon divergence** $\mathrm{JSD}(p\|q) = \tfrac12 D_{\mathrm{KL}}(p\|m) + \tfrac12 D_{\mathrm{KL}}(q\|m)$ with $m = \tfrac12(p + q)$ is symmetric, non-negative and zero only when $p = q$. So, with an optimal discriminator and unlimited capacity, the generator minimises the JSD, and the global optimum is $p_g = p_{\mathrm{data}}$, where $D^{*} = \tfrac12$ everywhere and $V = -\log 4$.

![The optimal discriminator early in training and at equilibrium](fig:optimal_d)

In practice neither network has unlimited capacity, $D$ is never optimal, and the game is optimised with alternating stochastic gradient steps, so this is an idealisation that explains the goal rather than a guarantee.

### The non-saturating generator loss

Early in training $D(G(z)) \approx 0$. The minimax generator loss $\log(1 - D(G(z)))$ is almost flat there, so its gradient is tiny exactly when the generator is worst. Goodfellow et al. therefore proposed that the generator **maximise** $\log D(G(z))$, i.e. minimise $-\log D(G(z))$, which is steep near zero. Both objectives share the same fixed point. In code, this is binary cross-entropy with the fakes labelled as **real**.

![Minimax vs non-saturating generator loss](fig:saturating)

### One training step

```python
fake = G(torch.randn(b, z_dim))
loss_D = bce(D(x_real), ones) + bce(D(fake.detach()), zeros)   # D step
opt_D.zero_grad(); loss_D.backward(); opt_D.step()
loss_G = bce(D(fake), ones)                                    # G step (non-saturating)
opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

`.detach()` prevents the discriminator step from updating the generator. DCGAN defaults: Adam, learning rate 2 × 10⁻⁴, β₁ = 0.5.

## 3. Training challenges and fixes

### Reading the diagnostics

GAN losses **oscillate** instead of decreasing, because each network's objective moves whenever the other improves. A healthy run keeps $D(\text{real})$ somewhat above $D(\text{fake})$ with neither pinned at 0 or 1. If $D(\text{fake}) \to 0$ and the discriminator loss → 0, the discriminator has won and the generator's gradients are uninformative. **Judge progress from fixed-noise samples and metrics, not from losses.**

![Real training curves of the lecture DCGAN](fig:losses)

### Mode collapse

The generator covers only part of the data distribution: sharp but repetitive samples (for MNIST, only a few digit classes). It arises because producing whatever currently fools $D$ is rewarded; when $D$ adapts, $G$ may jump to another mode rather than spread out. On the 8-cluster ring experiment, the standard GAN covered 7 of 8 modes at the end of training while WGAN-GP covered 8 (same networks and steps).

![Mode collapse experiment on a ring of 8 Gaussians](fig:mode_collapse)

### Failure modes and remedies

| Problem | Symptom | Remedies |
|---|---|---|
| Mode collapse | Samples alike; classes missing from the histogram | Wasserstein loss (+GP), minibatch statistics in D, unrolled D, more diverse data |
| Vanishing gradients | D(fake) ≈ 0; G stops improving | Non-saturating loss; weaker or regularised D; Wasserstein loss |
| Non-convergence | Samples keep changing style; wild oscillation | Lower learning rates; TTUR; spectral normalisation; EMA of G |
| D overfits (small data) | D(real) → 1 on training images only | Discriminator augmentation (ADA); more data |
| Artefacts | Checkerboards, textures stuck to screen positions | Resize-then-convolve upsampling; StyleGAN2/3 architectures |

### The Wasserstein GAN with gradient penalty

The Wasserstein-1 (earth-mover) distance measures the minimum "work" to move probability mass from one distribution to the other. By Kantorovich–Rubinstein duality it equals $\sup_{\|f\|_L \leq 1} \mathbb{E}_{p_{\mathrm{data}}}[f(x)] - \mathbb{E}_{p_g}[f(x)]$, so a **critic** $D$ restricted to 1-Lipschitz functions can estimate it:

$$\min_G \max_{\|D\|_L \leq 1} \mathbb{E}_{x \sim p_{\mathrm{data}}}[D(x)] - \mathbb{E}_{z}[D(G(z))]$$

Unlike the JSD, it can vary usefully even when the two distributions do not overlap, giving the generator more informative directional feedback. WGAN-GP (Gulrajani et al., 2017) encourages the Lipschitz condition by adding to the critic's loss

$$\lambda\, \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right], \quad \hat{x} = \epsilon x + (1-\epsilon) G(z), \; \epsilon \sim U[0, 1]$$

with λ = 10 and typically 5 critic steps per generator step. This encourages gradient norms near 1 on sampled interpolates; it does not guarantee a globally 1-Lipschitz critic or complete mode coverage. The critic outputs an unbounded score, not a probability.

### Other stabilisation techniques

* **Spectral normalisation** (Miyato et al., 2018): divide each discriminator layer's weight by its largest singular value, a cheap way to bound the Lipschitz constant (`torch.nn.utils.spectral_norm`).
* **TTUR** (Heusel et al., 2017): different learning rates for D and G.
* **EMA of generator weights:** sample from an exponential moving average of G's parameters.
* **Adaptive discriminator augmentation (ADA):** augment images shown to D so it cannot memorise small datasets.
* **R1 regularisation:** penalise D's gradient on real data (default in StyleGAN2).

## 4. Variants: from DCGAN to StyleGAN

### DCGAN

Radford et al. (2015) made convolutional GANs train reliably: strided/transposed convolutions instead of pooling; batch normalisation in both networks (except G's output and D's input); ReLU in G with a tanh output; LeakyReLU in D; Adam with learning rate 2 × 10⁻⁴ and β₁ = 0.5. The lab model follows these rules for 28 × 28 images.

![DCGAN generator and guidelines](fig:dcgan_arch)

![Real DCGAN samples across epochs](fig:epochs)

### Conditional GANs and image-to-image translation

A **conditional GAN** (Mirza & Osindero, 2014) feeds a condition $y$ to both networks: $G(z, y)$ and $D(x, y)$. $D$ must check that $x$ is real **and** matches $y$, so $G$ learns to respect the condition.

* **pix2pix** (Isola et al., 2017): paired translation (sketch → photo, labels → street scene) with a conditional GAN plus an L1 reconstruction loss and a "PatchGAN" discriminator that judges local patches.
* **CycleGAN** (Zhu et al., 2017): unpaired translation (horse ↔ zebra) with two generators $G: X \to Y$, $F: Y \to X$ and a **cycle-consistency** loss:

$$\mathcal{L}_{\mathrm{cyc}} = \mathbb{E}_x\left[\|F(G(x)) - x\|_1\right] + \mathbb{E}_y\left[\|G(F(y)) - y\|_1\right]$$

![Conditional GAN](fig:cgan)

### StyleGAN and BigGAN

**StyleGAN** (Karras et al., 2019; StyleGAN2 2020; StyleGAN3 2021) maps $z$ through a mapping network to an intermediate style vector $w$ that modulates every layer of the generator, giving photorealistic faces and separate control of coarse (pose, face shape) and fine (hair, skin texture) attributes. Its well-organised latent space enabled attribute editing. **BigGAN** (Brock et al., 2018) scaled class-conditional GANs on ImageNet and popularised the **truncation trick**: sampling $z$ from a truncated normal trades diversity for quality.

### Latent interpolation and GAN inversion

Interpolating between noise vectors gives smooth transitions, evidence of a continuous mapping. In high dimensions, **spherical interpolation (slerp)** keeps intermediate vectors at a typical norm. GANs have **no encoder**; editing a real photo requires **GAN inversion**, i.e. optimising a latent vector so that $G$ reproduces the photo.

![Latent interpolation in the lecture DCGAN](fig:interpolation)

## 5. Evaluation, applications and risks

### Fréchet Inception Distance (FID)

Embed real and generated images with an Inception-v3 network (2048-dimensional pool features), fit a Gaussian to each set and compute the Fréchet distance:

$$\mathrm{FID} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)$$

Lower is better. FID penalises both unrealistic samples (means differ) and low diversity (covariances differ). Limitations: assumes Gaussian features; biased for small sample sizes (use ≥ 10,000 samples or KID); ImageNet features may not suit other domains; values are only comparable with identical feature networks, preprocessing and sample sizes.

**Worked example (1-D intuition).** If real features have mean 0 and variance 1 and generated features have mean 0.5 and variance 0.25, then FID = 0.5² + (1 + 0.25 − 2√(1 × 0.25)) = 0.25 + (1.25 − 1) = **0.50**. A collapsed generator with variance 0.01 and mean 0 gives 0 + (1 + 0.01 − 2 × 0.1) = **0.81**, worse despite a perfect mean.

![What FID measures in feature space](fig:fid)

**Lab distinction:** `frechet_distance` uses the same distribution-comparison formula with features from the lab's `DigitCNN`, not Inception-v3. Call its output a **custom feature Fréchet distance**. Its scale differs from standard Inception FID, so do not compare it with published FID values. Check the classifier's accuracy and keep its weights, preprocessing, and sample size fixed for comparisons within the lab.

### Other metrics

| Metric | Captures | Limitation |
|---|---|---|
| Inception Score (IS, higher is better) | Confident and varied class predictions: $\exp\left(\mathbb{E}_x D_{\mathrm{KL}}(p(y\mid x)\,\|\,p(y))\right)$ | Ignores the real data; fooled by class-diverse but unrealistic images |
| Precision / recall (Kynkäänniemi et al., 2019) | Fidelity and coverage separately | Sensitive to feature space and neighbourhood size |
| KID | Kernel version of FID; unbiased at small sample sizes | Less widely reported |
| Human evaluation | Realism, preference, task fit | Needs rubrics, multiple raters, agreement statistics |

### GANs in 2026

Diffusion models replaced GANs as the leading text-to-image generators, but GANs remain important where **speed** matters (one forward pass): super-resolution and restoration (e.g. Real-ESRGAN), **adversarial diffusion distillation** that compresses a many-step diffusion model into 1–4 steps (e.g. SDXL-Turbo), adversarial losses inside the **autoencoders and tokenizers** of latent diffusion models, **GAN vocoders** for speech synthesis (e.g. HiFi-GAN), image-to-image translation for simulation-to-real transfer and medical imaging, and synthetic tabular data.

### VAE vs GAN

| | VAE | GAN |
|---|---|---|
| Training signal | Maximise ELBO (likelihood bound) | Two-player game with a learned discriminator |
| Likelihood | Lower bound available | None |
| Encoder | Yes | No (needs inversion) |
| Samples | Smooth, often blurry | Sharp |
| Diversity | Good coverage | Risk of mode collapse |
| Training | Stable | Unstable; needs tricks |

# Common misconceptions

| Misconception | Correction |
|---|---|
| "A low generator loss means good images." | GAN losses are relative to the current discriminator; judge samples and metrics. |
| "The generator learns by looking at real images." | It only receives gradients through D; it never sees real data directly. |
| "At the end of training D should be perfect." | At the ideal equilibrium D outputs ½ everywhere; a perfect D means G has failed. |
| "Mode collapse means blurry images." | Mode collapse means low diversity; the images can be very sharp. |
| "FID is an absolute quality score." | FID values are only comparable under identical feature networks, preprocessing and sample sizes. |
| "GANs are obsolete." | They power super-resolution, fast distilled diffusion, sharp autoencoders and vocoders. |
| "WGAN's critic outputs a probability." | It outputs an unbounded score; the difference of average scores estimates the Wasserstein distance. |

# Responsible AI lens: deepfakes and synthetic media

* **Harms:** fraud (fake identities, voice cloning), harassment and non-consensual intimate imagery, political disinformation, and the **liar's dividend**, where genuine evidence is dismissed as fake.
* **Detection is an arms race:** detectors trained on one generator often fail on another, and adversarial training is literally optimised to fool a detector.
* **Provenance:** C2PA content credentials, invisible watermarks (e.g. SynthID) and platform labelling. Credentials can be stripped and open models may not watermark, so provenance is a layer, not a solution.
* **Law:** Article 50 of the EU AI Act requires deployers to disclose deepfakes (applicable since 2 August 2026) and providers to mark synthetic content in a machine-readable way.
* **Data and bias:** face datasets such as FFHQ raise consent questions and have demographic imbalances that generators reproduce.

Ask students to identify where their project could create synthetic media and how it would be labelled.

# Lab guide and answers

**Runtime:** Colab T4 (about 15 s per DCGAN epoch; whole lab ≈ 15 minutes of compute) or CPU (1–3 minutes per epoch; reduce epochs to 3–4). **Hand-in:** samples per epoch, diagnostics, coverage histogram and Fréchet distances, ring experiment, four answers.

## TODOs

* **TODO 1:** `bce(D(x_real), ones) + bce(D(x_fake.detach()), zeros)`. Forgetting `.detach()` makes the D step also update G in the wrong direction.
* **TODO 2:** `bce(D(x_fake), ones)` (non-saturating).
* **TODO 3 (slerp):** Ω = arccos of the cosine between normalised vectors; weights sin((1−t)Ω)/sin Ω and sin(tΩ)/sin Ω. The printed midpoint norm shows lerp shrinks the vector (≈ 0.7 × typical norm for nearly orthogonal vectors), slerp keeps it.
* **TODO 4:** generate, classify with `clf`, `argmax`. A healthy DCGAN gives all 10 classes roughly 5–15% each; fewer than ~8 classes above 5% indicates partial collapse.
* **TODO 5:** Fréchet distance with `scipy.linalg.sqrtm`, keeping the real part. Expected ordering: real vs real (smallest) < GAN final < GAN after epoch 1 < uniform noise.
* **TODO 6:** `torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)[0]`, then `((grad.norm(dim=1) - 1)**2).mean()`.
* **Expected lecture results:** DCGAN trained 8 epochs on CPU; ring experiment final coverage standard 7/8, WGAN-GP 8/8. Individual runs vary; students should report seeds.

## Troubleshooting

* *Samples are pure noise after an epoch:* check that images are normalised to [−1, 1] and the generator ends with tanh.
* *D loss → 0 immediately:* the fakes are detached in the G step too, or learning rates differ by accident.
* *`sqrtm` warnings or complex values:* expected with finite samples; take the real part.

# Practice questions with model answers

## Multiple choice

1. The optimal discriminator for a fixed generator is: **(a)** $p_g/(p_{\mathrm{data}} + p_g)$; **(b)** $p_{\mathrm{data}}/(p_{\mathrm{data}} + p_g)$; **(c)** ½ always; **(d)** $p_{\mathrm{data}}$. *Answer: (b).*
2. The non-saturating loss is used because: **(a)** it gives stronger generator gradients when D easily rejects fakes; **(b)** it removes the need for a discriminator; **(c)** it prevents overfitting; **(d)** it computes the likelihood. *Answer: (a).*
3. Which symptom most directly indicates mode collapse? **(a)** Oscillating losses; **(b)** blurry samples; **(c)** sharp samples covering only a few classes; **(d)** D(real) ≈ 0.7. *Answer: (c).*
4. FID compares: **(a)** pixel values of paired images; **(b)** means and covariances of network features for real and generated images; **(c)** classifier accuracy; **(d)** training losses. *Answer: (b).*

## Short answer

1. **Derive the optimal discriminator.** (4 marks) *Model answer:* $V = \int p_{\mathrm{data}} \log D + p_g \log(1 - D)\,dx$; pointwise maximise $a\log y + b\log(1-y)$; derivative $a/y - b/(1-y) = 0$ gives $y = a/(a+b)$; hence $D^{*} = p_{\mathrm{data}}/(p_{\mathrm{data}} + p_g)$.
2. **Explain why the Wasserstein loss helps and how WGAN-GP encourages the Lipschitz condition.** (4 marks) *Model answer:* the Wasserstein objective can provide useful directional gradients when distributions are separated, helping training and coverage (2); the penalty $(\|\nabla D(\hat x)\| - 1)^2$ encourages input-gradient norms near 1 at real/fake interpolates, without guaranteeing a global constraint or complete coverage (2).
3. **Compute a 1-D FID-style distance between N(0, 1) real features and N(1, 0.25) generated features.** (3 marks) *Model answer:* $(0-1)^2 + (1 + 0.25 - 2\sqrt{0.25}) = 1 + 0.25 = 1.25$.
4. **Compare VAEs and GANs on three criteria.** (6 marks) *Model answer:* likelihood (bound vs none), encoder (yes vs no), sample sharpness (blurry vs sharp), diversity (good vs collapse risk), training stability (stable vs unstable); two marks per well-explained criterion.

## Exam-style question

**"A start-up wants to generate realistic product photos. Explain how a GAN would be trained, what can go wrong, how you would evaluate it, and what responsible-use measures you would recommend."** (20 marks)

*Marking guide:* architecture and objective, including the minimax and non-saturating losses (5); failure modes with diagnosis and remedies (5); evaluation with FID and at least one complementary metric or human study, including caveats (5); responsible use: provenance/labelling, data rights, misuse potential, disclosure duties (5). Strong answers suggest a conditional GAN (product category), mention diffusion models as an alternative, and justify the choice.

# Glossary

| Term | Meaning |
|---|---|
| Generator | Network mapping noise (and optionally a condition) to samples |
| Discriminator / critic | Network judging real vs fake (probability) or scoring realism (WGAN critic) |
| Minimax objective | The two-player value function D maximises and G minimises |
| Non-saturating loss | Generator minimises −log D(G(z)) instead of log(1 − D(G(z))) |
| Jensen–Shannon divergence | Symmetric divergence minimised by the ideal GAN |
| Mode collapse | Generator produces only a subset of the data's variety |
| Wasserstein distance | Earth-mover distance; basis of WGAN |
| Gradient penalty | Term encouraging input-gradient norms near 1 on sampled real/fake interpolates |
| Spectral normalisation | Weight normalisation bounding D's Lipschitz constant |
| DCGAN | Convolutional GAN architecture guidelines (2015) |
| Conditional GAN | GAN whose networks receive a condition y |
| CycleGAN | Unpaired image translation with cycle consistency |
| StyleGAN | Style-based generator with a mapping network |
| FID | Fréchet Inception Distance; lower is better |
| Truncation trick | Sampling z from a truncated distribution to trade diversity for quality |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Foster, D. (2023) *Generative Deep Learning*, 2nd ed.: chapter 4, "Generative Adversarial Networks", and chapter 10, "Advanced GANs" (ProGAN, StyleGAN, and others).

**Papers**

* Goodfellow et al. (2014) [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* Radford et al. (2015) [DCGAN](https://arxiv.org/abs/1511.06434)
* Arjovsky et al. (2017) [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* Karras et al. (2019) [StyleGAN](https://arxiv.org/abs/1812.04948)
* Heusel et al. (2017) [TTUR and FID](https://arxiv.org/abs/1706.08500)

**Tutorials**

* [PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html): official reference implementation on faces.
* [Google for Developers: GAN course](https://developers.google.com/machine-learning/gan): short course covering loss functions, common problems and variants.

# Link to the group project

Projects that generate images or need fast image-to-image translation can use a GAN or, more likely in 2026, a pretrained diffusion model (next week). Either way, this week's evaluation toolkit (FID with an appropriate feature network, coverage checks, human evaluation with a rubric) is required for the project's evaluation section. Projects producing realistic people or voices must include a labelling and consent plan.
