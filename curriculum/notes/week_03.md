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

## Follow one fake image through two jobs

Use a forger-and-checker analogy, then state its limit: both neural networks update, and neither has human intent. Draw one fake image. For D’s update its label is 0 and its route to G is detached. For G’s update its desired label is 1 and its route to G must remain connected.

**Check the distinction.** Detach changes the gradient connection, not the visible pixels. Computing a gradient is different from stepping an optimiser. An optimiser changes only its registered weights.

**Explain the weak-gradient claim correctly.** If s is D’s logit and p = sigmoid(s), the minimax derivative with respect to s is −p; the non-saturating derivative is p − 1. At p = 0.01 these are −0.01 and −0.99. This is an illustrative local derivative, not proof that every generator weight gets a useful gradient.

**Repeat the explanation loop.** Predict labels, trace the gradient, run one update, inspect fixed-code images, and explain what changed. If students equate sharpness with success, ask for a missing digit class and return to the coverage histogram.



## 1. Why adversarial learning?

Basic VAEs compare pixels with a likelihood-based loss. When details are uncertain, predicting average intensities can give blur. A **generative adversarial network (GAN)** instead learns a realism checker from data.

The **discriminator** learns to separate real and generated images. The **generator** learns changes that cause the discriminator to accept its images. The checker supplies a changing training signal. Goodfellow et al. introduced this approach in 2014.

* The **generator** $G$ maps noise $z \sim p(z)$ (typically $\mathcal{N}(0, I)$, 64–512 dimensions) to a sample $G(z)$.
* The **discriminator** $D$ maps a sample to the probability that it is real.
* In this unconditional lab, G receives training feedback through D, without a paired target image for each code. Conditional variants can also use reconstruction targets.

GANs are **implicit generative models**: we can sample from them, but they give no density $p(x)$. This rules out direct likelihood-based evaluation, though other GAN anomaly methods exist, and is why GAN evaluation uses sample-based metrics such as FID.

![GAN architecture](fig:gan_arch)

> **Key idea:** A GAN replaces a hand-written loss ("be close pixel by pixel") with a learned, adaptive loss ("be hard to tell apart from real data"). The price is that training becomes a two-player game rather than the minimisation of a fixed function.

## 2. The GAN objective

Read **minimax** as two goals: the generator minimises a quantity that the discriminator maximises. E means an average over samples. $p_{\mathrm{data}}$ describes real examples and $p_g$ describes generated examples. A gradient measures how a loss changes when an input or weight changes.

### The minimax game

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log\left(1 - D(G(z))\right)\right]$$

**Read one pair.** If D(real) = 0.9 and D(fake) = 0.2, the two terms sum to ln(0.9) + ln(0.8), about −0.329. D wants this sum higher. G can affect the fake term.

For the discriminator this is (the negative of) the binary cross-entropy of a real-vs-fake classifier with labels 1 for real and 0 for fake. The generator appears only in the second term.

### The optimal discriminator (derivation)

Write the expectation over fakes as an expectation over the generator's distribution $p_g$. For a fixed $G$:

$$V(D, G) = \int \left[p_{\mathrm{data}}(x) \log D(x) + p_g(x) \log(1 - D(x))\right] dx$$

**Read the integral.** At each x, weight the real-source and generated-source log scores by their densities. The integral adds contributions across the whole input space. This form lets us optimise D point by point.

For each $x$, the function $a \log y + b \log(1 - y)$ is maximised at $y = a/(a + b)$. Hence

$$D^{*}_G(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + p_g(x)}$$

**Worked example.** If $p_{\mathrm{data}}(x) = 0.3$ and $p_g(x) = 0.1$, D*(x) = 0.3/0.4 = 0.75. If both densities match, the answer is 0.5 wherever their sum is positive.

Substituting back gives

$$V(D^{*}_G, G) = -\log 4 + 2\,\mathrm{JSD}(p_{\mathrm{data}} \,\|\, p_g)$$

where the **Jensen–Shannon divergence** $\mathrm{JSD}(p\|q) = \tfrac12 D_{\mathrm{KL}}(p\|m) + \tfrac12 D_{\mathrm{KL}}(q\|m)$ with $m = \tfrac12(p + q)$ is symmetric, non-negative and zero only when $p = q$. So, with an optimal discriminator and unlimited capacity, the generator minimises the JSD, and the global optimum is $p_g = p_{\mathrm{data}}$, where $D^{*} = \tfrac12$ wherever the shared density is positive and $V = -\log 4$.

**Worked example.** At JSD = 0, V = −ln(4), about −1.386. A positive JSD raises this ideal-discriminator objective, so the generator seeks matching distributions.

![The optimal discriminator early in training and at equilibrium](fig:optimal_d)

In practice neither network has unlimited capacity, $D$ need not be optimal, and the game is optimised with alternating stochastic gradient steps, so this is an idealisation that explains the goal rather than a guarantee.

### The non-saturating generator loss

Early in training, $D(G(z))$ is often close to zero: D easily rejects generated images. The original minimax loss can give G **vanishing gradients**, meaning very weak guidance for a weight change.

Use the **non-saturating loss**, $-\log D(G(z))$. At a fake probability of 0.1 it is about 2.30, while at 0.8 it is about 0.22. Minimising it pushes fake scores up. In code, compare fake logits with target 1. D still uses target 0 for fakes in its own update.

![Minimax vs non-saturating generator loss](fig:saturating)

### One training step

```python
fake = G(torch.randn(b, z_dim))
loss_D = bce(D(x_real), ones) + bce(D(fake.detach()), zeros)   # D step
opt_D.zero_grad(); loss_D.backward(); opt_D.step()
loss_G = bce(D(fake), ones)                                    # G step (non-saturating)
opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

`.detach()` cuts the gradient route to G during D’s loss calculation. `opt_D` changes only D’s registered weights. DCGAN defaults: Adam, learning rate 2 × 10⁻⁴, β₁ = 0.5.

## 3. Training challenges and fixes

### Reading the diagnostics

GAN losses can rise and fall because each network changes the other’s task. This can cause **instability**. Inspect losses and D’s real/fake scores alongside images. Very confident D scores can mean weak generator feedback, but no single curve proves failure or success.

Reuse the same noise codes for snapshots so changes reflect training. Then check variety and coverage with metrics and class counts. A low generator loss can coexist with repeated or poor images.

![Real training curves of the lecture DCGAN](fig:losses)

### Mode collapse

**Mode collapse** means G misses substantial parts of the data distribution. It may make sharp images of only a few digit classes. Different random codes then give similar outputs.

G can exploit one pattern that currently fools D, then switch when D adapts. In the measured eight-cluster experiment, the standard GAN covered seven modes at the end while WGAN-GP covered eight. This is evidence for that run, not a universal guarantee.

![Mode collapse experiment on a ring of 8 Gaussians](fig:mode_collapse)

### Failure modes and remedies

| Problem | Symptom | Remedies |
|---|---|---|
| Mode collapse | Samples alike; classes missing from the histogram | Wasserstein loss (+GP), minibatch statistics in D, unrolled D, more diverse data |
| Weak gradients | Little G improvement; inspect gradients rather than D(fake) alone | Non-saturating loss; weaker or regularised D; Wasserstein loss |
| Non-convergence | Samples keep changing style; wild oscillation | Lower learning rates; TTUR; spectral normalisation; EMA of G |
| D overfits (small data) | D(real) → 1 on training images only | Discriminator augmentation (ADA); more data |
| Artefacts | Checkerboards, textures stuck to screen positions | Resize-then-convolve upsampling; StyleGAN2/3 architectures |

### The Wasserstein GAN with gradient penalty

The Wasserstein-1 (earth-mover) distance measures the minimum "work" to move probability mass from one distribution to the other. By Kantorovich–Rubinstein duality it equals $\sup_{\|f\|_L \leq 1} \mathbb{E}_{p_{\mathrm{data}}}[f(x)] - \mathbb{E}_{p_g}[f(x)]$, so a **critic** $D$ restricted to 1-Lipschitz functions can estimate it:

$$\min_G \max_{\|D\|_L \leq 1} \mathbb{E}_{x \sim p_{\mathrm{data}}}[D(x)] - \mathbb{E}_{z}[D(G(z))]$$

**Score-gap example.** If average real score = 2.0 and fake score = 0.5, the gap is 1.5. The critic increases the gap under its smoothness constraint. G tries to raise fake scores.

Unlike the JSD, it can vary usefully even when the two distributions do not overlap, giving the generator more informative directional feedback. WGAN-GP (Gulrajani et al., 2017) encourages the Lipschitz condition by adding to the critic's loss

$$\lambda\, \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right], \quad \hat{x} = \epsilon x + (1-\epsilon) G(z), \; \epsilon \sim U[0, 1]$$

with λ = 10 and typically 5 critic steps per generator step. This encourages gradient norms near 1 on sampled interpolates; it does not guarantee a globally 1-Lipschitz critic or complete mode coverage. The critic outputs an unbounded score, not a probability.

**Penalty example.** For gradient norm 1.4 and $\lambda = 10$, the point penalty is 10 × (1.4 − 1)² = 1.6. Norm 1 gives zero. This encourages smoothness at sampled points.

### Other stabilisation techniques

* **Spectral normalisation** (Miyato et al., 2018): divide each discriminator layer's weight by its largest singular value, a way to control layer scale; the full-network constraint depends on layers and implementation (`torch.nn.utils.spectral_norm`).
* **TTUR** (Heusel et al., 2017): different learning rates for D and G.
* **EMA of generator weights:** sample from an exponential moving average of G's parameters.
* **Adaptive discriminator augmentation (ADA):** augment images shown to D so it cannot memorise small datasets.
* **R1 regularisation:** penalise D's gradient on real data (default in StyleGAN2).

## 4. Variants: from DCGAN to StyleGAN

### DCGAN

**DCGAN** means deep convolutional GAN. Convolutions learn from local pixel patterns. G uses transposed convolutions to increase spatial size. D uses strided convolutions to reduce it.

The lab follows Radford et al. (2015): batch normalisation in selected layers, ReLU in G, tanh at G’s output and LeakyReLU in D. It uses Adam, learning rate 0.0002 and beta1 = 0.5. These are useful starting settings, not a guarantee of stable training.

![DCGAN generator and guidelines](fig:dcgan_arch)

![Real DCGAN samples across epochs](fig:epochs)

### Conditional GANs and image-to-image translation

A **conditional GAN** (Mirza & Osindero, 2014) feeds a condition $y$ to both networks: $G(z, y)$ and $D(x, y)$. $D$ must check that $x$ is real **and** matches $y$, so $G$ learns to respect the condition.

* **pix2pix** (Isola et al., 2017): paired translation (sketch → photo, labels → street scene) with a conditional GAN plus an L1 reconstruction loss and a "PatchGAN" discriminator that judges local patches.
* **CycleGAN** (Zhu et al., 2017): unpaired translation (horse ↔ zebra) with two generators $G: X \to Y$, $F: Y \to X$ and a **cycle-consistency** loss:

$$\mathcal{L}_{\mathrm{cyc}} = \mathbb{E}_x\left[\|F(G(x)) - x\|_1\right] + \mathbb{E}_y\left[\|G(F(y)) - y\|_1\right]$$

**Round-trip example.** If a two-pixel input [0.2, 0.8] returns as [0.3, 0.6], its L1 round-trip error is |0.3 − 0.2| + |0.6 − 0.8| = 0.3. Both translation directions contribute.

**Distinguish the discriminators.** The class-conditioned diagram supplies the condition to both networks, as in the original conditional GAN and pix2pix. Standard CycleGAN uses one discriminator for each image domain; it judges target-domain images without receiving the corresponding source image.

![Conditional GAN](fig:cgan)

### StyleGAN and BigGAN

**StyleGAN** maps random code $z$ to style vector $w$, then uses that style at several generator layers. Coarse layers affect pose or shape. Finer layers affect hair or texture. This gives editing controls, although they need not be fully independent.

**BigGAN** scales class-conditioned generation. Its **truncation trick** restricts sampled noise to a smaller range, trading some variety for quality.

### Latent interpolation and GAN inversion

Interpolating between noise vectors can give smooth transitions; inspect the images rather than assuming every intermediate code is useful. **Spherical interpolation (slerp)** avoids lerp's usual drop in norm for nearly orthogonal codes. Equal-length endpoints stay on a sphere; the lab's unequal random endpoints need not keep exactly one norm. Basic GANs have **no encoder**. **GAN inversion** finds a code for a real image by optimisation or with an additional encoder.

![Latent interpolation in the lecture DCGAN](fig:interpolation)

## 5. Evaluation, applications and risks

### Fréchet Inception Distance (FID)

Embed real and generated images with an Inception-v3 network (2048-dimensional pool features), fit a Gaussian to each set and compute the Fréchet distance:

$$\mathrm{FID} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)$$

**Read the symbols first.** $\mu$ (mu) is the average feature vector. $\Sigma$ (Sigma) is a covariance matrix describing feature variation. Tr adds diagonal entries. The one-dimensional worked example below makes the calculation smaller.

Lower is better. Both feature means and covariances can change with quality and coverage; neither is a pure quality or diversity measure. Limitations: assumes Gaussian features; biased for small sample sizes (use ≥ 10,000 samples or KID); ImageNet features may not suit other domains; values are only comparable with identical feature networks, preprocessing and sample sizes.

**Worked example (1-D intuition).** If real features have mean 0 and variance 1 and generated features have mean 0.5 and variance 0.25, then FID = 0.5² + (1 + 0.25 − 2√(1 × 0.25)) = 0.25 + (1.25 − 1) = **0.50**. A collapsed generator with variance 0.01 and mean 0 gives 0 + (1 + 0.01 − 2 × 0.1) = **0.81**, worse despite a perfect mean.

![What FID measures in feature space](fig:fid)

**Lab distinction:** `frechet_distance` uses the same distribution-comparison formula with features from the lab's `DigitCNN`, not Inception-v3. Call its output a **custom feature Fréchet distance**. Its scale differs from standard Inception FID, so do not compare it with published FID values. Check the classifier's accuracy and keep its weights, preprocessing, and sample size fixed for comparisons within the lab.

### Other metrics

| Metric | Captures | Limitation |
|---|---|---|
| Inception Score (IS, higher is better) | Confident and varied class predictions: $\exp\left(\mathbb{E}_x D_{\mathrm{KL}}(p(y\mid x)\,\|\,p(y))\right)$ | Ignores the real data; fooled by class-diverse but unrealistic images |
| Precision / recall (Kynkäänniemi et al., 2019) | Realism (precision) and coverage of the real data (recall), measured separately | Sensitive to feature space and neighbourhood size |
| KID | Kernel MMD on Inception features; unbiased estimate still has sampling uncertainty | Sampling variance and feature choices |
| Human evaluation | Realism, preference, task fit | Needs rubrics, multiple raters, agreement statistics |

### GANs in 2026

GAN ideas remain useful when fast generation matters. Uses include super-resolution and restoration, such as Real-ESRGAN. An **audio vocoder**, such as HiFi-GAN, turns a sound representation into a waveform.

Adversarial diffusion distillation trains a faster generator from a slower diffusion model, sometimes using 1–4 steps. Adversarial losses also help keep autoencoder outputs sharp. Other uses include image translation and synthetic tabular data. Each needs task-specific evaluation.

### VAE vs GAN

| | VAE | GAN |
|---|---|---|
| Training signal | Maximise ELBO (likelihood bound) | Two-player game with a learned discriminator |
| Likelihood | Lower bound with an appropriate decoder likelihood | No tractable explicit likelihood in the basic GAN |
| Encoder | Yes | Basic GAN: no; inversion needs extra work |
| Samples | May blur uncertain details | May be sharp; inspect quality |
| Diversity | Must evaluate coverage | Risk of mode collapse |
| Training | One objective; failures still occur | Moving game; monitor both networks |

# Common misconceptions

| Misconception | Correction |
|---|---|
| "A low generator loss means good images." | GAN losses are relative to the current discriminator; judge samples and metrics. |
| "The generator copies one real image for each noise code." | This unconditional G gets adversarial feedback through D, without a paired image target. Image-conditioned variants can also receive real input images. |
| "At the end of training D should be perfect." | At the ideal matched-distribution solution D outputs ½ on shared positive support. Scores alone do not prove training success or failure. |
| "Mode collapse means blurry images." | Mode collapse means low diversity; the images can be very sharp. |
| "FID is an absolute quality score." | FID values are only comparable under identical feature networks, preprocessing and sample sizes. |
| "GANs are obsolete." | They power super-resolution, fast distilled diffusion, sharp autoencoders and vocoders. |
| "WGAN's critic outputs a probability." | It outputs an unbounded score; the difference of average scores estimates the Wasserstein distance. |

# Responsible AI lens: deepfakes and synthetic media

* **Harms:** fraud (fake identities, voice cloning), harassment and non-consensual intimate imagery, political disinformation, and the **liar's dividend**, where genuine evidence is dismissed as fake.
* **Detection is an arms race:** detectors trained on one generator often fail on another, and adversarial training is literally optimised to fool a detector.
* **Provenance (source history):** C2PA content credentials, invisible watermarks (e.g. SynthID) and platform labelling. Credentials can be stripped and open models may not watermark, so provenance is one layer of protection, not a complete solution.
* **Law:** Article 50 distinguishes provider marking from deployer disclosure, including deepfakes. The scope includes exceptions and special disclosure treatment for artistic works. The rules apply from 2 August 2026; systems placed on the market before that date have until 2 December 2026 for the Article 50(2) marking duty. This transition does not postpone every Article 50 duty. Read the [official Article 50 text](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) and [Commission timing guidance](https://digital-strategy.ec.europa.eu/en/factpages/quick-facts-transparency-rules-ai-systems) for the role and use concerned.
* **Data and bias:** face datasets such as FFHQ raise consent questions and have demographic imbalances that generators reproduce.

Ask students to identify where their project could create synthetic media and how it would be labelled.

# Lab guide and answers

**Runtime:** Colab or a laptop CPU. On the laptop CPU used to check this pack (9 October 2026), each DCGAN epoch took about 2 minutes and the complete solution notebook about 19 minutes; other jobs were running, so treat these as upper bounds. A GPU is much faster. For a short CPU session, reduce `EPOCHS` to 3–4. **Hand-in:** samples per epoch, diagnostics, coverage histogram and Fréchet distances, ring experiment, four answers.

## TODOs

* **TODO 1:** `bce(D(x_real), ones) + bce(D(x_fake.detach()), zeros)`. Forgetting `.detach()` computes unwanted G gradients in the D backward pass. Which weights actually change is controlled by the optimiser’s parameter list.
* **TODO 2:** `bce(D(x_fake), ones)` (non-saturating).
* **TODO 3 (slerp):** Ω = arccos of the cosine between normalised vectors; weights sin((1−t)Ω)/sin Ω and sin(tΩ)/sin Ω. Inspect both endpoint and midpoint norms. Lerp's midpoint is about 0.7 of the endpoint norm for equal-length, nearly orthogonal codes; slerp preserves the norm only for equal-length endpoints.
* **TODO 4:** generate, classify with `clf`, `argmax`. Compare class counts with held-out real digits and inspect images. Missing or strongly underrepresented classes suggest coverage problems; classifier errors and finite samples also affect the histogram.
* **TODO 5:** Fréchet distance with `scipy.linalg.sqrtm`, discarding only negligible imaginary numerical round-off; large complex parts require investigation. Compare real-vs-real, final-GAN, epoch-1 and uniform-noise scores. A well-trained run may improve towards the real reference; record the actual ordering and investigate surprises.
* **TODO 6:** `torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)[0]`, then `((grad.norm(dim=1) - 1)**2).mean()`.
* **Archived lecture run:** DCGAN trained 8 epochs on CPU; ring experiment final coverage standard 7/8, WGAN-GP 8/8. These are recorded results, not targets that every run must reproduce. Students should report seeds and actual results.

## Troubleshooting

* *Samples are pure noise after an epoch:* check that images are normalised to [−1, 1] and the generator ends with tanh.
* *D loss → 0 immediately:* the fakes are detached in the G step too, or learning rates differ by accident.
* *`sqrtm` warnings or complex values:* small imaginary round-off can occur; investigate large imaginary parts, nonfinite values or a substantially negative distance.

# Practice questions with model answers

## Multiple choice

1. The optimal discriminator for a fixed generator is: **(a)** $p_g/(p_{\mathrm{data}} + p_g)$; **(b)** $p_{\mathrm{data}}$; **(c)** ½ always; **(d)** $p_{\mathrm{data}}/(p_{\mathrm{data}} + p_g)$. *Answer: (d).*
2. The non-saturating loss is used because: **(a)** it gives stronger generator gradients when D easily rejects fakes; **(b)** it removes the need to update the discriminator during training; **(c)** it prevents the discriminator from overfitting to the real data; **(d)** it computes the exact likelihood of each generated image. *Answer: (a).*
3. Which symptom most directly indicates mode collapse? **(a)** Oscillating generator and discriminator losses; **(b)** blurry samples that average several digits; **(c)** sharp samples covering only a few classes; **(d)** D(real) ≈ 0.7 on held-out real images. *Answer: (c).*
4. FID compares: **(a)** pixel values of each generated image and its nearest real image; **(b)** means and covariances of network features for real and generated images; **(c)** how accurately a classifier labels the generated images; **(d)** the final generator and discriminator losses from training. *Answer: (b).*

## Short answer

1. **Derive the optimal discriminator.** (4 marks) *Model answer:* $V = \int p_{\mathrm{data}} \log D + p_g \log(1 - D)\,dx$; pointwise maximise $a\log y + b\log(1-y)$; derivative $a/y - b/(1-y) = 0$ gives $y = a/(a+b)$; hence $D^{*} = p_{\mathrm{data}}/(p_{\mathrm{data}} + p_g)$.
2. **Explain why the Wasserstein loss helps and how WGAN-GP encourages the Lipschitz condition.** (4 marks) *Model answer:* the Wasserstein objective can provide useful directional gradients when distributions are separated, helping training and coverage (2); the penalty $(\|\nabla D(\hat x)\| - 1)^2$ encourages input-gradient norms near 1 at real/fake interpolates, without guaranteeing a global constraint or complete coverage (2).
3. **Compute a 1-D FID-style distance between N(0, 1) real features and N(1, 0.25) generated features.** (3 marks) *Model answer:* $(0-1)^2 + (1 + 0.25 - 2\sqrt{0.25}) = 1 + 0.25 = 1.25$.
4. **Compare VAEs and GANs on three criteria.** (6 marks) *Model answer:* VAE likelihood bound versus a basic GAN's lack of tractable explicit likelihood; VAE encoder versus extra work for GAN inversion; reconstruction-plus-KL objective versus an adversarial game; possible VAE blur versus possible GAN sharpness; coverage evaluation for both, including GAN collapse risk. These are tendencies and design differences, not guarantees. Award two marks per well-explained criterion.

## Exam-style question

**"A start-up wants to generate realistic product photos. Explain how a GAN would be trained, what can go wrong, how you would evaluate it, and what responsible-use measures you would recommend."** (20 marks)

*Marking guide:* architecture and objective, including the minimax and non-saturating losses (5); failure modes with diagnosis and remedies (5); evaluation with FID and at least one complementary metric or human study, including caveats (5); responsible use: provenance and labelling, data rights, misuse potential, disclosure duties (5). Strong answers suggest a conditional GAN (product category), mention diffusion models as an alternative, and justify the choice.

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
| Spectral normalisation | Normalise layer weight scale; network smoothness depends on the full design |
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


# Detailed explanations behind the short slides

The lecture shows the essential idea first. These fuller explanations retain the supporting comparisons, examples and checks for revision.

## Slide 6: The motivation: a learned loss for realism

- Pixel losses can reward average-looking images, giving the blur seen in basic VAEs.

- A **discriminator** learns to check whether an image came from the real dataset or generator.

- A **generator** learns to make images the discriminator accepts as real.

- The discriminator supplies a changing training signal for image realism.

- A GAN can sample images but does not directly calculate their likelihood p(x). This is an **implicit model**.

The generator learns from a checker trained on real examples.

## Slide 8: Generator vs. discriminator

The networks change each other's training signal. Their losses need to stay in balance.

**Generator G.** 

- Input: random code z, optionally with a class label

- Output: a generated image

- Goal: make D rate generated images as real

- Grow small features into an image using upsampling

- Use G to generate after training

**Discriminator D.** 

- Input: a real or generated image

- Output: one real/fake score

- Goal: rate real images high and generated images low

- Read the image with a convolutional network (CNN)

- Usually needed only during training

## Slide 11: The minimax game (Goodfellow et al., 2014)

- D tries to increase V: score real images high and generated images low.

- G tries to decrease V: change its images so D rates them higher.

- Update D, then update G. The expectations mean averages over real images and random codes.

## Slide 12: The optimal discriminator and what G really minimises

- With G fixed, the ideal D compares the real and generated densities at each point.

- Example: if both densities equal 0.2 at x, D*(x) = 0.2 / (0.2 + 0.2) = 0.5.

- Under this ideal D, G minimises **Jensen-Shannon divergence (JSD)**, a difference between distributions. The best case is $p_g = p_{\mathrm{data}}$.

## Slide 14: Why practitioners use the non-saturating generator loss

- Early in training, D often rates generated images near zero.

- The original minimax loss can then give G a very weak gradient, meaning little guidance for a weight change.

- The **non-saturating loss**, −log D(G(z)), gives a stronger signal when D rejects the image.

- It has the same ideal target. We use it for the image GAN in the lab.

## Slide 15: One training step in PyTorch

- `BCEWithLogitsLoss` compares raw real/fake scores with target labels.

- During D’s update, `.detach()` prevents changes to G through that loss.

- During G’s update, target 1 asks D to accept generated images as real.

- This example uses Adam with learning rate 0.0002 and beta1 = 0.5.

## Slide 18: Real training curves: losses do not tell you quality

- The losses can rise and fall as the two networks adapt.

- In this run, D rates real images higher than generated images.

- Compare fixed-noise sample images over time. Check quality and variety with separate measures.

## Slide 20: Failure modes: symptoms and remedies

| Problem | Symptom | Common remedies |
|---|---|---|
| Mode collapse | Many random codes produce similar images | Count generated classes. Try WGAN-GP, minibatch statistics, multiple/unrolled D or better data coverage. |
| Vanishing gradients | G receives little useful feedback | Try non-saturating loss, a weaker/regularised D or Wasserstein loss. |
| Unstable training | Images and losses keep changing sharply | Try smaller or separate learning rates (TTUR), spectral normalisation or averaged G weights (EMA). |
| D memorises training data | High real scores only on seen images | Apply matching image changes to real and fake samples, such as adaptive augmentation (ADA). |
| Image artefacts | Checkerboard patterns or odd textures | Try resize-then-convolve upsampling or improved StyleGAN architectures. |

## Slide 21: The Wasserstein GAN with gradient penalty (WGAN-GP)

- The **critic** D returns a score rather than a probability. Compare its average real and generated scores.

- **Wasserstein distance** measures the work of moving one distribution into another. Its signal can help when their regions do not overlap.

- The gradient penalty checks points between real and fake samples. It encourages smooth critic slopes. It does not guarantee a global slope limit.

## Slide 22: Stabilisation techniques you will meet

**Wasserstein loss + GP.** Use a critic score and a gradient penalty to encourage useful, smoother feedback.

**Spectral normalisation.** Rescale layer weights using their largest singular value to limit how sharply D changes.

**TTUR.** Two time-scale update rule: use different learning rates for D and G.

**EMA of G weights.** Exponential moving average: keep a smoothed copy of G's weights for sample generation. This adds a small update cost.

**Augmentation (ADA).** Change both real and fake images before D sees them, reducing memorisation of a small dataset.

**R1 penalty.** Penalise large input gradients of D on real images. StyleGAN2 uses this regulariser.

## Slide 27: Conditional GANs: choose what to generate

- Give the same condition y to G(z, y) and D(x, y).

- Example: y = 7 asks G to make a seven. D checks both realism and agreement with label 7.

- Conditions can also be text, sketches, maps or other images.

## Slide 28: Landmark GAN variants

**pix2pix (2017).** Learn with matching image pairs, such as each sketch and its photo. Use a conditional GAN plus pixel L1 loss.

**CycleGAN (2017).** Learn from two collections without exact pairs, such as horses and zebras. Use a round-trip loss.

**StyleGAN (2019–21).** Use a mapping network and style controls at several layers to change features such as pose or texture.

**BigGAN (2018).** Generate selected ImageNet classes at large scale. Restrict noise values to trade variety for quality.

## Slide 29: CycleGAN's key idea: cycle consistency

- G turns an image from X into Y. F turns an image from Y back into X.

- Example: horse image, generated zebra, reconstructed horse. Compare the final horse with the original.

- Add the round-trip **cycle loss** to the two adversarial losses. L1 adds absolute pixel differences.

## Slide 30: Latent interpolation in a GAN

- Choose two random codes and decode the points between them.

- Basic GANs have no encoder. **GAN inversion** searches for a code that reproduces a real image.

- StyleGAN provides extra style controls for image editing.

## Slide 32: Fréchet Inception Distance (FID)

- Use the same pretrained network to turn real and generated images into feature vectors.

- Compare their feature means and covariance matrices with FID. Lower values mean more similar statistics in this feature space.

- FID = 0 means matching Gaussian feature statistics, rather than proof of identical images. Report sample count and feature network.

## Slide 34: Evaluating GANs: the main tools

| Metric | What it captures | Limitations |
|---|---|---|
| FID (lower is better) | Difference between real/generated feature statistics | Needs many samples. The feature network and implementation affect the result. |
| Inception Score (higher) | Confident predictions across varied classes | Uses no real reference set. High scores can hide poor images. |
| Precision & recall | Precision: realistic samples. Recall: coverage of real-data regions. | Both depend on features and how nearby points are defined. |
| KID | Difference between feature distributions, with an unbiased estimator | A small sample still has uncertainty. Report sample size. |
| Human evaluation | People judge realism, preference or task usefulness | Use clear criteria, several raters and report agreement. |

## Slide 35: Where GANs are used in 2026

**Super-resolution.** Upscaling and restoring photos and video (e.g. Real-ESRGAN): one fast forward pass.

**Fast diffusion.** Adversarial diffusion distillation turns 50-step diffusion into 1–4-step generators (e.g. SDXL-Turbo).

**Sharper autoencoders.** Adversarial losses keep the VAE/VQ decoders inside latent diffusion and image tokenizers sharp.

**Audio vocoders.** GAN vocoders (e.g. HiFi-GAN) turn spectrograms into natural speech in real time.

**Image translation.** Domain adaptation, style transfer and simulation-to-real for robotics and medical imaging.

**Synthetic data.** Augmenting rare classes; tabular data generators, with privacy checks.

## Slide 36: Responsible AI lens: deepfakes and synthetic media

- **Deepfakes** are generated or changed media that appear to show a real person or event.

- Misuse can include fraud, harassment, non-consensual imagery and false political claims.

- The **liar's dividend** means someone dismisses real evidence by claiming it is fake.

- Detectors may fail on new generators. Check source history, labels and content credentials (C2PA).

- Use authorised data and follow applicable AI Act disclosure duties.

## Slide 37: VAE vs. GAN: a summary for the exam

|  | VAE (week 2) | GAN (week 3) |
|---|---|---|
| Training signal | Maximise ELBO (likelihood bound) | Two-player game with a learned discriminator |
| Likelihood p(x) | Lower bound with an appropriate decoder likelihood | No tractable explicit likelihood in the basic GAN |
| Encoder | Yes: images → latent codes | Basic GAN: no; inversion needs extra work |
| Sample quality | May blur uncertain details | May be sharp; inspect realism |
| Diversity | Must evaluate coverage | Risk of mode collapse |
| Training | One objective; failures still occur | Moving game; monitor both networks |
| Typical use today | Autoencoder in latent diffusion; anomaly detection | Super-resolution, fast distillation, vocoders |

## Slide 38: Lab 3: train and diagnose a DCGAN

Google Colab or a local CPU (about 2 minutes per DCGAN epoch on the laptop CPU used to check this pack)

Notebook with samples per epoch, diagnostics, custom feature Fréchet distance and class histogram, ring experiment and written answers

- Build a generator and discriminator for 28 × 28 digit images.

- Implement D’s real/fake loss and G’s non-saturating loss.

- Train and record losses, real/fake scores and images from fixed noise codes.

- Decode paths between codes and inspect repeated or missing digit classes.

- Train a digit classifier. Use its features for a custom Fréchet distance and class histogram.

- Compare standard GAN and WGAN-GP on eight toy clusters. Count covered clusters.

## Slide 39: Summary

- G creates samples. D learns to tell real and generated samples apart.

- The minimax game has an ideal solution where the two data distributions match.

- The non-saturating G loss improves early feedback: −log D(G(z)).

- Check fixed-noise images and class coverage. Repeated outputs can signal mode collapse.

- WGAN-GP, spectral normalisation, TTUR, EMA and augmentation can help training.

- DCGAN, conditional GAN, pix2pix, CycleGAN and StyleGAN change the design or task.

- Use FID, precision/recall and human checks together.

- Applications include super-resolution, fast generators and audio. Synthetic media needs consent and disclosure.
