# Lecture plan at a glance

Diffusion models are the backbone of today's image, video and audio generators. Students should leave able to write the forward process and its closed form, the noise-prediction loss and the DDPM update; explain latent diffusion and classifier-free guidance; and use and evaluate a pretrained pipeline. All MNIST, 2-D and Stable Diffusion images in the slides are real outputs generated for this lecture (`curriculum/assets/make_week_04.py`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | Restoring a photo from static: small steps, random start, a description helps. |
| 5–17 min | 5–8 | 1 · Why diffusion? | Limitations of VAEs and GANs; the trilemma; the core idea in one slide. |
| 17–50 min | 9–19 | 2 · Forward and reverse | Forward process and closed form; real noising; schedules; reverse process; noise-prediction loss; code; score view; real 2-D and MNIST results. Quiz. |
| 50–57 min | – | Break | |
| 57–74 min | 20–23 | 3 · Architecture | U-Net vs DiT; latent diffusion pipeline; role of each component. |
| 74–97 min | 24–29 | 4 · Sampling and guidance | Samplers; steps; classifier-free guidance; guidance scale; seeds and negative prompts. |
| 97–118 min | 30–37 | 5 · Variants, evaluation, responsibility | Flow matching; distillation; variants; evaluation; family comparison; responsible AI. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Write the closed-form forward process on the board early and point out that it is the reparameterisation trick from week 2. Students who see this link find the rest of the lecture much easier.

<!-- pagebreak -->

# Lecture notes

## 1. Why diffusion?

| | VAE | GAN | Diffusion |
|---|---|---|---|
| Sample quality | Blurry | Sharp | Sharp, detailed |
| Coverage | Good | Mode collapse risk | Good |
| Training | Stable | Unstable game | Stable regression |
| Text conditioning | Weak | Hard to scale | Excellent |
| Sampling | 1 pass | 1 pass | 10–1000 passes (reducible) |

Diffusion models combine the stable, likelihood-based training of VAEs with the sample quality of GANs, at the price of slow sampling. The **generative trilemma** (quality, diversity, speed) summarises the trade-off; distillation and flow matching now move diffusion towards the fast corner.

![The generative trilemma](fig:trilemma)

**Core idea.** A fixed **forward process** gradually adds Gaussian noise until the data become pure noise. A network learns the **reverse process**, removing a little noise at a time. Each reverse step is an easy denoising problem, trained as ordinary regression. Generation starts from pure noise and applies the learned reverse step repeatedly. Randomness in the starting noise gives diversity; a prompt gives control. (Sohl-Dickstein et al., 2015; Ho, Jain & Abbeel, 2020.)

## 2. Forward and reverse processes

### The forward process

With a small noise schedule $\beta_1, \ldots, \beta_T$ (DDPM: linear from $10^{-4}$ to $0.02$, $T = 1000$):

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)$$

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Because Gaussians compose, we can jump directly to any step:

$$q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right) \quad\Leftrightarrow\quad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

This is the reparameterisation trick from week 2. The scaling by $\sqrt{1-\beta_t}$ keeps the total variance at 1 ("variance-preserving"). At $t = T$, $\bar{\alpha}_T \approx 4 \times 10^{-5}$: essentially pure noise.

**Worked example.** For a pixel with clean value $x_0 = 0.8$, at a step where $\bar{\alpha}_t = 0.36$ and with sampled noise $\epsilon = -0.5$: $x_t = \sqrt{0.36}(0.8) + \sqrt{0.64}(-0.5) = 0.6 \times 0.8 - 0.8 \times 0.5 = 0.48 - 0.40 = 0.08$.

![The forward process on real digits](fig:forward_mnist)

### Noise schedules

The schedule decides how quickly information is destroyed. The linear schedule leaves many late steps with almost no signal; the **cosine schedule** (Nichol & Dhariwal, 2021) removes signal more evenly. The **signal-to-noise ratio** $\mathrm{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$ is the natural way to compare schedules (Karras et al., 2022).

![Linear vs cosine schedules](fig:schedules)

### The reverse process

For small $\beta_t$ the reverse step is approximately Gaussian, so it is modelled as

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\, \sigma_t^2 I\right), \qquad \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)$$

with $\sigma_t^2 = \beta_t$ in the simplest DDPM. One network $\epsilon_\theta$, shared over all steps, receives the timestep through a sinusoidal embedding.

### The training objective

$$\mathcal{L}_{\mathrm{simple}}(\theta) = \mathbb{E}_{x_0,\, t,\, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\; t\right)\right\|^2\right]$$

1. Sample a training image $x_0$, a timestep $t \sim U\{1, \ldots, T\}$ and noise $\epsilon \sim \mathcal{N}(0, I)$.
2. Form $x_t$ with the closed form.
3. Predict the noise and minimise the squared error.

Ho et al. derived this from the **ELBO** (week 2): a diffusion model is a hierarchical latent variable model with a fixed, parameter-free "encoder" (the noising chain), and $\mathcal{L}_{\mathrm{simple}}$ is a reweighted version of its bound. Diffusion models are therefore likelihood-based, which explains their good coverage, and trained by regression, which explains their stability.

```python
t   = torch.randint(0, T, (batch,))
eps = torch.randn_like(x0)
xt  = abar[t].sqrt() * x0 + (1 - abar[t]).sqrt() * eps
loss = F.mse_loss(model(xt, t), eps)
```

Training costs one network evaluation per example; **sampling** costs one per step (1000 for ancestral DDPM).

### The score-based view

The **score** of a distribution is $\nabla_x \log p(x)$. Predicting the noise is equivalent to estimating the score of the noised data: $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\,\nabla_{x_t} \log p_t(x_t)$ (denoising score matching). Song et al. (2021) showed that diffusion is a discretised **stochastic differential equation** with an equivalent deterministic **probability-flow ODE**; ODE solvers are the basis of fast samplers.

### Real results

A three-layer MLP trained with $\mathcal{L}_{\mathrm{simple}}$ on a 2-D swiss roll turns Gaussian noise into the spiral during the reverse process. A tiny U-Net ({{UNET_PARAMS}} parameters) trained for {{MNIST_EPOCHS}} epochs on MNIST ({{MNIST_DEVICE}}, about {{MNIST_MIN}} minutes) generates recognisable digits; structure appears in the middle steps and detail at the end (coarse-to-fine).

![Reverse process on 2-D data](fig:reverse_2d)

![Reverse process on MNIST](fig:reverse_mnist)

![Samples from the tiny MNIST diffusion model](fig:mnist_samples)

## 3. Architecture and latent diffusion

### Denoiser networks

* **U-Net** (Ronneberger et al., 2015; used by DDPM, Stable Diffusion 1.x/2.x, SDXL): convolutional encoder–decoder with skip connections; attention layers at low resolution; output has the input's shape.
* **Diffusion transformer (DiT)** (Peebles & Xie, 2022): split the latent into patches (tokens) and process them with transformer blocks; conditioning through adaptive layer norm or joint attention; scales better with compute (Stable Diffusion 3, FLUX, most video models).

Both receive the timestep and the condition at every block.

![U-Net and diffusion transformer](fig:unet_dit)

### Latent diffusion (Rombach et al., 2022)

1. A **VAE** (week 2) compresses a 512 × 512 × 3 image to a 64 × 64 × 4 latent (48× fewer numbers).
2. A **text encoder** (CLIP's text transformer in SD 1.5; T5 as well in later models) turns the prompt into token embeddings.
3. The **denoiser** runs the reverse process in latent space and reads the text through **cross-attention** (week 5).
4. The **VAE decoder** turns the final latent into pixels.

Measured component sizes in Stable Diffusion 1.5: U-Net ≈ 860 M parameters, CLIP text encoder ≈ 123 M, VAE ≈ 84 M. The **scheduler** (sampler) is not a network and can be swapped without retraining.

![Latent diffusion pipeline](fig:ldm_pipeline)

## 4. Sampling, guidance and prompts

### Samplers

| Sampler | Type | Typical steps |
|---|---|---|
| DDPM ancestral | Stochastic reverse chain | 250–1000 |
| DDIM (Song, Meng & Ermon, 2020) | Deterministic update on selected timesteps (noise parameter η = 0) | 20–100 |
| DPM-Solver++, Euler, Heun | ODE solvers; Euler is first order | 15–30 |
| Distilled (consistency models, adversarial diffusion distillation) | Student model trained to jump far | 1–4 |

**DDIM update.** From the noise prediction $\hat\epsilon$ at step $t$, estimate the clean image and jump to an earlier step $t'$:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat\epsilon}{\sqrt{\bar{\alpha}_t}}, \qquad x_{t'} = \sqrt{\bar{\alpha}_{t'}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t'}}\,\hat\epsilon$$

The same trained network is used; only the sampling procedure changes.

**Code distinction:** `sample_ddim(steps=50)` selects timesteps spanning index `T - 1` down to 0 and uses both current and next schedule coefficients in each jump. Lowering `sample_ddpm(steps)` merely truncates its reverse loop and starts at a lower index with fresh Gaussian noise; it is not the same faster sampler. Use the full DDPM schedule for that baseline, and use DDIM for the skipping comparison.

![Effect of the number of steps](fig:sd_steps)

### Classifier-free guidance (Ho & Salimans, 2022)

During training the prompt is replaced by an empty prompt about 10% of the time, so one network learns both $\epsilon_\theta(x_t, c)$ and $\epsilon_\theta(x_t, \varnothing)$. At sampling:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + w\left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right)$$

$w = 1$ is the plain conditional model; $w \approx 5$–$8$ is typical; very large $w$ over-saturates colours and reduces diversity. Each step needs two network evaluations (batched together in practice).

![Classifier-free guidance as vector extrapolation](fig:cfg)

![Effect of the guidance scale](fig:sd_guidance)

### Seeds and negative prompts

The **seed** fixes the starting noise (and any sampling noise); same seed and settings give the same image. A **negative prompt** replaces $\varnothing$ in the guidance formula, so sampling moves away from it. Negative prompts are useful but blunt: they cannot reliably remove small details.

![Seeds and negative prompts](fig:seeds_negative)

**Measured timings on the lecture machine** ({{SD_DEVICE}}): {{SD_TIMES}}. Always report hardware with timings.

## 5. Variants, evaluation and responsible use

### Flow matching and rectified flow

Flow matching (Lipman et al., 2022) and rectified flow (Liu et al., 2022) learn a **velocity field** that transports noise to data along simple, usually straight, paths:

$$x_t = (1 - t)\, x_0 + t\, \epsilon, \qquad \mathcal{L}_{\mathrm{FM}} = \mathbb{E}\left[\left\| v_\theta(x_t, t) - (\epsilon - x_0) \right\|^2\right]$$

Sampling integrates the learned ODE from noise ($t = 1$) to data ($t = 0$). Straighter paths need fewer steps; Stable Diffusion 3 and FLUX use rectified flow with transformer backbones. Conventions for the direction of time differ between papers.

![Curved diffusion paths vs straight flow-matching paths](fig:flow_paths)

### Distillation

A student model is trained to reproduce in one or a few steps what the teacher sampler does in many. **Consistency models** (Song et al., 2023) and **adversarial diffusion distillation** (SDXL-Turbo, SD-Turbo; which adds a GAN discriminator) produce usable images in 1–4 steps, with some loss of diversity and control.

![SD-Turbo with 1, 2 and 4 steps](fig:sd_turbo)

### Variants and applications

* **Editing:** image-to-image (SDEdit: noise part-way, then denoise with a new prompt), inpainting and outpainting.
* **Control:** ControlNet adds edges, depth or pose as conditions; IP-Adapter conditions on a reference image.
* **Personalisation:** DreamBooth and LoRA (week 8) teach new subjects or styles from a few images.
* **Video:** spatio-temporal diffusion transformers (e.g. the Sora and Veo families).
* **Audio:** latent diffusion over spectrograms or audio-codec latents.
* **Science:** protein and molecule design, weather forecasting, materials.

### Evaluating image generators

| Aspect | Metric | Notes |
|---|---|---|
| Realism and diversity | FID, KID | Reference set, feature network and sample size must match |
| Prompt alignment | CLIP score (cosine similarity ×100) | Fooled by keywords; weak on counting and relations |
| Compositional correctness | GenEval, T2I-CompBench | Detector-based checks of objects, counts, colours, positions |
| Human preference | Pairwise votes, preference models, public arenas | Most trusted; needs rubric and multiple raters |
| Efficiency | Steps, seconds per image, memory | Report hardware |

### The families compared

| | VAE | GAN | Diffusion | Flow matching |
|---|---|---|---|---|
| Loss | ELBO | Adversarial | Noise regression (ELBO-based) | Velocity regression |
| Stability | High | Low | High | High |
| Quality | Blurry | Sharp | Very high | Very high |
| Diversity | Good | Collapse risk | Good | Good |
| Sampling | 1 pass | 1 pass | 20–50 (1–4 distilled) | Few |
| 2026 role | Latent compression | Super-resolution, distillation | Image, video, audio | Newest image/video models |

# Common misconceptions

| Misconception | Correction |
|---|---|
| "The network removes all the noise in one go." | It predicts the noise at one noise level; generation applies many small steps (or a distilled few). |
| "Training loops over all 1000 steps for each image." | Each training example uses one random timestep; only sampling loops. |
| "Diffusion runs on pixels in Stable Diffusion." | It runs in the VAE's latent space; pixels appear only after decoding. |
| "Higher guidance is always better." | It trades diversity and naturalness for prompt adherence; very high values over-saturate. |
| "The seed is irrelevant." | The seed determines the starting noise; record it for reproducibility. |
| "Changing the sampler requires retraining." | Samplers use the same noise predictor; swap them freely (quality may change). |
| "Generated images are always new." | Models can memorise and reproduce duplicated training images. |

# Responsible AI lens: diffusion models

* **Training data and copyright:** large scraped image–text datasets; disputes over consent and licensing continue in several jurisdictions. Do not treat the legal position as settled.
* **Memorisation:** Carlini et al. (2023) extracted near-copies of training images from diffusion models, especially heavily duplicated ones: a privacy and copyright risk.
* **Bias:** audits have found strong occupational, gender and racial stereotypes in generated people; test prompts systematically across groups.
* **Misuse:** keep safety filters on; do not generate identifiable real people; provenance (C2PA, watermarks) and EU AI Act Article 50 disclosure for synthetic content.
* **Licences:** Stable Diffusion 1.5 uses the CreativeML OpenRAIL-M licence with use restrictions; SD-Turbo is a research release whose commercial use needs Stability AI's licence. Check before project use.
* **Energy:** many steps per image; distillation and smaller models reduce the footprint.

# Lab guide and answers

**Runtime:** Colab T4 GPU. Part A (2-D and MNIST diffusion) also runs on CPU with fewer epochs. Part B needs about 4 GB of GPU memory for SD 1.5 in float16; without a GPU the notebook falls back to SD-Turbo. **Hand-in:** trained models, sweep grids, DDPM vs DDIM timing, CLIP-score table and three written answers.

## TODOs

* **TODO 1** `a.sqrt() * x0 + (1 - a).sqrt() * eps`; at t = 999 the signal fraction is ≈ 4 × 10⁻⁵.
* **TODO 2** random t per example, `eps = randn_like`, `xt = q_sample(...)`, `F.mse_loss(model(xt, t), eps)`.
* **TODO 3** DDPM step: `(x - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sqrt(beta_t) * z` (no noise at t = 0).
* **TODO 4** DDIM: `x0_hat = (x - sqrt(1 - a_t) * eps) / sqrt(a_t)`; `x = sqrt(a_next) * x0_hat + sqrt(1 - a_next) * eps`. This uses 50 rather than 1000 network calls; measure actual runtime and quality. Unlike lowering `sample_ddpm(steps)`, the selected timesteps span the full noise schedule.
* **TODO 5** negative prompt passed as `negative_prompt=...`.
* **TODO 6** `F.cosine_similarity(img_emb, txt_emb)`. Expect the control (unrelated prompt) to score clearly lower (≈ 15–20 vs 28–35 on the ×100 scale).

## Troubleshooting

* *CUDA out of memory with SD 1.5:* restart the runtime; use `pipe.enable_attention_slicing()` or `pipe.enable_model_cpu_offload()`.
* *Black images:* the safety checker replaced a flagged image, or float16 overflow on some older GPUs; change the prompt or use float32.
* *Slow MNIST training on CPU:* reduce to 1–2 epochs; digits will be rougher but recognisable.

# Practice questions with model answers

## Multiple choice

1. In $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$, as $t \to T$: **(a)** $\bar{\alpha}_t \to 1$; **(b)** $x_t$ approaches pure noise; **(c)** $x_t \to x_0$; **(d)** the variance explodes. *Answer: (b).*
2. Classifier-free guidance requires: **(a)** a separate classifier; **(b)** training with the prompt sometimes dropped; **(c)** a GAN discriminator; **(d)** 1000 steps. *Answer: (b).*
3. In Stable Diffusion the text influences the denoiser through: **(a)** the VAE; **(b)** cross-attention; **(c)** the seed; **(d)** the scheduler. *Answer: (b).*
4. DDIM differs from DDPM mainly in: **(a)** the training loss; **(b)** the network architecture; **(c)** a deterministic sampler that skips steps; **(d)** the dataset. *Answer: (c).*

## Short answer

1. **Compute $x_t$ for $x_0 = 1.0$, $\bar{\alpha}_t = 0.64$, $\epsilon = 0.5$.** (2 marks) *Model answer:* $0.8 \times 1.0 + 0.6 \times 0.5 = 1.1$.
2. **Why is diffusion training stable compared with GAN training?** (3 marks) *Model answer:* each step is a fixed regression target (the known noise) with a mean-squared-error loss; no adversary or moving objective; the loss derives from a likelihood bound, so there is no incentive to drop modes.
3. **Explain latent diffusion and why it matters in practice.** (4 marks) *Model answer:* a VAE compresses images (e.g. 48×) and diffusion runs in the latent space, conditioned on text via cross-attention; the decoder maps back to pixels (2); it reduces compute and memory enough to train and run high-resolution models on modest GPUs, enabling open models such as Stable Diffusion (2).
4. **Describe the effect of the guidance scale using the CFG formula.** (4 marks) *Model answer:* the guided prediction extrapolates from the unconditional towards the conditional prediction by factor w (2); larger w increases prompt adherence but reduces diversity and can over-saturate; w = 1 recovers the conditional model (2).

## Exam-style question

**"Explain how a text-to-image latent diffusion model is trained and how it generates an image from a prompt. Discuss two ways generation has been made faster and how you would evaluate the resulting images."** (20 marks)

*Marking guide:* forward process and closed form (3); noise-prediction objective and its link to the ELBO (3); latent diffusion components: VAE, text encoder, denoiser with cross-attention (4); sampling with classifier-free guidance (3); two speed-ups with trade-offs, e.g. ODE samplers, distillation, flow matching (4); evaluation with FID, CLIP score and human preference including limitations (3).

# Glossary

| Term | Meaning |
|---|---|
| Forward process | Fixed Markov chain that gradually adds Gaussian noise |
| Noise schedule | Sequence β_t controlling how much noise is added per step |
| ᾱ_t | Product of (1 − β_s); fraction of original signal variance remaining |
| Reverse process | Learned chain that removes noise step by step |
| Noise prediction ε_θ | Network output: the noise present in x_t |
| Score | Gradient of the log density; equivalent (up to scale) to noise prediction |
| DDPM / DDIM | Original stochastic sampler / deterministic faster sampler |
| U-Net | Convolutional encoder–decoder with skip connections |
| Diffusion transformer (DiT) | Transformer over latent patches used as the denoiser |
| Latent diffusion | Diffusion in a VAE's latent space |
| Cross-attention | Mechanism letting image features attend to text tokens |
| Classifier-free guidance | Extrapolating conditional vs unconditional predictions by w |
| Negative prompt | Prompt whose direction guidance moves away from |
| Flow matching / rectified flow | Learning a velocity field along (straight) noise-to-data paths |
| Distillation | Training a few-step student to mimic a many-step sampler |
| CLIP score | Cosine similarity between CLIP embeddings of image and prompt |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Foster, D. (2023) *Generative Deep Learning*, 2nd ed.: chapter 8, "Diffusion Models" (builds a DDIM model in Keras).

**Papers**

* Ho, Jain & Abbeel (2020) [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
* Rombach et al. (2022) [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* Ho & Salimans (2022) [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
* Lipman et al. (2022) [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
* Peebles & Xie (2022) [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

**Courses, explainers and videos**

* [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1): notebooks with the Diffusers library.
* [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/index)
* [MIT 6.S184: Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)
* [3Blue1Brown / Welch Labs: But how do AI images and videos actually work?](https://www.youtube.com/watch?v=iv-5mZ_9CPY)
* [Lilian Weng: What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Jay Alammar: The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)

# Link to the group project

Most image-generation projects in 2026 should start from a pretrained diffusion or flow model rather than training from scratch. Ask groups to (1) choose a model and check its licence, (2) define the controls they will expose (prompt templates, guidance, seeds, negative prompts), (3) plan an evaluation combining CLIP score or FID with a small human study, and (4) record all settings for reproducibility. Groups needing a specific style or subject can plan a LoRA fine-tune (week 8).
