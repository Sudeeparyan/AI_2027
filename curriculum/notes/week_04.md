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

## Use one scalar before an image tensor

Let $x_0 = 1$, $\epsilon = -0.5$ and $\bar{\alpha}_t = 0.64$. The noisy value is 0.8 × 1 + 0.6 × (−0.5) = 0.5. Ask which value the network should predict: the added noise −0.5. The trainer knows that draw; the network only sees noisy input and timestep. Lost signal cannot identify the original image uniquely.

**Separate the two loops.** Training draws a random timestep and updates weights. Generation follows decreasing noise levels with fixed weights. Predict the next operation, run a small example, inspect it, then explain it aloud. Repeat until students can narrate both loops without saying the model retrains while generating.

**Connect to the pipeline.** Prompt vectors condition a smaller image latent. The sampler chooses the numerical updates; the VAE decoder supplies final pixels. Ask why a text-to-image request does not require a real image or the VAE encoder.

**Keep claims testable.** Full DDPM and schedule-spanning DDIM use one denoiser. Flow matching learns a marginal velocity from training pairs, so actual sample trajectories can curve. Quality, coverage, guidance effects and speed require measured evidence.



## 1. Why diffusion?

| | VAE | GAN | Diffusion |
|---|---|---|---|
| Sample quality | May blur | May be sharp | Can be detailed; evaluate |
| Coverage | Evaluate | Mode collapse risk | Evaluate |
| Training | One objective; failures occur | Moving game | Fixed-target regression; failures occur |
| Text conditioning | Possible | Possible | Often useful; test prompt fit |
| Sampling | 1 pass | 1 pass | 10–1000 passes (reducible) |

Noise-prediction diffusion uses one regression objective instead of an adversarial game. Traditional sampling repeats many network calls. The **generative trilemma** (quality, diversity, speed) names three things to measure on the actual task. Distillation and suitable samplers can reduce calls; no family guarantees quality or coverage.

![The generative trilemma](fig:trilemma)

**Core idea.** Create a training example by adding known Gaussian noise to real data. Tell a network the noise level and train it to predict that noise. This is **regression**, prediction of numerical values.

To generate, start from a new noisy state. Use the prediction and a sampler to step towards less noisy data, then repeat. One trained network handles many noise levels. Starting noise supplies variety. A prompt can guide content. See Sohl-Dickstein et al. (2015) and Ho, Jain & Abbeel (2020).

## 2. Forward and reverse processes

Keep three quantities separate: $x_0$ is clean data, $x_t$ is noisy data at timestep $t$, and $\epsilon$ is the actual noise added. A schedule supplies known coefficients. The network receives $x_t$ and $t$, while the sampled $\epsilon$ remains a separate target for the training loss.

### The forward process

With a small noise schedule $\beta_1, \ldots, \beta_T$ (DDPM: linear from $10^{-4}$ to $0.02$, $T = 1000$):

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)$$

**One-number example.** With $\beta_t = 0.1$ and previous value 1, the next-state Gaussian has mean $\sqrt{0.9}$, about 0.949, and variance 0.1. A sampled next value need not equal the mean.

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Because Gaussians compose, we can jump directly to any step:

$$q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right) \quad\Leftrightarrow\quad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

**Worked example.** Let $x_0 = 1$, $\epsilon = -0.5$ and $\bar{\alpha}_t = 0.64$. Then $x_t$ = 0.8 × 1 + 0.6 × (−0.5) = 0.5. The target noise remains −0.5.

This is the reparameterisation trick from week 2. For initially unit-variance data the scaling keeps marginal variance at one. In general the variance is $\bar{\alpha}_t$ times the data variance plus $1 - \bar{\alpha}_t$, tending towards one. At $t = T$, $\bar{\alpha}_T \approx 4 \times 10^{-5}$: essentially pure noise.

**Worked example.** For a pixel with clean value $x_0 = 0.8$, at a step where $\bar{\alpha}_t = 0.36$ and with sampled noise $\epsilon = -0.5$: $x_t = \sqrt{0.36}(0.8) + \sqrt{0.64}(-0.5) = 0.6 \times 0.8 - 0.8 \times 0.5 = 0.48 - 0.40 = 0.08$.

![The forward process on real digits](fig:forward_mnist_slide)

### Noise schedules

A **noise schedule** controls how much signal remains at each timestep. Linear and cosine schedules remove signal at different rates. Late steps in the shown linear schedule contain almost no original signal.

The **signal-to-noise ratio** compares signal variance with noise variance. For clean data standardised to unit variance, $\mathrm{SNR}(t) = \bar{\alpha}_t/(1-\bar{\alpha}_t)$. For clean variance $v$, the numerator is $\bar{\alpha}_t v$. If $\bar{\alpha}_t = 0.8$, SNR = 0.8/0.2 = 4. Use this ratio to compare noise levels across schedules (Nichol & Dhariwal, 2021; Karras et al., 2022).

![Linear vs cosine schedules](fig:schedules)

### The reverse process

For small $\beta_t$ the reverse step is approximately Gaussian, so it is modelled as

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\, \sigma_t^2 I\right), \qquad \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)$$

with $\sigma_t^2 = \beta_t$ in the simplest DDPM. One network $\epsilon_\theta$, shared over all steps, receives the timestep through a sinusoidal embedding.

**Reverse-mean example.** Let $x_t = 0.5$, $\beta_t = 0.1$, $\alpha_t = 0.9$, $\bar{\alpha}_t = 0.81$ and predicted noise = 0.2. The displayed mean is approximately 0.479. DDPM may add sampling noise after this calculation.

### The training objective

$$\mathcal{L}_{\mathrm{simple}}(\theta) = \mathbb{E}_{x_0,\, t,\, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\; t\right)\right\|^2\right]$$

**Squared-error example.** If one true noise value is −0.5 and the prediction is −0.3, its squared error is (−0.5 + 0.3)² = 0.04. Average these errors over the tensor and examples.

1. Sample a training image $x_0$, a timestep $t \sim U\{1, \ldots, T\}$ and noise $\epsilon \sim \mathcal{N}(0, I)$.
2. Form $x_t$ with the closed form.
3. Predict the noise and minimise the squared error.

The simplified noise loss connects to the **ELBO**, the lower bound from week 2. A diffusion model has a series of hidden noisy states. Its forward chain is fixed, rather than learned by an encoder. Ho et al. derive a reweighted training loss from the bound.

A direct numerical target makes training simpler than a two-player GAN game. It often works well. Neither the loss nor likelihood interpretation guarantees sample quality or complete data coverage.

```python
t   = torch.randint(0, T, (batch,), device=x0.device)
eps = torch.randn_like(x0)
a   = abar.to(x0.device)[t].view(batch, 1, 1, 1)
xt  = a.sqrt() * x0 + (1 - a).sqrt() * eps
loss = F.mse_loss(model(xt, t), eps)
```

Training costs one network evaluation per example; **sampling** costs one per step (1000 for ancestral DDPM).

### The score-based view

The **score** is $\nabla_x \log p(x)$, a gradient pointing towards more likely data. At noise level t, a noise prediction can be rescaled into a score estimate: $\epsilon_\theta(x_t,t) \approx -\sqrt{1-\bar{\alpha}_t}\,\nabla_{x_t}\log p_t(x_t)$.

A **stochastic differential equation (SDE)** describes a continuous path with random changes. Song et al. (2021) relate diffusion to this path and a deterministic **probability-flow ODE**. An ODE describes a changing state without fresh random noise. Numerical solvers approximate these paths using discrete steps.

### Real results

A three-layer MLP trained with $\mathcal{L}_{\mathrm{simple}}$ on a 2-D swiss roll turns Gaussian noise into the spiral during the reverse process. A tiny U-Net ({{UNET_PARAMS}} parameters) trained for {{MNIST_EPOCHS}} epochs on MNIST ({{MNIST_DEVICE}}, about {{MNIST_MIN}} minutes) generates recognisable digits; structure appears in the middle steps and detail at the end (coarse-to-fine).

![Reverse process on 2-D data](fig:reverse_2d_slide)

![Reverse process on MNIST](fig:reverse_mnist)

![Samples from the tiny MNIST diffusion model](fig:mnist_samples)

## 3. Architecture and latent diffusion

### Denoiser networks

* **U-Net** (Ronneberger et al., 2015; used by DDPM, Stable Diffusion 1.x/2.x, SDXL): convolutional encoder–decoder with skip connections; attention layers at low resolution; output in this noise-prediction setup has the input’s shape.
* **Diffusion transformer (DiT)** (Peebles & Xie, 2022): split the latent into patches (tokens) and process them with transformer blocks; timestep and applicable conditions enter through adaptive normalisation or joint attention. Transformer backbones appear in Stable Diffusion 3, FLUX and some video models. Their performance still depends on data, model size and compute.

Both require timestep information. Where conditions enter depends on the architecture; the MNIST teaching model uses timesteps without text.

![U-Net and diffusion transformer](fig:unet_dit)

### Latent diffusion (Rombach et al., 2022)

In the **Stable Diffusion 1.5 example**:

1. A **VAE** (week 2) compresses a 512 × 512 × 3 image to a 64 × 64 × 4 latent (48× fewer numbers).
2. A **text encoder** (CLIP's text transformer in SD 1.5; T5 as well in later models) turns the prompt into token embeddings.
3. The **denoiser** runs the reverse process in latent space and reads the text through **cross-attention** (week 5). Later architectures can condition on text differently.
4. The **VAE decoder** turns the final latent into pixels.

Parameter counts of the Stable Diffusion 1.5 components, counted from the downloaded weights: U-Net ≈ 860 M, CLIP text encoder ≈ 123 M, VAE ≈ 84 M. The **scheduler** (sampler) is not a network. Compatible schedules and prediction conventions can be swapped without retraining the denoiser.

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

**DDIM example.** With $x_t = 0.5$, $\bar{\alpha}_t = 0.64$ and predicted noise −0.5, $\hat{x}_0 = (0.5 - 0.6 \times (-0.5))/0.8 = 1$. At the final clean coefficient $\bar{\alpha} = 1$, the deterministic update returns 1.

The same trained network is used; only the sampling procedure changes.

**Code distinction:** `sample_ddim(steps=50)` selects timesteps spanning index `T - 1` down to 0 and uses both current and next schedule coefficients in each jump. Lowering `sample_ddpm(steps)` merely truncates its reverse loop and starts at a lower index with fresh Gaussian noise; it is not the same faster sampler. Use the full DDPM schedule for that baseline, and use DDIM for the skipping comparison.

![Effect of the number of steps](fig:sd_steps)

### Classifier-free guidance (Ho & Salimans, 2022)

During training the condition is sometimes dropped; the chosen dropout rate depends on the model. One network learns both $\epsilon_\theta(x_t, c)$ and $\epsilon_\theta(x_t, \varnothing)$. At sampling:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + w\left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right)$$

**Guidance example.** For one predicted noise value, let unconditioned = 0.2, conditioned = 0.1 and w = 3. The guided prediction is 0.2 + 3 × (0.1 − 0.2) = −0.1. Guidance can extrapolate beyond either prediction.

$w = 1$ gives the plain conditional prediction in this formula. Values around 5–8 are common in SD 1.5 examples, but useful settings depend on the model. Very large $w$ can over-saturate colours and reduce variety. Active CFG uses conditional and reference predictions, often batched together. Some pipelines skip the reference evaluation when CFG is disabled; the lab's SD-Turbo path forces guidance to zero.

![Classifier-free guidance as vector extrapolation](fig:cfg)

![Effect of the guidance scale](fig:sd_guidance)

### Seeds and negative prompts

A **seed** sets the random-number sequence used for starting and sampling noise. A fixed setup makes comparisons easier; model/software revisions, hardware and nondeterministic operations can still change the image. A **negative prompt** replaces $\varnothing$ in the guidance formula, so sampling moves away from it. Negative prompts are useful but blunt: they cannot reliably remove small details. In the lecture run, the wall lamp is still present with the negative prompt "lamp, blurry, low quality"; the same seed only changed the room's layout.

![Seeds and negative prompts](fig:seeds_negative_slide)

**Measured timings on the lecture machine** ({{SD_DEVICE}}): {{SD_TIMES}}. Always report hardware with timings.

## 5. Variants, evaluation and responsible use

### Flow matching and rectified flow

Flow matching (Lipman et al., 2022) and rectified flow (Liu et al., 2022) learn a **velocity field** from selected training paths. This illustration uses straight paths between paired data and noise points:

$$x_t = (1 - t)\, x_0 + t\, \epsilon, \qquad \mathcal{L}_{\mathrm{FM}} = \mathbb{E}\left[\left\| v_\theta(x_t, t) - (\epsilon - x_0) \right\|^2\right]$$

**Flow example.** Let $x_0 = 0.8$, $\epsilon = -0.2$ and $t = 0.25$. The mixed value is 0.55, and target velocity is −1.0. A backward time step uses that velocity to move towards the data end.

Sampling integrates the learned ODE from noise ($t = 1$) to data ($t = 0$). Simpler training paths can help sampling efficiency, but learned marginal trajectories need not be straight and accurate few-step generation is not automatic; Stable Diffusion 3 and FLUX use rectified flow with transformer backbones. Conventions for the direction of time differ between papers.

![Schematic curved VP training pairs and straight linear training pairs](fig:flow_paths)

### Distillation

**Distillation** trains a faster student model to imitate a slower teacher. The teacher takes many generation steps. The student aims to produce a similar result in fewer calls. Consistency models and adversarial diffusion distillation, including SD-Turbo and SDXL-Turbo, can use 1–4 steps. Compare speed, detail, variety and prompt control rather than assuming the faster output is equivalent.

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
| Human preference | Pairwise votes, preference models, public arenas | Needs a rubric, multiple raters and agreement checks |
| Efficiency | Steps, seconds per image, memory | Report hardware |

### The families compared

| | VAE | GAN | Diffusion | Flow matching |
|---|---|---|---|---|
| Loss | ELBO | Adversarial | Noise regression (ELBO-based) | Velocity regression |
| Training | One objective | Moving game | Noise regression | Velocity regression |
| Quality | Evaluate | Evaluate | Evaluate | Evaluate |
| Diversity | Evaluate | Collapse risk; evaluate | Evaluate | Evaluate |
| Sampling | One decoder pass | One generator pass | Many calls; some distilled models use 1–4 | Numerical solver calls; evaluate cost |
| 2026 role | Latent compression | Super-resolution, distillation | Image, video, audio | Newest image/video models |

# Common misconceptions

| Misconception | Correction |
|---|---|
| "The network removes all the noise in one go." | It predicts the noise at one noise level; generation applies many small steps (or a distilled few). |
| "Training loops over all 1000 steps for each image." | Each training example uses one random timestep; only sampling loops. |
| "Diffusion runs on pixels in Stable Diffusion." | It runs in the VAE's latent space; pixels appear only after decoding. |
| "Higher guidance is always better." | It can strengthen prompt influence but reduce variety or introduce artefacts. Test an appropriate range for the actual model. |
| "The seed is irrelevant." | The seed determines the starting noise; record it for reproducibility. |
| "Changing the sampler requires retraining." | Samplers use the same noise predictor; use compatible sampler settings and schedules without retraining; quality may change. |
| "Generated images are always new." | Models can memorise and reproduce duplicated training images. |

# Responsible AI lens: diffusion models

* **Training data and copyright:** large scraped image–text datasets; disputes over consent and licensing continue in several jurisdictions. Do not treat the legal position as settled.
* **Memorisation:** Carlini et al. (2023) extracted near-copies of training images from diffusion models, especially heavily duplicated ones: a privacy and copyright risk.
* **Bias:** audits have found strong occupational, gender and racial stereotypes in generated people; test prompts systematically across groups.
* **Misuse:** keep safety filters on; do not generate identifiable real people; record source history (provenance), including supported C2PA or watermark tools. EU AI Act Article 50 distinguishes provider marking from deployer disclosure, with scope and exceptions; consult the official text.
* **Licences:** Stable Diffusion 1.5 uses the CreativeML OpenRAIL-M licence with use restrictions; SD-Turbo’s official model card lists both non-commercial and commercial uses subject to the applicable Stability AI licence. Read the exact model licence and current terms before project use.
* **Energy:** fewer network calls can reduce per-image cost. Measure the actual workload and include extra training costs when comparing distilled models.

# Lab guide and answers

**Runtime:** a Colab T4 GPU is intended. On a GPU with at least 8 GB of memory the notebook trains six MNIST epochs and uses SD 1.5 in float16; this GPU path was not timed for this release, and older GPUs may need float32. Without such a GPU the notebook trains two MNIST epochs and falls back to SD-Turbo for Part B. On the laptop CPU used to check this pack (9 October 2026), that CPU path ran end to end in about 17 minutes, of which the two MNIST epochs took about 11. **Hand-in:** trained models, sweep grids, DDPM vs DDIM timing, CLIP-score table and three written answers.

## TODOs

* **TODO 1** `a.sqrt() * x0 + (1 - a).sqrt() * eps`; at t = 999 the signal fraction is ≈ 4 × 10⁻⁵.
* **TODO 2** random t per example, `eps = randn_like`, `xt = q_sample(...)`, `F.mse_loss(model(xt, t), eps)`.
* **TODO 3** DDPM step: `(x - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sqrt(beta_t) * z` (no noise at t = 0).
* **TODO 4** DDIM: `x0_hat = (x - sqrt(1 - a_t) * eps) / sqrt(a_t)`; `x = sqrt(a_next) * x0_hat + sqrt(1 - a_next) * eps`. This uses 50 rather than 1000 network calls; measure actual runtime and quality. Unlike lowering `sample_ddpm(steps)`, the selected timesteps span the full noise schedule.
* **TODO 5** negative prompt passed as `negative_prompt=...`.
* **TODO 6** `F.cosine_similarity(img_emb, txt_emb)`. Compare matched and unrelated prompts, record the actual scores and inspect the images. A matched prompt often scores higher, but the ordering and values are not guaranteed; the score can miss object counts or relations.

## Troubleshooting

* *CUDA out of memory with SD 1.5:* restart the runtime; use `pipe.enable_attention_slicing()` or `pipe.enable_model_cpu_offload()`.
* *Black images:* the safety checker replaced a flagged image, or float16 overflow on some older GPUs; change the prompt or use float32.
* *Slow MNIST training on CPU:* reduce to 1–2 epochs and inspect the samples; less training may produce poor or unrecognisable digits.

# Practice questions with model answers

## Multiple choice

1. In $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$, as $t \to T$: **(a)** $x_t$ approaches pure noise; **(b)** $\bar{\alpha}_t \to 1$; **(c)** $x_t \to x_0$; **(d)** the variance explodes. *Answer: (a).*
2. Classifier-free guidance requires: **(a)** a separately trained image classifier; **(b)** 1000 sampling steps for every image; **(c)** a GAN discriminator to score each step; **(d)** training with the prompt sometimes dropped. *Answer: (d).*
3. In Stable Diffusion the text influences the denoiser through: **(a)** the VAE; **(b)** cross-attention; **(c)** the seed; **(d)** the scheduler. *Answer: (b).*
4. DDIM differs from DDPM mainly in: **(a)** the training loss used to fit the noise predictor; **(b)** the architecture of the denoising network it trains; **(c)** a selected-step sampler, deterministic at $\eta = 0$ in this lab; **(d)** the dataset and the image resolution it is trained on. *Answer: (c).*

## Short answer

1. **Compute $x_t$ for $x_0 = 1.0$, $\bar{\alpha}_t = 0.64$, $\epsilon = 0.5$.** (2 marks) *Model answer:* $0.8 \times 1.0 + 0.6 \times 0.5 = 1.1$.
2. **Why does noise-prediction training avoid the adversarial game used by GANs?** (3 marks) *Model answer:* each step is a fixed regression target (the known noise) with a mean-squared-error loss; no adversary or moving objective; the loss derives from a likelihood bound, but finite data, capacity and optimisation can still miss parts of the distribution.
3. **Explain latent diffusion and why it matters in practice.** (4 marks) *Model answer:* a VAE compresses images (e.g. 48×) and diffusion runs in the latent space, conditioned on text via cross-attention; the decoder maps back to pixels (2); it reduces compute and memory enough to train and run high-resolution models on modest GPUs, enabling open models such as Stable Diffusion (2).
4. **Describe the effect of the guidance scale using the CFG formula.** (4 marks) *Model answer:* the guided prediction extrapolates from the unconditional towards the conditional prediction by factor w (2); larger w can strengthen prompt influence, reduce diversity or over-saturate; w = 1 recovers the conditional model (2).

## Exam-style question

**"Explain how a text-to-image latent diffusion model is trained and how it generates an image from a prompt. Discuss two ways generation has been made faster and how you would evaluate the resulting images."** (20 marks)

*Marking guide:* forward process and closed form (3); noise-prediction objective and its link to the ELBO (3); latent diffusion components: VAE, text encoder, denoiser with cross-attention (4); sampling with classifier-free guidance (3); two speed-ups with trade-offs, e.g. ODE samplers, distillation, flow matching (4); evaluation with FID, CLIP score and human preference including limitations (3).

# Glossary

| Term | Meaning |
|---|---|
| Forward process | Fixed Markov chain that gradually adds Gaussian noise |
| Noise schedule | Sequence $\beta_t$ controlling how much noise is added per step |
| $\bar{\alpha}_t$ | Product of $(1 - \beta_s)$ for $s = 1, \ldots, t$; fraction of original signal variance remaining |
| Reverse process | Learned chain that removes noise step by step |
| Noise prediction $\epsilon_\theta$ | Network output: the noise present in $x_t$ |
| Score | Gradient of the log density; equivalent (up to scale) to noise prediction |
| DDPM / DDIM | Stochastic reverse chain / selected-step sampler, deterministic when η = 0 |
| U-Net | Convolutional encoder–decoder with skip connections |
| Diffusion transformer (DiT) | Transformer over latent patches used as the denoiser |
| Latent diffusion | Diffusion in a VAE's latent space |
| Cross-attention | Mechanism letting image features attend to text tokens |
| Classifier-free guidance | Extrapolating conditional vs unconditional predictions by w |
| Negative prompt | Prompt whose direction guidance moves away from |
| Flow matching / rectified flow | Learn velocity targets from selected paired paths; sampling trajectories can curve |
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


# Detailed explanations behind the short slides

The lecture shows the essential idea first. These fuller explanations retain the supporting comparisons, examples and checks for revision.

## Slide 6: Limitations of VAEs and GANs that diffusion addresses

|  | VAE (week 2) | GAN (week 3) | Diffusion (today) |
|---|---|---|---|
| Sample quality | May blur | May be sharp | Can be detailed; evaluate |
| Coverage / diversity | Evaluate | Mode collapse risk | Evaluate |
| Training | One objective; failures occur | Moving game | Fixed-target regression; training still needs checks |
| Conditioning on text | Possible | Possible | Often useful; test prompt fit |
| Sampling speed | One pass | One pass | Many passes (10–1000), now reduced |

## Slide 7: The generative trilemma: quality, diversity, speed

- A generator should make useful images, cover varied examples and run quickly.

- Traditional diffusion often gives strong quality and variety but takes many steps.

- **Distillation** trains a faster model to imitate a slower one. Some models use 1–4 steps.

- Compare quality, variety and speed on your own task.

## Slide 8: The core idea in one slide

- **Forward process**: add known random noise to a real image. This rule is fixed.

- **Training**: show a noisy image and its noise level. Ask the network to predict the added noise.

- **Reverse process**: use that prediction to take a step towards a less noisy image.

- **Generation**: start with noise and repeat reverse steps.

- A different starting noise can give a different image. A text condition helps guide what it shows.

Training knows the noise target. Generation reuses the trained predictor at many noise levels.

## Slide 10: The forward (noising) process

- At each forward step, keep some signal and add Gaussian noise. $\beta_t$ controls the noise amount.

- The closed form creates a noise level directly: $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$.

- For the shown 1000-step schedule, almost no signal remains at the end. Training can choose any t without running earlier steps.

## Slide 12: Noise schedules decide how fast information is destroyed

- A **noise schedule** sets how much signal remains at each timestep.

- The linear schedule leaves many late steps close to pure noise.

- A cosine schedule spreads the signal loss differently. It can improve results.

- **Signal-to-noise ratio** compares the remaining image signal with the noise amount.

## Slide 13: The reverse (denoising) process is learned

- The same network predicts noise at every timestep. The timestep tells it the current noise level.

- Use predicted noise $\epsilon_\theta$ and the schedule to calculate the mean of the next, less noisy state.

- DDPM also adds fresh sampling noise except at the final step. Small forward steps motivate the Gaussian reverse approximation.

## Slide 14: Training objective: predict the noise

- Choose a real example $x_0$, a random step $t$ and random noise $\epsilon$. Create noisy $x_t$ directly.

- Predict $\epsilon$ from $x_t$ and $t$. Minimise mean squared error, the average squared difference between true and predicted noise.

- This simplified loss comes from a reweighted ELBO. The fixed noising process plays the encoder role in a hierarchical latent model.

## Slide 15: Training and sampling in a few lines

- One training example uses one randomly chosen noise level.

- Generation repeatedly calls the same trained network and updates the noisy state.

- The lab implements the noising rule, training loss and reverse loop.

## Slide 16: The score-based view (Song et al., 2021)

- The **score** is the gradient of log probability: a direction towards more likely noisy data.

- A noise prediction can be converted into a score estimate using the stated noise-level scale.

- A **stochastic differential equation (SDE)** describes continuous changes with random noise.

- A **probability-flow ODE** describes a related deterministic path. Numerical solvers can take fewer steps.

## Slide 18: A tiny diffusion model on MNIST, trained for this lecture

- Tiny U-Net (under 1 M parameters), a few epochs.

- Structure appears in the middle steps; details are refined at the end.

- The lab trains the same model: six epochs on a Colab GPU, or two on a CPU.

## Slide 21: Two denoiser architectures: U-Net and diffusion transformer

- **U-Net** shrinks image features, expands them and joins saved features with skip connections.

- **Diffusion transformer (DiT)** processes image or latent patches as tokens.

- Both receive the timestep. Text-conditioned versions also receive text information.

- The denoiser output matches the input's numerical shape in our noise-prediction setup.

## Slide 22: Latent diffusion: the Stable Diffusion pipeline

- A VAE compresses a 512 × 512 × 3 image into 64 × 64 × 4 values in this Stable Diffusion example.

- A **text encoder** turns the prompt into vectors, numerical descriptions of its text pieces.

- The denoiser works on the smaller hidden state. **Cross-attention** lets it read the prompt vectors.

- The VAE decoder turns the final state into pixels.

## Slide 23: What each component contributes

**Autoencoder (VAE).** Compress images to fewer numbers and decode the finished hidden state. Some fine detail may change.

**Text encoder.** Turn prompt tokens into vectors. Prompt understanding depends partly on this model.

**Denoiser.** Predict noise from the noisy state, timestep and any prompt information.

**Scheduler / sampler.** Use predictions to update the state. Compatible samplers can change without retraining the denoiser.

## Slide 25: Samplers: from 1000 steps to a handful

**DDPM (ancestral).** 

- Adds fresh random noise at reverse steps

- The original sampler can use hundreds to 1000 steps

- Uses the trained noise predictor and schedule

**ODE solvers.** 

- DDIM at $\eta = 0$ gives deterministic discrete updates; DPM-Solver, Euler and Heun are ODE methods

- Choose fewer update steps, often 20–50

- Repeatability also depends on model, software and hardware

- Use a compatible sampler and schedule

**Distilled models.** 

- Train a faster model to imitate a slower one

- Examples: consistency models and SDXL-Turbo

- Some use 1–4 steps

- Measure any change in detail, variety and prompt fit

## Slide 27: Classifier-free guidance (Ho & Salimans, 2022)

- During training, sometimes drop the text condition. One model learns prompted and unprompted predictions.

- During generation, calculate both predictions. Guidance weight w strengthens their difference.

- With this convention, w = 1 is the conditional prediction. Large w can reduce naturalness and variety. Useful ranges depend on the model.

## Slide 28: Effect of the guidance scale (same seed)

- The grid holds the prompt and seed fixed and changes guidance w.

- Compare prompt fit, colours and visual artefacts across the settings.

- This run has a useful balance around w = 3-7.5. It is not a rule for every model.

- Very high guidance can reduce naturalness and variety.

## Slide 29: Seeds and negative prompts

- A **seed** sets the random-number sequence used to draw starting noise.

- Record the seed, model, sampler and settings. Software and hardware can still affect exact repeatability.

- A **negative prompt** replaces the empty condition in guidance, pushing away from its features. It does not guarantee their removal.

## Slide 32: The flow-matching objective

- Mix clean data and random noise using t. Here t = 0 is data and t = 1 is noise.

- The target velocity $\epsilon - x_0$ gives the direction and size of change along the training line.

- Train $v_\theta$ with squared error. Generate by taking steps backwards from t = 1 to t = 0. Some papers reverse this convention.

## Slide 33: Few-step generation with a distilled model (SD-Turbo)

- SD-Turbo is a distilled Stable Diffusion 2.1 model trained with an adversarial loss.

- It can generate with one denoiser call and guidance disabled.

- Compare time, image detail, variety and prompt fit. This example uses 512-pixel outputs.

## Slide 34: Beyond text-to-image: diffusion variants

**Editing.** Image-to-image changes a noised input. Inpainting fills a selected region. Outpainting extends the border.

**Control.** ControlNet uses edges, depth or pose. IP-Adapter uses a reference image to guide generation.

**Personalisation.** DreamBooth or LoRA adapts weights using a few authorised examples of a subject or style (week 8).

**Video.** Add time-aware processing so neighbouring frames form a coherent clip.

**Audio.** Generate using noise and denoising over sound representations, such as spectrograms.

**Science.** Use related methods for proteins, molecules, weather or material design.

## Slide 35: Evaluating image generators

| Aspect | Metric | Notes |
|---|---|---|
| Realism and variety | FID, KID | Compare many real and generated images using a stated feature network. |
| Prompt fit | CLIP score | Compare image/text vector similarity. High similarity can still hide counting or layout errors. |
| Correct objects and relations | GenEval, T2I-CompBench | Test counts, colours and positions. Detector-based checks also make mistakes. |
| Human preference | Pairwise ratings | Use a clear rubric and several raters. Preferences can differ. |
| Efficiency | Steps, seconds per image, memory | State the hardware and model settings. |

## Slide 36: Comparing the generative families (weeks 2–4)

|  | VAE | GAN | Diffusion | Flow matching |
|---|---|---|---|---|
| Training loss | ELBO | Adversarial game | Noise regression (ELBO-based) | Velocity regression |
| Training | One objective | Moving game | Noise regression | Velocity regression |
| Sample quality | Evaluate | Evaluate | Evaluate | Evaluate |
| Diversity | Evaluate | Collapse risk; evaluate | Evaluate | Evaluate |
| Sampling cost | One decoder pass | One generator pass | Many calls; some distilled models use 1–4 | Numerical solver calls; evaluate cost |
| Main use in 2026 | Latent compression | Super-resolution, distillation | Image, video, audio | Latest image/video models |

## Slide 37: Responsible AI lens: diffusion models

- **Training data and copyright:** models trained on billions of scraped image–text pairs; legal disputes over consent and licensing continue in several countries.

- **Memorisation:** researchers extracted near-copies of training images from diffusion models (Carlini et al., 2023), especially duplicated ones.

- **Bias:** prompts such as 'a CEO' or 'a nurse' reproduce stereotypes; audit outputs across groups.

- **Misuse and source history (provenance):** safety filters, watermarks and C2PA credentials (week 10); EU AI Act Art. 50 labelling.

- **Energy:** fewer calls can reduce per-image cost; measure the workload and include any extra training cost.

## Slide 38: Lab 4: train a diffusion model and use Stable Diffusion

Google Colab T4 GPU recommended (without one, Part A trains fewer epochs and Part B falls back to SD-Turbo)

Notebook with trained models, sweep grids, timing and CLIP-score table, and written answers

- Implement direct noising at a selected timestep and view noisy images.

- Train noise predictors on 2-D data and digits. Implement reverse sampling.

- Load a pretrained text-to-image model with Hugging Face Diffusers.

- Hold controls fixed while testing steps, guidance, seeds, negative prompts and samplers.

- Compare SD-Turbo with the longer-step pipeline. Measure time per image.

- Calculate CLIP similarity, inspect prompt fit and report a visible failure case.

## Slide 39: Summary

- A fixed rule adds noise. A trained network predicts noise at any timestep.

- Create $x_t$ directly using signal and noise weights. Train with squared noise error.

- Generation starts with noise and repeats reverse updates using the same network.

- Latent diffusion combines a VAE, text encoder and denoiser.

- Guidance changes prompt influence. Seeds and negative prompts help control experiments.

- Flow matching predicts velocity. Distillation teaches a faster model.

- Evaluate image quality, prompt fit, variety and speed together.

- Check data rights, memorisation, bias and source history.


## Expanded training and sampling fragment

The projected fragment calls the lab helper for readability. This expanded DDPM update uses a timestep tensor for the whole image batch. It assumes the lab model and schedules have been created; the training optimiser update follows the loss calculation. Sampling reuses trained weights with evaluation mode and gradients disabled.

```python
# ---- training step
t   = torch.randint(0, T, (batch,), device=x0.device)
eps = torch.randn_like(x0)
a   = abar.to(x0.device)[t].view(batch, 1, 1, 1)  # image batch
xt  = a.sqrt() * x0 + (1 - a).sqrt() * eps
loss = F.mse_loss(model(xt, t), eps)

# ---- sampling (DDPM ancestral)
x = torch.randn(n, 1, 28, 28, device=x0.device)
for t in reversed(range(T)):
    tb = torch.full((n,), t, device=x.device, dtype=torch.long)
    e = model(x, tb)
    x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
    if t > 0:
        x = x + betas[t].sqrt() * torch.randn_like(x)
```
