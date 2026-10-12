# R2 · Week 04: diffusion

Checked **2026-10-10**. Scope: built slide titles and lab headings. Proposals need implementation and timing.

**Teaching explanations (adapt, then state the limit).**
- Progressively repair a noisy drawing: inspect corruption before noise prediction. Generation starts from new noise, without recovering a particular drawing. [Hugging Face course](https://huggingface.co/learn/diffusion-course/en/unit1/1).
- Turn training into a labelled exercise: clean sample + known noise + timestep → noisy sample; predict the known noise. Show `[batch, 2]` points before image tensors. This is the noise-prediction DDPM objective, not every diffusion parameterisation. [DDPM paper](https://arxiv.org/abs/2006.11239).
- A sampling schedule resembles choosing intermediate stops on a journey. The scheduler updates the sample; the learned network predicts its direction. Stops must suit the checkpoint, rather than being interchangeable. This analogy is our adaptation. [Diffusers schedulers](https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers).

**Five misconceptions to correct.**

| Misconception | Correction |
|---|---|
| Reverse diffusion undoes an exact photograph | It samples a learned distribution. |
| Training runs the entire reverse loop | A random timestep supplies each training example. |
| More steps always improve any model | Distilled models have their own operating range. |
| Stronger guidance guarantees better images | Guidance changes a tradeoff; it is model-dependent. |
| A seed fixes output on every computer | Software, hardware and settings also matter. |

**Current practice.** Label Stable Diffusion 1.5 historical. Official `stabilityai/sd-turbo` uses distillation from Stable Diffusion 2.1, 1–4 steps, `guidance_scale=0.0`, and no negative-prompt guidance; exclude Turbo from ordinary guidance experiments. [Model card](https://huggingface.co/stabilityai/sd-turbo). Diffusers documents flow-matching schedulers for Flux/Qwen-Image; discuss without large downloads. R0's published Colab versions (`torch 2.11.0`, `diffusers 0.40.0`, `transformers 5.18.0`, `ipywidgets 7.7.1`) require compatibility testing. [R0](platform_2026.md).

**Redraw.** Separate clean/noise/timestep/denoiser/loss training from sampling. Label shapes, solid data arrows and dashed repetition; add a role legend. Latent diagrams must show encoding/decoding, avoiding implied pixel-space denoising.

**Beginner barriers.** Titles jump from DDPM to scores, latent diffusion, guidance and flow matching. Lab headings also require a U-Net, pretrained image pipeline and CLIP. Teach a single 2-D noise example first; move score/flow derivations and image-model installation to guided stretch sections.

**CPU core proposal: target ≤15 minutes, unmeasured.** Generate synthetic 2-D points locally; train a small timestep-conditioned MLP with capped updates; sample a short reverse trajectory. Print shapes; check finite loss and output shape. A noise-level slider changes the forward plot without retraining (`skip-test`). Compare measured stored image grids for steps/guidance. MNIST training, Stable Diffusion and CLIP downloads are optional stretches. Measure downloads, setup and computation before accepting the core.

**Gemini role.** Optional feedback on a synthetic prompt-comparison explanation, never required image generation. Use shared `USE_GEMINI=False`, local answer rubric and few cached calls. R0 flags unresolved EEA/UK distributed-client terms: do not promise regional free eligibility. [R0](platform_2026.md).

**Further reading:** [course](https://huggingface.co/learn/diffusion-course/en/unit1/1), [DDPM](https://arxiv.org/abs/2006.11239), [schedulers](https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers), [SD-Turbo card](https://huggingface.co/stabilityai/sd-turbo), [course walkthrough video](https://www.youtube.com/watch?v=09o5cv6u76c).

Handoff: research complete; primary sources checked. No lab/API/widget execution; CPU timing and Colab compatibility await B/V.
