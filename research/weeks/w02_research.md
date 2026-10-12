# R1 · Week 02: variational autoencoders

Checked **2026-10-10**. Read: built slide titles and lab headings only. Proposals below are teaching designs, not measured results.

**Explanations to adapt (redraw; do not copy).**
- A probabilistic encoder returns a cloud of possible codes rather than one postcode. The [original VAE paper](https://arxiv.org/html/1312.6114v11), §2, motivates a learned approximate posterior. Adapt to a sketch of a digit with uncertain position; latent coordinates need not correspond to named human properties.
- Teach the objective as two bills: reconstruction mismatch and departure from a chosen prior. Connect these to the [PyTorch VAE example](https://github.com/pytorch/examples/blob/main/vae/main.py), then name the negative ELBO. The bills are mathematical losses, not monetary costs; show how sum/mean choices affect their relative scale.
- Reparameterisation moves randomness into a separate noise draw: choose noise, stretch it, shift it. Work `mu=2`, `sigma=0.5`, `epsilon=-1` → `z=1.5` before the equation. This is a worked adaptation of [Kingma and Welling §2.4](https://arxiv.org/html/1312.6114v11), not an experimental result.

**Five misconceptions to confront.** (1) The encoder directly outputs a single latent sample. (2) `logvar` is the standard deviation. (3) The ELBO is the exact data likelihood. (4) Lower reconstruction error alone proves better generation. (5) Increasing beta always improves both reconstruction and useful latent structure. State that beta changes the objective and requires comparison, not a guaranteed outcome.

**Current practice and versions.** [Diffusers `AutoencoderKL`](https://huggingface.co/docs/diffusers/api/models/autoencoderkl) documents VAE image encoding and latent decoding. Its defaults are not properties of every VAE. R0's official Colab snapshot records torch `2.11.0+cpu`, diffusers `0.40.0`, ipywidgets `7.7.1`; E2 must validate compatibility. This lab need not install diffusion packages. The PyTorch example shows reparameterisation and separate BCE/KL terms.

**Own-style visuals and beginner barriers.** Use image `[B,784]` → encoder → `mu,logvar [B,d]` → reparameterise → decoder → reconstructed pixels. Feed labelled noise into the sampling block; use a dashed gradient/feedback arrow only where appropriate and a role legend. Add a two-column reconstruction/generation diagram and an ELBO cost strip. Titles jump from likelihood integrals to Gaussian KL; start with one scalar Gaussian and explain every symbol. Lab barriers: `forward`, tuple unpacking, `exp(0.5*logvar)`, batch reduction and interpreting a 2-D plot of a 16-D latent space. State explicitly when only two coordinates vary.

**CPU core proposal, target ≤15 minutes of waiting; unmeasured.** Train one small dense MNIST VAE on a bounded subset; log reconstruction and KL separately; decode fixed latent points. Use a slider for decoding coordinates or display measured beta runs, keeping repeated beta training and conditional VAE as stretch. Check positive scales, shapes, finite losses and generation without an input image. Optional Gemini: explain a synthetic two-term loss table; it cannot validate generation quality. Use the shared opt-in helper/local interpretation questions. R0's unresolved EEA/Swiss/UK API-client terms remain applicable; no free-access promise or billing change.

**Further reading:** [Original VAE paper](https://arxiv.org/abs/1312.6114) · [PyTorch VAE implementation](https://github.com/pytorch/examples/blob/main/vae/main.py) · [Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl) · [MIT 6.S191 2026 generative-modeling video](https://www.youtube.com/watch?v=R8V8CbuxryI).

Handoff: research complete; CPU timing, model quality and live Gemini/widgets still require implementation and verification.
