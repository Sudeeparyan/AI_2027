# W04 lecture activation handoff — 11 October 2026

- Activated after W01 pilot completion; all original/candidate hashes matched. Compare `source_hashes.json` before applying the five candidate sources; preserve any new edits.
- Static precheck passed: 40 core slides; 46 expected after beginner enrichment; 47 modern notes, 146–171 words; diagram bodies 36–40 words including legends. Original equations on core slides 10, 13, 14, 27 and 32 are unchanged.
- Three visible quizzes retain answers in the notes; the hook and recap add prediction/discussion prompts. Word notes retain the MSc theory and add numerical examples, notation, FAQ and five answered checks.
- Build, coverage/prose/math audits, render sheets, Word pages and notebook images are pending with the coordinator. No runtime or visual-quality pass is claimed.

| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 04 | weeks/week_04.yaml; beginner/week_04.json | All | Old notes did not use the complete teaching template | Major | Authored ordered six-section notes with defined terms and worked examples | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | weeks/week_04.yaml | 6, 8, 16, 23, 36–40 | Dense lists and duplicated comparisons limited prediction and visual teaching | Major | Shortened bodies, added E3 example strips/mini-tables and two quizzes; retained detail in Word | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | beginner/week_04.json | Five diagrams | Fixed arithmetic, learned prediction and numerical updates were conflated | Major | Explicit roles, labelled data arrows, known-noise target branches, separate loss/update and null arrows between independent exercises | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | figures/week_04.py | U-Net/DiT; latent pipeline | Skip features used feedback styling; scheduler and predictor were combined | Major | Solid saved-feature routes; separate denoiser/scheduler blocks; role legends | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | figures/week_04.py; notes/week_04.md | Flow paths | Endpoint labels contradicted the data-at-zero interpolation | Major | Consistent data t=0 and noise t=1; distinguish training pairs from learned trajectories | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | notes/week_04.md; weeks/week_04.yaml | Lab/evidence | Historical digit/image timings could be mistaken for the new point core | Major | Label stored measured runs and pending new-core timing; keep every result placeholder | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | notes/week_04.md; labs/week_04_lab.py | Optional feedback | Provider choice changed during preparation | Minor | Shared ask_model with local rubric; exactly one Gemini/OpenAI switch, account/region/cost caveat | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

Proposed glossary additions/clarifications for M1 (merge only if absent):

| Term | Plain definition |
|---|---|
| Denoiser | A learned network that predicts noise or another correction from a noisy state and its noise level. |
| Sampler | A numerical rule that changes the state during generation using the network prediction. |
| Noise schedule | Fixed coefficients specifying how signal and noise change across levels. |
| DDPM | Denoising diffusion probabilistic model; this lab samples reverse steps with fresh noise before the final step. |
| DDIM | Denoising diffusion implicit model; this lab uses deterministic updates at selected levels with fixed learned weights. |
| Latent | A numerical representation processed instead of the original pixels; a decoder maps it back to pixels. |
| Flow matching | Training a network to predict velocities along chosen paths between data and noise. |
| Distillation | Training a student model from a teacher's behaviour to reduce generation work. |
| Cosine similarity | Direction similarity: dot product divided by both vector lengths. It does not establish every aspect of image quality. |

Primary-source proposals for M1, checked 11 October 2026:

- [DDPM paper](https://arxiv.org/abs/2006.11239), [latent diffusion paper](https://arxiv.org/abs/2112.10752), [flow matching paper](https://arxiv.org/abs/2210.02747).
- [Diffusers DDIM documentation](https://huggingface.co/docs/diffusers/en/api/schedulers/ddim).
- [SD-Turbo official model card](https://huggingface.co/stabilityai/sd-turbo): distilled SD 2.1-derived model, documented 1–4 steps and guidance disabled.
- [Historical SD 1.5 mirror card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5): the mirror is not affiliated with Runway. Keep checkpoint identity separate from historical experiment labels.
- [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32), [European Commission Article 50 FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act).
- Existing W04 research video retained: [Hugging Face diffusion course walkthrough](https://www.youtube.com/watch?v=09o5cv6u76c); playback not checked in this draft pass.

Handoff: ready to apply after the coordinator's exact pilot-completion instruction. Recheck original source hashes, copy the five candidates and these logs, then coordinate rebuild, audits, smoke/full CPU evidence and visual QA.

Activation: applied 11 October 2026 after W01 pilot completion; every original source and candidate hash matched. YAML applied with edit_slides. Rebuild, audits, CPU smoke/full measurement and visual QA remain pending with root.

Audit correction, 11 October: coverage anchors restored with explanatory paragraphs; diagram budgets now include check footers; lab map has a truthful independent vector input joined at reporting; code is labelled concise pseudocode with full implementation retained in Word/notebook; manual Word slide headings aligned; latent-pipeline labels separated. Source prechecks passed; coordinator rebuild/audits/render still pending.

## V · Visual QA (Claude, 11 Oct 2026)

| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 04 | labs/week_04_lab.py | Part B DDPM/DDIM plots | 1,500 updates: samples a diffuse cloud, not the spiral | Medium | Timed 1.5k/3k/6k/10k updates (26 s at 10k): core 10,000 updates; samples trace the spiral | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | weeks/week_04.yaml, notes | 44; Word p? (lab guide) | 'timing pending' / 'still needs measurement' | Medium | {{W04_CORE_TIME}} placeholders from ledger | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 04 | weeks/week_04.yaml | 2 | Outcomes terse | Low | Kept: four MIMLOs leave a 6-word budget; detail stays in notes | Closed: accepted teaching/layout decision; final view confirmed |

## Final coordinated closure — 12 October 2026
- Final current coverage: 46 slides, 33 Word pages, 11 delivered diagram PNGs and 13 notebook images; actual views plus byte-identical previously viewed images are documented in the final review manifests. Layout report [], zero open findings.
- Latest successful cached local CPU full core: 39.4 s (0.7 min), recorded in `research/lab_runs_2026-10.json`; final smoke PASS (12.9 s). Current executed core code matches the successful full-run evidence.
- All 26 current release gates passed, including coverage 100%, strict notes, body ≤ 40, fonts ≥ 14, semantic helpers and shared diagram/attachment parity. See `research/ITERATION3_2026-10-12.json`.
- No timed Colab, clean-install/download, live-provider, browser-widget or optional GPU/stretch claim. Earlier pending handoffs above describe intermediate states.
