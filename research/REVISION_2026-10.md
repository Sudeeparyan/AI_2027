# October 2026 beginner teaching-pack revision

Status on 10 October 2026: ready for review. The audit has no open rows. Every check in the evidence table below was run on the final build of 10 October, except the full lab runs of 9 October, whose code has not changed since (see that row). The report states what each check covers and what was not checked.

## What students now receive

Every week keeps 46 slides and the Section 7.3 topic order. The opening course map highlights the current class. Starting vocabulary comes before the core explanation. Five technical diagrams trace the overall system, mechanism, training or preparation, use, and notebook workflow. Each diagram has a concrete worked example and a check with an instructor answer. The last slide is a six-card recap infographic with an illustrative example, exit question and bridge to the next class.

Across the course, these are 84 shared figures: 60 technical diagrams, 12 highlighted course maps and 12 recaps. Each appears in editable PowerPoint objects, SVG and PNG; Word and both notebooks use the exact PNG bytes. Notebook explanations sit beside the code they introduce. Every executable cell has a short purpose, reason and expected observation. Student TODOs and instructor answers remain separate.

The common glossary contains 123 plain-language definitions. The teacher's improvement loop is: ask students to predict, trace, run, change and explain; use the final exit question to record a misconception; begin the next class with a clearer example. Full equations and extended explanations remain in the Word notes, while projected text is shorter.

Open [the visual gallery](revision_review.html) to browse all twelve recaps, diagrams and rendered slides. [The audit](AUDIT_2026-10.md) records concrete defects and closure evidence. Sources remain editable under `curriculum/`; generated teaching files are under `deliverables/`.

## Changes by week

| Week | Main explanation and accuracy changes |
|---|---|
| 1 | Trace adjacent character counts, smoothing, probabilities, scoring and sampling before introducing a neural model. Separate model probability from factual confidence, training from generation, and deterministic decoding from positive-temperature sampling. GAN generation uses its generator; the discriminator belongs to training. Unsaved old numerical observations are explicitly historical and unverified. |
| 2 | Separate reconstruction from prior sampling. Explain reparameterisation with small numbers, then ELBO/KL symbols. Identify greyscale BCE as a conventional soft-target surrogate. Explain what regularisation and free bits do without promising filled latent space or a minimum actual KL. Correct stored AUC rounding. |
| 3 | Show discriminator and generator updates separately. Explain detach as a gradient boundary and optimiser ownership as the update boundary. Plot saturation against the discriminator logit. Qualify FID/KID, interpolation, conditional training and toy comparisons. Explain Article 50 roles and exceptions against the official text. |
| 4 | Trace a numerical noising example before the equation. Separate training and repeated denoising. Correct image-batch coefficient shapes and batched timesteps in lecture code. Distinguish DDIM eta=0, ODEs and flow matching. Qualify guidance, scheduler compatibility and seed reproducibility. Add a practical CPU route for small image-generation GPUs. |
| 5 | Complete Q/K and V branches, target-to-loss route, vocabulary head and residual additions. Match the worked value vectors to the lab. Mask scores before softmax; create the mask on the tensor's device. Explain the KV cache using the last known token. Qualify attention interpretation, multihead costs and efficiency claims. Use evaluation mode during generation. |
| 6 | Separate architecture, access, licence and model capability. Show conceptual pre-training/SFT/preference tuning without implying the notebook implements training. Distinguish local tokenisation from hosted message requests. Explain completion and token/time budgets for thinking. Correct licence examples and optional-client imports. |
| 7 | Define each scorer before interpreting accuracy. Validate an eight-digit ASCII ID, catch validation errors and validate a repaired reply again. Require valid order-swapped judge verdicts selecting the same underlying answer. Report injection keyword flags as a heuristic requiring reply inspection. Give context engineering a readable full-width diagram. |
| 8 | Separate SFT objectives from which weights change. Show the parallel LoRA paths and distinguish prompt from prefix tuning. Remove an invented fallback parameter chart; use the stored measured counts. Match percentage rounding across chart/body. Cite both recorded full CPU runs (76 and 53 minutes of training, identical losses) and explain tiny smoke mode and prepared classroom results. |
| 9 | Trace image/text vectors through a shared space, separate training loss from frozen-feature measurement, and name ln N as the uniform-choice baseline. Correct retrieval precision terminology. Preserve the historical PCA plotting pixels with a readable companion legend. Restore extended tables and equation readings. |
| 10 | Separate VLM, image generation/editing, speech and provenance tasks. A prompt's image score is compared with the stated mean, not assumed best among all prompts. WER checks transcription, not naturalness. Plain PNG metadata is not C2PA and provenance is not truth. Correct SD-Turbo licence guidance and explain smaller-device fallback. |
| 11 | Trace local generation, the HTTP API and interface as distinct routes. Bound server startup, allocate the lab's own port, and propagate producer errors out of streaming. Explain measured TTFT, throughput, cache identity and memory estimates separately. Qualify client compatibility, local costs, Docker and fake quantisation. |
| 12 | Separate indexing, retrieval, generation and tool execution. Show source/citation checks, missing-answer abstention and valid ranking metrics. Validate tool-call objects and arguments before approval/execution. Keep email simulated. Explain blocked observations, bounded loops, permissions and MCP as a protocol rather than a safety guarantee. |

## Diagram and document pipeline

Shared scenes use consistent labelled roles, forward data routes, explicit branches and dashed repetition. Arial glyph measurements guide wrapping; native and portable renderers share geometry and split multiline labels consistently. Round-corner adjustments are restricted to rounded rectangles: applying them to ellipses made PowerPoint reject a deck. Code excerpts and ordinary instructional text now have a 14-point floor; the actual PPTX text audit complements manual review of raster figures.

The course map replaces the old agenda's visible list; timings remain in Word. The recap replaces the old summary and moves to slide 46. Original summary detail remains in Word. Slide-reference ranges account for this move. The glossary is generated from one canonical source; its table spacing was adjusted to avoid an orphan final row. Extended Word tables and fenced code are restored rather than flattened into dense paragraphs.

## Final review round (9–10 October)

A second pass looked at every rendered slide and Word page of all twelve weeks and checked each number against the stored results. The defects it found are rows in [the audit](AUDIT_2026-10.md); the main kinds were:

- **Speaker notes.** Weeks 6–12 had pasted one Analogy, Ask and Common mistake onto up to 12 slides of a section, and Week 5 labelled answers as mistakes. Every slide in Weeks 5–12 now has its own Ask, Answer and Common mistake.
- **Claims checked against the stored runs.** For example, the Week 9 confusion-matrix notes now name the measured confusions, the Week 10 image-to-image text describes what the stored images show, and the Week 7 ROUGE-1 example now gives F1 ≈ 0.71 from unrounded recall and precision.
- **Lab times.** Every complete solution notebook was run in full (see the table below). Slides, notes, lab headers and the instructor guide now give these measured times and the hardware. They no longer give estimates for a Colab T4 that was never timed.
- **Quizzes.** 21 of the 24 slide quizzes and 33 of the 45 practice questions had the same answer letter, and in most of them the correct option was also clearly the longest. The answers are now spread across all four letters, distractors are written as fully as the answers, and the explanations name the right letters.
- **Word layout.** In weeks 9–12, each step of a flow slide and the lab's runtime and hand-in boxes printed as loose bullets without labels; each step is now one labelled bullet with the slide's name for it. Option labels such as "(c)" and numbers before units ("14 GB") no longer end a line, and each exit question stays on the page of its answer.
- **Current names and exact numbers.** Vertex AI and Azure AI Foundry now appear under their current names, Gemini Enterprise Agent Platform and Microsoft Foundry, with the former name in brackets. Week 11's batching text prints the chart's 13.2 and 86.2 tokens/s, so the printed values give the stated 6.5×.
- **Typography and notation.** Minus signs, en dashes, ×, →, sub- and superscripts, Greek letters and a `<document>` tag that Word dropped were fixed. Figure text that projected below 14 pt was enlarged, and diagram legends now list only the roles a diagram uses.

New automatic checks catch these defects if they come back: `audit_prose.py` (REPEAT, LABEL, QUIZ, TAG, ARROW, APPROX, HYPHEN and others), `audit_flat_bullets.py`, `audit_inline_math.py` (now including spelled-out Greek letters before a superscript, such as "sigma²"), `audit_figure_text.py` and `lint_figures.py`. `research/REVIEW_PROCESS.md` describes how to repeat the review.

## Evidence (final build, 10 October 2026)

| Check | Observed evidence |
|---|---|
| Weekly builds | `build_week.py --weeks 1-12`: all twelve weeks built, each with 46 slides; no unfilled placeholder. |
| Course build | `build_course.py`: instructor guide, common glossary (123 terms) and Section 7.3 rationale. |
| Lab helpers / runtime helpers | 8 and 24 tests passed (streaming failure, server startup, occupied and dynamic ports, malformed calls, simulated approval, image-device selection). |
| Beginner support / verifier | 3 tests passed; `verify_beginner.py` 12 of 12 packs passed. |
| Technical diagrams | `test_technical_diagrams.js`: 60 authored routes and 60 scenes with measured Arial text widths passed, plus 15 legacy feedback cases and scene parity. |
| Technical figures | 10 tests passed; `lint_figures.py`: 110 drawn figures, 0 with layout issues. |
| Teaching notes | `test_teaching_notes.js`: table columns, download sizes and resumed numbering passed. |
| Text audits | `audit_inline_math.py` 0 problems; `audit_prose.py` 0 problems; `audit_slide_text.py` 0 text runs below 14 pt in every week; `audit_figure_text.py` 0 pictures projecting text below 14 pt; `audit_flat_bullets.py` 0 problems. |
| Coverage | `audit_coverage.py`: 214 of 214 Section 7.3 Detail items found (week counts 26, 16, 17, 16, 19, 22, 18, 20, 13, 14, 14, 19). |
| Protected inputs | `verify_pilot_artifacts.py`: all 7 checks passed; all 2,633 files in `context/`, `sources/` and `legacy/` have unchanged SHA-256 hashes. |
| Links | `check_links.py`: 176 of 180 URLs reachable, 0 failures; the 4 others return 403 to automated requests (the three publisher book pages and the ACM Stochastic Parrots page, see below). |
| Office render | `render.py --weeks 1-12` in PowerPoint and Word: 552 slides and 376 Word pages, and `slides.layout.json` reports no overflowing text box in any deck. The Word notes were rendered again after the last builder change, and Week 11's slides after its slide 24 text changed. The changed pages were viewed again after each fix. |
| Smoke runs | `test_labs.py --weeks 1-12` with smoke settings (CPU, offline, `.venv-labs`): 12 of 12 passed, from 21 s (Week 2) to 481 s (Week 7). Week 12 ran again after one of its instructor answers changed (101 s). Each run's notebook hash (`build/week_NN/lab_test_smoke.json`) equals the delivered solution notebook. |
| Full lab runs (9 October) | Every solution notebook ran to the end with `test_labs.py --full`, models already downloaded. Minutes: W1 0.9, W2 9.3, W3 19.2, W4 16.6, W6 8.4, W8 84.1 and W10 6.9 on the laptop CPU; W5 10.1, W7 58.3, W9 0.8, W11 3.3 and W12 4.6 on an NVIDIA GTX 1650 (4 GB). Records: `research/lab_runs_2026-10.json`. After these runs, Markdown, comments and lab headers changed (for example to state the measured times). A comparison of code cells found no other code change, except two metadata fields added to Week 1's results export. |

No failed or interrupted run is counted as a pass.

## Measurement and verification limits

Stored measurements change only when a complete new run replaces them through the results → `fill_results.py` → `fill.json` pipeline. Smoke settings check that the code runs; they do not replace lecture results. Full runs on 9 October reproduced the stored numbers of weeks 7, 9 and 12 exactly, the Week 10 answers and scores, and the Week 8 losses to six decimals. Week 5's notes now quote a full run of the notebook itself (validation loss 1.56) beside the archived experiment (1.51). Week 11's repeat GPU run measured lower throughput than the stored one (23.4 instead of 48.3 tokens/s for a batch of 4), so the stored timings are kept and the notes say that timings depend on hardware and load. Week 2's measured AUC is 0.9995; the redrawn chart prints four decimals (the old saved plot rounded it to 1.000). Week 8's stored counts, accuracies and training times are observations from recorded runs, not promises for a new student run. Week 12's stored answers come from Qwen2.5-1.5B on the GPU, while the CPU route uses the 0.5B model.

All times come from one laptop. They do not cover a Colab T4 session, a clean package installation or download time, other GPU models, the GPU route of the weeks that were timed on the CPU (and the reverse), optional hosted APIs, public Gradio links or a built Docker deployment. The local runner skips install and interactive cells on purpose. The image-lab policy selects the CPU below 8 GB of VRAM and accepts an explicit CPU override; its branches are unit-tested, not benchmarked on every graphics card.

The four blocked URLs are the Foster and Tunstall O'Reilly book pages, the Rothman Packt book page and the ACM Stochastic Parrots page. The three publisher titles were also opened through the browser search service; the ACM page still blocked access, while [the author's official page](https://faculty.washington.edu/ebender/stochasticparrots/) confirms the paper and original link. This is not a claim that the complete books or blocked paper were downloaded and re-read.

Beginner readability has been reviewed against worked traces, vocabulary, diagrams and actual page renders. Student understanding still needs classroom feedback; the exit questions and teacher log support that continuing improvement.

## Primary sources and provenance

The reference ledger is `curriculum/sources.yaml`. Revision research uses original papers and official documentation, including [VAE](https://arxiv.org/abs/1312.6114), [GAN](https://arxiv.org/abs/1406.2661), [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), [flow matching](https://arxiv.org/abs/2210.02747), [attention](https://arxiv.org/abs/1706.03762), [Chinchilla](https://arxiv.org/abs/2203.15556), [self-consistency](https://arxiv.org/abs/2203.11171), [LLM judging](https://arxiv.org/abs/2306.05685), [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [CLIP](https://arxiv.org/abs/2103.00020), [Pydantic constraints](https://docs.pydantic.dev/latest/concepts/fields/), [TRL SFT](https://huggingface.co/docs/trl/en/sft_trainer), [PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html), and release-specific model cards.

SD-Turbo guidance was checked against its [official licence file](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md): conditional commercial use under the applicable Community License is distinct from a blanket non-commercial label. Current legal statements refer to official text and identify roles/scope rather than promising legal compliance.

New diagrams are authored code/vector scenes. Historical result graphics retain their measured data; wrappers and legends improve readability without inventing observations. No generated bitmap artwork substitutes for technical evidence. Starting-state snapshots and protected hashes are under `build/revision_before_20261008/`; only Weeks 1–4 had fresh baseline renders before the initial Office failure at Week 5. The gallery labels other starting images as saved snapshots.

No commit, push or publication was performed.
