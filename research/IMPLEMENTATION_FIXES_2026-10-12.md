# Implementation corrections — completed 13 October 2026

The thirteen independent implementation findings have been repaired. Further visual review also corrected teaching order, figure parity, answer alignment and misleading captions. Changes make existing explanations more concrete; the module order, learning outcomes, MSc content and 46-slide deck sizes are retained.

34 release gates and the final affected-artifact checks passed; all twelve smoke runs and clean pinned full CPU cores passed. The longest core was Week 8: 728.7 seconds (12.15 minutes), within the 15-minute budget. Visual review covers 552 slides, 412 weekly Word pages, 103 diagram PNGs, 130 notebook images and 54 course pages. All twelve slide layout reports are empty; zero open findings. Current Office render inputs, inspected image hashes and full-run executable code match the distributed artifacts.

| Original finding | Correction and evidence | Status |
|---|---|---|
| A01 / W08 | Hosted prompts use entirely invented demonstrations and query; sentinel regression passed. | Closed |
| A02 / W08 | Trainer explicitly follows the selected CPU; actual device configuration and full CPU fine-tuning passed. | Closed |
| A03 / W07–08 | Optional OpenAI caps use the documented minimum; local Week 7 routing keeps its separate eight-token cap. | Closed |
| A04 / shared | Both providers share a 40-attempt budget per notebook kernel, including bounded retries; rerunning setup preserves it. Setup-rerun regressions passed. | Closed |
| A05 / shared | Local .env key files require explicit selection and provider opt-in; no automatic key-file selection. | Closed |
| A06 / W11 | Complete direct pins installed successfully in a clean local environment; pip check and SDK constructors passed. | Closed |
| A07 / W07–08 | Both labs separate development from the frozen final test. Week 8 checks adapter, training-data and settings fingerprints before reusing final-test results. | Closed |
| A08 / evidence | Unsupported archived runtimes removed; current measured results promoted with dated records and hashes. Archives preserved. | Closed |
| A09 / teaching | Worked examples precede the relevant slide formulas; final Word references were checked for consistency. | Closed |
| A10 / glossary | 253 canonical terms and all twelve mappings generate matching weekly/course definitions; glossary tests passed. | Closed |
| A11 / W10 | Numerical projector/loss/gradient/update trace added and independently checked; final diagrams inspected. | Closed |
| A12 / labs | Core/change/stretch/common-error support, semantic checks and missing-widget fallbacks completed and reviewed. | Closed |
| A13 / W02 | KL weight comparison is qualified by magnitude, beta and training; final rendered wording checked. | Closed |

An actual clean pinned Week 8 run exposed a Windows fused-loss failure. The portable loss keeps real LoRA training and is checked against loss values, gradients and batch normalisation. The successful full rerun took 728.7 seconds; failed attempts remain in the logs.

| Visual finding | Week | Location | Problem corrected | Status |
|---|---|---|---|---|
| V01 | 02 | slide 9 / Word page 10 | Word renders latent_ae_vs_vae with numeric axes and vertical colourbar; slide uses latent_ae_vs_vae_slide with bottom legend and different titles. Copies differ. | Closed after replacement review |
| V02 | 03 | slide 23 / Word page 13 | Word mode_collapse adds per-panel modes-covered counts and different titles; the slide uses mode_collapse_slide. Copies differ. | Closed after replacement review |
| V03 | 03 | slide 31 / Word page 15 | Word epochs figure lacks the explanatory row subtitles shown by epochs_slide in PowerPoint. Copies differ. | Closed after replacement review |
| V04 | 03 | slide 30 / Word page 8 | Check asks whether inference updates weights; its supplied answer describes which network generates, without answering the weight-update question. | Closed after replacement review |
| V05 | 03 | Word page 14 / curriculum/notes/week_03.md:159 | ADA augmentation is said to make memorisation impossible: so it cannot memorise small datasets. It reduces overfitting risk without that guarantee. | Closed after replacement review |
| V06-01 | 06 | 9,14 | Worked numerical examples appear below equations, with no earlier visible worked example. | Closed after replacement review |
| V07-01 | 07 | ROUGE-1 | Word-count example appears below the metric equations, with no earlier visible worked calculation. | Closed after replacement review |
| V08-01 | 08 | 13,19,20,27 | SFT, reward/RL, DPO and LoRA equations precede visible numerical worked examples. | Closed after replacement review |
| V08-02 | 08 | mechanism,training,inference diagrams | Visible check questions differ from modern notes Ask/Answer; answers are present in Word explanation but not aligned in slide notes. | Closed after replacement review |
| V09-01 | 09 | rendered slides 23 and 24 | InfoNCE equation precedes worked numerical loss comparison; teaching standard requires concrete example first. | Closed after replacement review |
| V09-02 | 09 | rendered slide 29; Word page 13 | SigLIP sigmoid pair-loss mechanism lacks a concrete numeric binary-probability example before its equation. | Closed after replacement review |
| V10-01 | 10 | Word page 19 | Current measured-results table expands inside a paragraph as literal Markdown pipe rows. | Closed after replacement review |
| V10-02 | 10 | Word page 29 / Slide 10 detailed explanation | Archived Qwen results are labelled GPU in one explanation but source evidence and other passages specify CPU. | Closed after replacement review |
| V11-01 | 11 | Slide 23 / Word page 12 | Quantisation scale/round/reconstruct formula has no preceding concrete numeric rounding example. Slide memory formula also precedes numeric size comparison. | Closed after replacement review |
| V11-02 | 11 | Word page 22 / Slide 1 detailed explanation | Detailed title paragraph still calls the current latency/batch/monitor charts archived model/settings. | Closed after replacement review |
| V11-03 | 11 | Full notebook cell 46 output image | Batch throughput output axis says generated tokens per second, while the computation uses requested output-token budget. | Closed after replacement review |
| V12-01 | 12 | Slide 20 / Word page 11 | Retrieval formulas precede a concrete numeric RRF example; slide metric formulas also lack a prior numeric example. | Closed after replacement review |
| V12-02 | 12 | Word page 9 comparison table | First comparison table categorically states fine-tuning has no citations and universal Low/High costs, conflicting with later qualified comparison. | Closed after replacement review |
| V12-03 | 12 | Full notebook cell 29 output chart | Limited reference-pattern score is labelled general Answer accuracy / accuracy in the output chart. | Closed after replacement review |
| V09-03 | 09 | Word page 11 worked softmax example | Hypothetically changing the correct caption score to1 retains the original softmax denominator, making the described arithmetic inconsistent. | Closed after replacement review |

Useful review points: the worked-example panels in Weeks 6–9, 11 and 12; the numerical projector trace in Week 10; and the student notebook checks. The initial audit and earlier release snapshots remain historical records.

Evidence: [current correction manifest](AUDIT_FIXES_2026-10-12.json), [original audit](IMPLEMENTATION_AUDIT_2026-10-12.md), and [visual reports](../build/audit_fixes_20261012/). Checks, actual viewed-image hashes, current deliverable hashes and retained failed attempts are recorded there.

Links: 206/209 reachable, three publisher/robot challenges blocked, zero failed links. Blocked pages still need manual access.

Not verified: timed free Colab, live Gemini/OpenAI, browser widget interaction, actual GPU training, public deployment, optional stretches, three publisher-blocked reading pages, and beginner/classroom effectiveness. Local timings exclude installation and downloads. A clean local pinned installation and pip check were verified separately. No credentials were read, no hosted requests were sent and no commit or push was made.

Vendor follow-up: the official [Google ADK documentation](https://adk.dev/) was checked on 13 October 2026. Its name is Agent Development Kit; the existing source and rendered tables were already correct. A provisional visual misreading was excluded from the findings. Other checked vendor/version sources and dates are in [platform research](weeks/platform_2026.md).


## Student-path retry — 13 October 2026

The retry found additional student-facing problems that solution-only execution had missed. Nineteen TODO checks now stop unfinished code before it becomes a result, plot, training run or optional demonstration. The error identifies the TODO and tells the student to complete it and rerun. The actual model algorithms and MSc content are retained.

All 50 final command groups passed, including all twelve CPU smoke runs, coverage at 100%, nine paired built-student/built-solution regression cases, shared setup checks, callbacks and SDK contracts. All five changed labs received successful new full CPU runs. The remaining seven full records retain exact current executable-code parity. The longest current core remains Week 8 at 728.7 seconds (12.15 minutes). Timings use the tested local CPU environment and cached models/data; installation, downloads, widgets and hosted routes are excluded.

See [the retry report](VERIFICATION_RETRY_2026-10-13.md) for these additional findings and the current verification limits.
