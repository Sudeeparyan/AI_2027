# w08_log

## Iteration 3 (Claude, 11 Oct 2026)

### L · Lecture
- 40 core notes rewritten to the six-label template; slide bodies and quiz options tightened.
- Figures redrawn with roles and labelled arrows: decision process, RLHF pipeline, LoRA (arrows now carry x [k], W x [d], A x [r], B(Ax) [d]).
- Results slides relabelled as the stored full run (1,216 examples, 76 min); CPU-core values added through new `W08_CORE_*` fill keys (`fill_results.week8_core`).
- Beginner support rewritten: prerequisites, five diagrams with real data flow (split → format → LoRA → train → compare; parallel LoRA path; one parser for three models), recap.
- Word notes: CPU-core lab guide, worked examples (answer-token loss, reward-model loss, DPO loss, LoRA count), FAQ, five self-checks.

### V · Visual QA
| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 08 | weeks/week_08.yaml, beginner/week_08.json, figures/week_08.py | 10 views (7, 15, 19, 20, 23, 25, 26, 34, 43, 44) | Visible text over 40 words | High | Overview merged Test into Compare (5 blocks); shorter labels, quiz options, read-as lines; KL caveat kept in notes; LoRA caption removed (count is on the next slide) | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | labs/week_08_lab.py | Part 1 box | Question mark before a blockquote line (audit_prose LOST) | Low | Rephrased as an instruction | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | notes/week_08.md | Reward-model example | `y_w`, `y_l` in plain text | Low | Inline maths | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | weeks/week_08.yaml | 20 notes | β named differently from slide 19 | Low | "drift-penalty strength from RLHF" on both | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | weeks/week_08.yaml | 28 | Second bullet repeated the rank of bullet 1 | Low | Measured CPU-core count (rank 8, 1,081,344) on slide and in notes | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | tools/fill_results.py | 37 | "base, zero-shot 52.5%, base, few-shot …" hard to parse | Low | Semicolons between models, as in the core string | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | all | 46 slides | Full render review (contact sheets) | — | Overview, LoRA, decision, RLHF, equations, quiz, lab slide read cleanly | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

### Codex continuation (11 Oct)
- Preserved all completed W08 sources and real CPU fine-tuning evidence; `compare_lab_code --weeks 8-9` confirmed unchanged W08 executable code.
- Viewed all 46 slides, 28 Word pages, eight delivered diagrams and eight notebook images; final changed slides were rendered and viewed again. Current built attachments are used for review; measured full outputs remain separate.
- Added missing definitions for every LoRA equation symbol and the inventor video to the deck resources (already sourced in W08 research). Body count remains ≤40.
| 08 | weeks/week_08.yaml | 27 | Equation key omitted h, x, W, A, B, d, k | High | Grouped symbol definitions and shorter numeric example | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 08 | weeks/week_08.yaml | 45 | Deck reading list lacked a video | Medium | Added the inventor video; paper remains in Word readings | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

- Final continuation review: 46 slides, 28 Word pages, eight diagram images and eight notebook images viewed; `slides.layout.json` is `[]`. Final slides 27/28/37/45 clear. Zero open findings. Unchanged executable lab code confirmed; prior real CPU full run retained. No Colab, clean-install, live provider or browser-widget claim.

## Final coordinated closure — 12 October 2026
- Final current coverage: 46 slides, 29 Word pages, 8 delivered diagram PNGs and 8 notebook images; actual views plus byte-identical previously viewed images are documented in the final review manifests. Layout report [], zero open findings.
- Latest successful cached local CPU full core: 797.3 s (13.3 min), recorded in `research/lab_runs_2026-10.json`; final smoke PASS (158.1 s). Current executed core code matches the successful full-run evidence.
- All 26 current release gates passed, including coverage 100%, strict notes, body ≤ 40, fonts ≥ 14, semantic helpers and shared diagram/attachment parity. See `research/ITERATION3_2026-10-12.json`.
- No timed Colab, clean-install/download, live-provider, browser-widget or optional GPU/stretch claim. Earlier pending handoffs above describe intermediate states.

## Additional independent teaching review — 12 October 2026
- Additional semantic review: rising validation loss suggests overfitting; it does not prove literal memorisation. Updated beginner training explanation and Answer; strict five-diagram notes PASS.
- Handoff: root-authorised narrow source changes complete; build/render checks remain coordinated by root. No new lab logic, runtime or provider claim.


## Runtime provenance repair — 12 October 2026
- Removed the undated archived 76-minute claim from the stored LoRA loss-curve figure title. Losses, settings, sample count and current dated CPU fine-tuning evidence remain. Root owns timing filler removal and the corrected float32 weight-size description.
- Source edits only; no build, render or notebook execution during active coordinated CPU runs. Coordinator must rebuild and inspect the changed charts and Word sections.

## Successful clean pinned CPU retry — 12 October 2026
- Actual full CPU core retry started 2026-10-12T22:26:04.169647+00:00, completed in 728.7s (12.15min), within the 900s budget. The coordinator/lab runner executed it; this evidence task checked the successful record, zero notebook errors and matching executed-notebook SHA256.
- Promoted the measured staged export to curriculum/assets/week_08/results_cpu_core.json with exact run start, runtime versions and export/notebook hashes; appended the genuine retry to research/lab_runs_2026-10.json. Original failed batch/attempt evidence remains unchanged.
- Retained archived stretch/GPU figures and non-timing metrics; no harvest mutation, build, fill, render, Colab timing, live provider or browser-widget verification performed here. Final fill/build/render review remains with Q1-V.


### Audit correction closeout — 13 October 2026

- Real portable CPU LoRA training; invented hosted examples, guarded final test, activity/check scaffolding and examples before formulas.
- Final current artifacts rebuilt, checked and visually reviewed; no open findings. Evidence: ../AUDIT_FIXES_2026-10-12.json.
- CPU/Office verification is local; Colab, live API and classroom limits remain documented.
