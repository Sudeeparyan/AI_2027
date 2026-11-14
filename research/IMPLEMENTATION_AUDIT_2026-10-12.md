# Independent implementation audit — 12 October 2026

Follow-up completed 13 October: the findings below are historical. See [the corrections report](IMPLEMENTATION_FIXES_2026-10-12.md) and [current evidence](AUDIT_FIXES_2026-10-12.json) for the repaired artifacts and remaining external verification limits.

The prompt is substantially implemented, but it is **not fully compliant or ready to describe as perfect**. Fresh automated checks pass while independent source and teaching reviews identify **13 confirmed findings: 3 high, 8 medium and 2 low**. Findings remain open; this audit did not change teaching sources.

This is a new review of the completed release against `PROMPT_END_TO_END.md`, `AGENTS.md` and `research/TASKS.md`. It does not replace or retrospectively change earlier review records. The prompt's teacher override permits continuing beyond the pilot, so the absence of a Week 1 approval stop is not a defect.

## Confirmed findings

High means a specified optional provider or device path has an implementation defect. Medium means an explicit requirement or teaching/evidence safeguard is incomplete. Low means a narrower teaching or structural correction. No actual provider request or private-data transmission occurred during this audit.

| ID / week | File and location | Problem and evidence | Severity | Suggested correction | Status |
|---|---|---|---|---|---|
| A01 / 08 | [Lab source](../curriculum/labs/week_08_lab.py), lines 266–270, 576, 588 | Hosted comparison promises only invented messages, but `few()` also inserts `SHOTS` selected from Banking77 training messages. A synthetic sentinel fixture proves that the generated hosted prompt contains both a dataset example and the invented query. | High | Construct a separate, fully invented hosted few-shot example set. Check the entire outgoing prompt, including demonstrations. | Open |
| A02 / 08 | [Lab source](../curriculum/labs/week_08_lab.py), lines 409–431 | `SFTConfig` does not pass `use_cpu=(DEVICE == "cpu")`. Trainer's default `use_cpu=False` can select an available accelerator despite shared `DEVICE="cpu"`. Subsequent prediction prepares inputs on the shared device. The CPU runner hides CUDA, so its successful runs do not cover this case. Installed Transformers device-selection and Trainer model-placement code confirm the mismatch. | High | Bind Trainer to the selected device; verify a GPU-equipped kernel still trains and evaluates on CPU with `USE_GPU=False`. | Open |
| A03 / 07, 08 | [W07 lab](../curriculum/labs/week_07_lab.py), line 879; [W08 lab](../curriculum/labs/week_08_lab.py), line 588; [shared setup](../curriculum/labs/_setup.py), line 187 | Calls supply `max_output_tokens=8` and `12`; the OpenAI Responses helper forwards them unchanged. The current official Responses API documents a minimum of 16. These optional OpenAI requests violate the documented contract; errors are caught and appear as ordinary fallback. This was verified against documentation, not by a live request. | High | Use valid caps for OpenAI, or apply a provider-specific minimum in the helper and test the transmitted arguments. | Open |
| A04 / all | [Shared setup](../curriculum/labs/_setup.py), lines 63–64; [builder](../tools/build_week.py), line 466 | Rerunning setup resets the combined attempt counters to zero. The notebook tells students to rerun setup when changing providers. A fixture starting at 40 used attempts returns to zero, allowing more requests in the same kernel/lab session. | Medium | Preserve session counters across setup reruns and provider changes. Test exhaustion followed by rerunning setup. | Open |
| A05 / all | [Shared setup](../curriculum/labs/_setup.py), lines 22, 83 | After API opt-in, an absent environment key causes the default `.env` file to be read automatically. The prompt requires an explicitly selected local file. A mocked reader confirms the default target; no real `.env` was read. Provider opt-in itself is correctly enforced. | Medium | Default the local path to `None`/empty and read a file only after the student sets its path explicitly. | Open |
| A06 / 11 | [Lab source](../curriculum/labs/week_11_lab.py), lines 61–62, 84–88; [builder](../tools/build_week.py), line 456 | The install cell requests FastAPI/httpx/uvicorn, plus injected provider/widget packages. It does not supply required torch/transformers/pandas/matplotlib. Its comment incorrectly says shared setup installs model and plotting libraries; setup imports torch instead. A prepared instructor environment passes, but these cells do not bootstrap a fresh local kernel. | Medium | Include the core dependencies in installation, with researched pins where applicable, and test a clean local kernel. Alternatively provide explicit complete student prerequisites, correcting the comment. | Open |
| A07 / 08 | [Lab source](../curriculum/labs/week_08_lab.py), lines 210, 252, 378, 446 | The change task tells students to change learning rate and rerun Parts 4–5. Part 5 and `evaluate()` use `test_ds`, encouraging comparison of settings on the final test. Changing few-shot examples and reevaluating the same test creates the same problem. This contradicts the pack's freeze-before-test guidance. | Medium | Compare changes on validation/development examples, freeze settings, then run the final test once. | Open |
| A08 / 04, 05 | [W04 notes](../curriculum/notes/week_04.md), lines 153, 228; [W05 notes](../curriculum/notes/week_05.md), line 315; [ledger](lab_runs_2026-10.json) | Archived assets supply W04's 8.1-minute U-Net training and SD image timings, and W05's 8.7-minute training quote. No matching run-ledger records were found. The available W04 records describe different CPU runs; the dated W05 GPU run has different training time/loss. The values are stored measurements, but the required dated runtime provenance is incomplete. | Medium | Recover genuine archived run records with their real dates/settings, or remove unsupported runtime quotes. Do not invent dates or relabel different runs. | Open |
| A09 / 01–03 | [W01 slides](../curriculum/weeks/week_01.yaml), lines 571, 616, 655, 689; [W02 slides](../curriculum/weeks/week_02.yaml), lines 452, 483, 517, 550; [W03 slides](../curriculum/weeks/week_03.yaml), lines 304, 334, 805, 835 | Several resolved topic sequences still put equations before their worked examples or intuition: W01 MLE slide 25 before the likelihood table at 26; W02 ELBO at 18 before two-forces explanation at 19, and Gaussian KL at 20 before examples at 21; W03 optimal-D equation at 15 before its picture at 16. | Medium | Reorder these existing slides to follow the required concrete example/intuition/diagram before equation pattern, preserving approximately 46 slides. | Open |
| A10 / 08 and shared | [W08 notes](../curriculum/notes/week_08.md), lines 282–301; [canonical glossary](../curriculum/glossary.json); [beginner builder](../tools/beginner.py), lines 24–27 | Advanced glossary rows such as RLHF, reward model, PPO, DPO and QLoRA remain separately authored in Word notes and absent from the canonical glossary. The resolved Word specification retains that table alongside a canonical Quick glossary. Canonicalisation currently covers beginner prerequisite/glossary entries only. | Medium | Merge the advanced glossary terms into the canonical file and generate the weekly glossary from it. Review descriptions such as RLHF/PPO for appropriate scope. | Open |
| A11 / 10 | [Beginner source](../curriculum/beginner/week_10.json), training diagram around lines 192–236; [slides](../curriculum/weeks/week_10.yaml), lines 417, 430; [notes](../curriculum/notes/week_10.md), lines 60–76, 480 | Visual-bridge training has a verbal cup/caption example and a stage diagram, but no worked projector dimensions, numerical loss or update trace. The patch-count examples explain image preparation, not the training mechanism. | Medium | Add a small labelled feature → projector → answer-token-loss training example with shapes or numbers in slides and deeper notes. Clearly label illustrative values. | Open |
| A12 / 08 | [Lab source](../curriculum/labs/week_08_lab.py), Part 5 at line 446; Parts 2, 3, 6 | Part 5 lacks the specified core/change/stretch/common-errors structure; Parts 2, 3 and 6 lack their own optional stretch. A global extensions section does not provide the required per-part scaffolding. | Low | Add concise per-part activity and common-error boxes without expanding required CPU work. | Open |
| A13 / 02 | [Notes](../curriculum/notes/week_02.md), line 181 | Mismatched BCE/KL reductions are said to guarantee KL dominance and that every sample becomes a blurry average digit. The relative weighting changes by 784 under the stated reductions, but the result also depends on magnitudes, β and training. | Low | Say the relative KL weight increases by 784 and can encourage collapse or poor reconstructions; keep the reduction advice. | Open |

A03's API constraint was checked on 12 October 2026 against the [official OpenAI Responses create reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create). The documented minimum is 16; this audit did not spend API credit. The default model identifiers also match their current official pages: [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) and [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini). This limited vendor check does not certify every historical model, price or version in the pack.

## What passed in this session

**29 commands passed**, including both builds, 25 structural/text/regression checks, link checking and the all-week smoke command. All **12 individual CPU smoke runs passed**. Exact commands, exit codes, times and logs are in [checks.json](../build/implementation_audit_20261012/checks.json). Smoke settings are deliberately small; these are not replacement full-core runtime measurements.

The checks include speaker notes, setup/helpers/runtime/evidence, output writers, beginner support, technical figures/diagrams, teaching notes, visual support, lecture flows, slide visuals, all five text/coverage audits, figure layout/text size, protected artifacts and executable full-run code parity. Coverage is **214/214 (100%)**. Links are **205/209 reachable, four blocked, zero confirmed failures**; the blocked pages remain unverified despite the command's successful exit.

Independent inventory of the resolved specifications found:

| Requirement | Result |
|---|---|
| Slide count | 46 per deck, 552 total |
| Modern speaker-note labels and length | All 552 notes have six sections; observed lengths 112–179 words; strict validator passes |
| Technical diagrams / infographic / video | Five technical diagrams, one infographic and at least one video per week |
| Notebook delivery | Student and solutions notebooks; output checks pass |
| Markdown before code / expected output | Present in all 24 delivered notebooks |
| Instructor/student answer separation | Structural checks pass; no instructor answer explanations/markers found in student Markdown |
| Widgets | Each week contains a `skip-test` widget cell; browser behaviour not verified |
| Real Week 8 fine-tuning | Actual LoRA/SFT job retained |
| CPU core evidence | Latest retained records are within 900 seconds; full executable code parity passes |
| Result filling | Independent calculation reproduces all 12 current `fill.json` values |
| Protected sources | Protected-artifact check passes; no changes to context/sources/legacy |

An exploratory count of editable slide text initially counted numeric step badges as words. Excluding those generated number labels, the inspected body text stays within 40 words. This was **not** raised as a finding. Original 40-slide source numbering is remapped by the builder; it was also **not** raised as a defect.

## Evidence and visual-review scope

Before rebuilding, the independent evidence review matched all **639 source hashes, 259 deliverable hashes and 1,187 reviewed-image hashes** in the prior release manifest, plus all 12 retained executed-full notebook hashes. The latest full CPU records agree with their recorded full-run outputs. The longest retained core is Week 8 at **797.3 seconds**. These are inspected prior measurements, not new full runs in this audit.

The builds regenerated tracked Office binaries. A fresh comparison normalising relationship identifiers and ignoring core document timestamps found **all 24 weekly Office files unchanged in content** relative to the retained render inputs. See [office_content_parity.json](../build/implementation_audit_20261012/office_content_parity.json). Teaching sources remain unchanged. Existing layout reports are empty for all 12 decks.

Fresh visual inspection was sampled, not a new exhaustive render:

- Slide sheets: W01 sheet 04, W02 sheet 04, W03 sheet 03, W04 sheet 03, W05 sheet 03, W08 sheet 05, W09 sheet 04, W10 sheet 03, W12 sheet 03 — **54 slides**.
- Word sheets: W08 and W11 sheet 02 — **8 pages**.
- Independent diagram/notebook image review: W08 and W12 sheets 01–02 — **16 diagram PNGs and 17 notebook images**.

No new clipping, broken-image or crossing-arrow defect was seen in these samples. The source-order issue A09 and missing worked example A11 remain substantive even where the pages look clean. The earlier complete visual review is retained historical evidence; this audit did not newly view every slide, Word page or course page.

Synthetic fixtures reproduced A01, A04 and A05 without real datasets, secrets or network calls. Their outcomes are in [fixture_results.json](../build/implementation_audit_20261012/fixture_results.json), with the [reproduction script](../build/implementation_audit_20261012/reproduce_findings.py). No credentials were read or printed.

## What remains unverified

| Area | Practical limit |
|---|---|
| Clean pinned installation | Current tests skip installation. Retained W08 full evidence uses Transformers 5.17.0, TRL 1.14.0, PEFT 0.21.0 and torch 2.14.0+cpu; delivered pins specify 5.18.0, 1.15.0, 0.21.1 and 2.11.0. Successful runs do not validate that different pinned environment. |
| Free Colab | No timed Colab run or cold installation/download measurement was performed or found in the release's claimed verification. Cached laptop times do not establish Colab timing. |
| Hosted providers | No live Gemini/OpenAI requests. Stub tests validate helper paths but missed the documented OpenAI token constraint and real prompt composition. |
| Widgets and student workflow | No browser slider interaction, full student-TODO completion trial or classroom comprehension study. Presence of cells does not establish usability. |
| GPU-equipped CPU default | No fresh GPU-equipped notebook run; A02 is confirmed from code and installed library device-selection behaviour. |
| Optional stretches/deployment | No new QLoRA/media/GPU stretch, public Gradio or Docker deployment validation. |
| Full runtimes and exhaustive visuals | Full CPU jobs and Office rendering were not repeated; retained evidence/code/content parity was checked instead. |
| External reading | Four publisher URLs block automated checks; no manual verification of those pages. |

Fix A01–A07 first, then complete evidence and teaching corrections A08–A13. Rebuild changed sources, test the affected full CPU paths, and review changed rendered output. A clean pinned local run and representative timed Colab/student/widget runs are needed before a stronger portability or beginner-usability claim. Local completion remains free; hosted access stays optional.

Audit-only changes: this report, task-board handoff and diagnostic scripts/logs under `build/implementation_audit_20261012/`. Generated Office files were rebuilt through the normal pipeline; they were not hand-edited. No commit or push.
