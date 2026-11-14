# How to review and improve a week

This is the working routine used for the October 2026 revision. Follow it when you (a person, Claude or
Codex) improve the pack again, so each pass finds the same kinds of mistakes and leaves the same
evidence behind. The rules are in `AGENTS.md` and the current tasks in `research/TASKS.md`
(`improvements.md` is superseded); the open and closed findings are in
`research/AUDIT_2026-10.md`; what changed is in `research/REVISION_2026-10.md`.

## Rules that never change

- Edit sources only: `curriculum/` and `tools/`. Never edit `deliverables/` by hand; rebuild it.
- Do not change `context/`, `sources/` or `legacy/`. Only Section 7.3 of the descriptor may change, and
  the week topics and order come from `curriculum/section_7_3.yaml`.
- Every lab number shown on a slide or in the notes comes from a measured run:
  lab export (`GENAI_RESULTS_PATH`) or asset script → `curriculum/assets/week_XX/*.json` →
  `tools/fill_results.py` → `fill.json` → a `{{PLACEHOLDER}}` in the YAML or notes. Never type a result by hand.
- A runtime, download size or hardware claim states what was measured, on which machine and on which
  date. If something was not timed, say so.
- `tools/audit_coverage.py` stays at 100%. About 46 slides per deck.
- Do not commit or push unless the teacher asks.

## The loop for one week

1. **Read the sources** for week N: `curriculum/weeks/week_NN.yaml` (slides and speaker notes),
   `curriculum/notes/week_NN.md` (Word notes), `curriculum/beginner/week_NN.json` (beginner slides,
   diagrams and walkthrough), `curriculum/labs/week_NN_lab.py` and `curriculum/figures/week_NN.py`.
   A quick way to read every slide with its notes:

   ```powershell
   .venv/Scripts/python.exe -c "import yaml,json;[print(i,json.dumps({k:v for k,v in s.items() if k!='notes'},ensure_ascii=False)[:700],'\nNOTES:',s.get('notes')) for i,s in enumerate(yaml.safe_load(open('curriculum/weeks/week_07.yaml',encoding='utf-8'))['slides'],1)]"
   ```

2. **Run the text audits** and fix every finding in the source:

   ```powershell
   .venv/Scripts/python.exe tools/audit_prose.py        # minus signs, ranges, lost glyphs, sentence starts, repeated
                                                        # notes, arrows, bare <tags>, quiz answer letters
   .venv/Scripts/python.exe tools/audit_inline_math.py  # LaTeX left over, raw _ or ^, maths spelled as words
   .venv/Scripts/python.exe tools/audit_slide_text.py
   .venv/Scripts/python.exe tools/audit_flat_bullets.py # slide steps or lab boxes printed as loose Word bullets
   ```

3. **Check every claim against evidence.** Numbers, model names, item counts, download sizes and timings
   must match the stored results (`curriculum/assets/week_NN/`), the lab code and the executed notebook
   (`build/week_NN/executed_solutions_full.ipynb`). Re-run the lab when the evidence is missing or old:

   ```powershell
   # quick structural check (tiny settings)
   .venv-labs/Scripts/python.exe tools/test_labs.py --weeks N
   # measured required CPU core, with locally cached models and data
   $env:HF_HUB_OFFLINE="1"; $env:GENAI_RESULTS_PATH="<scratch>/results_cpu_core.json"
   .venv-labs/Scripts/python.exe tools/test_labs.py --weeks N --full --device cpu
   # optional GPU/stretch measurements need their own settings, export and run record
   ```

   `test_labs.py` executes the built notebook in `deliverables/`, so run `build_week.py --weeks N` after
   any lab edit and before the run. Every current required core uses CPU by default, including bounded
   Week 8 LoRA and the smaller Week 10 VLM. Keep optional larger-model/GPU experiments separate.
   Compare the scratch export with its settings before promoting `results_cpu_core.json`, retain the
   original archived assets, and record successful full cores in `iteration_3_full_runs`. Then run
   `.venv/Scripts/python.exe tools/fill_results.py --weeks N`. Harvest figures only from the matching
   successful full notebook, never from smoke output.

4. **Rebuild and render:**

   ```powershell
   .venv/Scripts/python.exe tools/build_week.py --weeks N
   .venv/Scripts/python.exe tools/render.py --weeks N
   .venv/Scripts/python.exe tools/notes_sheets.py --weeks N
   .venv/Scripts/python.exe tools/lint_figures.py --weeks N
   .venv/Scripts/python.exe tools/audit_figure_text.py --weeks N
   ```

   `build/render/week_NN/slides.layout.json` must be `[]` (no text box overflows).

5. **Look at every slide and every Word page.** Slide images are `build/render/week_NN/slides-NN.png`;
   Word pages are tiled four to a sheet in `build/render/week_NN/notes_sheets/`. For small print, render
   two pages at higher resolution with `pdftoppm -r 170 -f A -l B -png build/render/week_NN/notes.pdf page`.
   `lint_figures.py` does not see the editable (native) diagrams, so check those by eye.

6. **Record** each defect and its fix as a row in the week's section of `research/AUDIT_2026-10.md`,
   rebuild, and look at the changed pages again.

## What to look for

These are the mistakes found so far. Most now have an automatic check; the rest need reading.

| Look for | Example | Checked by |
|---|---|---|
| Speaker-note boilerplate pasted on many slides | the same Ask/Example/Watch out sentence on 8 slides | `audit_prose.py` (REPEAT) |
| Missing or reordered modern note sections | repeated Explain, no Watch out, or an out-of-order Answer | strict `tools/speaker_notes.py`; `test_speaker_notes.py` |
| Required CPU core confused with an archived larger run | earlier Week 8 training time presented as today's core | latest successful `iteration_3_full_runs`; separate archived table |
| Provider credentials silently enable requests | a stored key activates a comparison | shared default-False switches; stubbed setup tests; `ask_model()` local fallback |
| An answer labelled as a mistake | "Ask: …? Common mistake: No." | `audit_prose.py` (LABEL) |
| Every Ask has an Answer that fits the slide | an Ask about RoPE on a slide about decoding | `audit_prose.py` (ANSWER) checks presence; reading checks meaning |
| Final test remains separate from choosing settings | picking the best prompt on the final test | reading; choose on development examples and freeze before testing |
| Document-level retrieval hits really include the needed fact | the right source document, but the wrong passage | inspect the actual chunks passed to the generator |
| Counts in the notes match the slide | "Recap the eight points" on a six-point slide | reading |
| Notes describe what the figure really shows | notes describe an invalid-answer chart the figure leaves out | reading |
| Preview belongs to the distributed Office file | the course guide was rebuilt with a new runtime, but its PNGs still show the old table | `tools/verify_render_inputs.py` after weekly and course rendering; then actual image review |
| Negative numbers and ranges | `-0.5` → −0.5, `5-8` → 5–8 | `audit_prose.py` (MINUS, RANGE) |
| Maths typed as words or ASCII | `epsilon_theta`, `abar = 0.64`, `x0` | `audit_inline_math.py` |
| Sentences that start with a lower-case code name | ". avg_nll measures" → ". The avg_nll function measures" | `audit_prose.py` (LOWER) |
| Unmeasured runtimes and sizes | "Downloads ≈ 3 GB" (it was 4.5 GB) | reading + lab run |
| Claims that only hold on one path | Flan-T5-small on the CPU path answers "ireland" | executed notebook |
| Picture text smaller than 14 pt when projected | chart tick labels | `audit_figure_text.py` |
| Too much projected text | repeated outcome descriptions and lab labels push a slide above 40 body words | `audit_body_words.py` for ordinary text shapes; inspect diagrams and image text separately |
| Legends over data or axis labels | figure-level legends | `lint_figures.py` + eye |
| Evaluation labelled as training | “Loss / update” beneath frozen-model scoring or validation | use `metric` (“Measure / check”) for checking; keep `loss` for a training objective/update; read the flow and run diagram tests |
| Windows locks an output during rebuilding | WinError 1224 or an UNKNOWN open/write error | atomic output writers preserve the old file; bounded replacement retries, writer tests and a successful rebuild |
| Stray paragraphs left from other weeks | an SD-Turbo licence line at the end of Week 6 | reading |
| One answer letter for most quizzes | 21 of 24 slide quizzes were B; 33 of 45 practice answers were (b) | `audit_prose.py` (QUIZ) |
| The correct option is the longest | a full sentence against three two-word distractors | `audit_prose.py` (QUIZ); when fixing, keep each distractor's wrong idea so the explanation still fits |
| A quiz explanation naming the wrong letter | "Answer: B" after the options were reordered | `audit_prose.py` (QUIZ) for slides; reading for the explanation's other letters |
| `<tag>` text that Word and Jupyter drop | "Use tags such as <document>" printed as "Use tags such as" | `audit_prose.py` (TAG) |
| Arrows typed as `->` | "retrieve -> answer" → "retrieve → answer" | `audit_prose.py` (ARROW) |
| Worked arithmetic that uses rounded inputs | F1 from 0.63 and 0.83 gives 0.72, not the stated 0.71 | reading (redo the sum) |
| Revision history written into teaching text | "the lecture now uses …", "historical exports cannot be repaired" | reading |
| Slide steps flattened into loose Word bullets | "- Input" above "- Image, document, audio or video"; an unlabelled runtime line | `audit_flat_bullets.py` |
| Greek letters spelled out next to a superscript, or hyphens as minus signs | "mu² + sigma² - log sigma²" | `audit_inline_math.py`; `audit_prose.py` (HYPHEN) |
| Product names that have changed | "Vertex AI" (now Gemini Enterprise Agent Platform), "Azure AI Foundry" (now Microsoft Foundry) | reading + the vendor's own page; keep the former name in brackets |
| A rounded ratio that the printed numbers do not give | "13 → 86 tokens/s (6.5×)" | reading; print the values the ratio came from |
| Runtimes that do not match a recorded run | "≈ 20 min on a Colab T4" with no T4 run | `research/lab_runs_2026-10.json` |
| Markdown inside a fill value | `**Measured (…):**` shows literal asterisks in PowerPoint notes | no Markdown in `fill_results.py` strings; search built PPTX notes for `**` |
| Internal build words in student text | "Read the result placeholder…"; "the null connector" for a missing diagram arrow | search spec and notes for `placeholder` and `null` near arrow/link/connector; reading |
| A recap example that needs the earlier slide | "0.8 − 0.3 = 0.5" or "top-p .8 keeps three" on the final recap | reading; give the inputs in the example itself |
| A rounded interval bound | Wilson 32/40 quoted as 65–89%; the exact upper bound 89.5% rounds to 90% | reading; recompute the bound before rounding |
| A notes example that skips the figure's own numbers | figure shows width 12 with 3 heads; notes give only the lab's 32/4 | reading; walk the figure's numbers first |
| A line break inside one JSON paragraph | `\n ` in a diagram explanation splits a Word paragraph | reading the Word render; search beginner JSON strings for `\n` |
| A line that ends with a bare option label | "… (c)" with "[4, 4]" on the next line | `tools/notes.js` joins the label to its option; `pdftotext -enc UTF-8 -layout build/render/week_NN/notes.pdf - \| grep -E "\((a\|b\|c\|d)\)\s*$"` finds any left |

## Helpers

- `tools/yaml_edit.py` changes chosen keys of chosen slides and leaves the rest of the YAML file
  byte for byte, then checks that nothing else moved:

  ```python
  import sys; sys.path.insert(0, "tools")
  from yaml_edit import edit_slides
  edit_slides(7, {16: {"title": "One idea on this slide"}})
  ```

- Write edit scripts to a file and run them with `.venv/Scripts/python.exe`. Inline shell heredocs can
  mangle backslashes and quotes (`\eta` became `eta`, `−` escapes broke).
- Set `PYTHONIOENCODING=utf-8` when printing Greek letters or dashes on Windows.
- Labs and model scripts run offline from the local Hugging Face cache with `HF_HUB_OFFLINE=1`.
- `tools/notes_sheets.py --weeks 1-12 --slide 44` tiles one slide from every week into
  `build/render/slide_44_weeks.png`, which is the quickest way to compare a slide type (lab, quiz, recap)
  across the course after a shared change.
- `research/lab_runs_2026-10.json` records every full lab run that a runtime on a slide, in the notes or in
  the instructor guide's lab table relies on (minutes, hardware, environment, notebook hash). Add a record
  when a new full run replaces a time; never write a time that has no record.
- `tools/compare_lab_code.py -v` compares the code cells of each week's last full run with the built
  notebook in order, retaining string contents while ignoring comments and formatting. Missing or failed
  full-run evidence fails the check. If it reports a code change, re-run the lab before trusting its stored
  results or time. The 10 October follow-up re-ran Weeks 1, 6 and 11 after their explicit hosted-call gates changed.
- `tools/harvest_figures.py --week N --dry-run` checks weeks 9–12 against the successful mode-specific
  full notebook before any assets change. A later smoke run cannot become the figure source. All figures
  must be present and valid PNGs before writing. New run records include the executed notebook's hash;
  older records are not retroactively given one. Run `tools/test_lab_evidence.py` after changing these guards.
- Optional provider code uses shared `ask_model()` with `USE_GEMINI=False` and `USE_OPENAI=False`.
  Enable exactly one explicitly; a stored credential alone never starts a request. Keep local fallback
  answers, fictional prompts and `skip-test` widgets. Test default, retry and smoke paths with stubbed
  clients; a CPU run does not verify the hosted endpoint. OpenAI API billing is separate from ChatGPT.
- Modern six-label speaker notes are strict by default: 110–180 total words including labels, and three
  to six Explain sentences of at most twenty words. Keep quiz `Answer: X` aligned with its option.
- The course guide derives current CPU times from the latest successful full/core CPU ledger records.
  Preserve timestamps, settings and both notebook hashes; older full experiments remain archived.
- `check_links.py` retries temporary server/transport failures at most once, preserving the attempt history.
  It still fails on missing pages and reports blocked pages as unverified. Run `test_link_checks.py` after
  editing this logic; an initial failed attempt remains part of the follow-up evidence.
- Notebooks are byte-identical when their sources have not changed, so a hash shows whether a smoke or full
  run still applies. PowerPoint and Word files are not: each build writes a new timestamp, and Word
  hyperlinks get random relationship ids. Compare their contents, not their hashes.
- Swapping quiz options: move the correct option and one distractor only, then rewrite every letter the
  explanation names by hand. A blind letter replacement is unsafe ("A bigger model" uses A as a word).

## Archived status (10 October 2026)

Every week completed the original review. The independent follow-up corrected additional source,
instructor-answer and evidence defects. The separate evidence tables in `research/REVISION_2026-10.md`
distinguish the original review from checks run again during the follow-up. Run and current image hashes
are retained in `research/FOLLOWUP_2026-10-10.json`.

| Week | Text audits | Evidence checked | Slides viewed | Word pages viewed | Audit section |
|---|---|---|---|---|---|
| 1–12 | pass (0 problems) | all current smoke hashes checked; full 1/6/11 repeated; other full-run code unchanged | earlier full review plus changed-image review | earlier full review plus changed-image review | yes, no open rows |

This table describes the earlier review; it does not certify the upgraded sources.

## Historical Iteration 3 snapshot (12 October 2026)

The prior release record reported 26 release gates, 12 final smoke runs and successful measured CPU cores, with strict modern notes, glossary and primary/video references. These claims describe the earlier snapshot in `ITERATION3_2026-10-12.json`; they do not certify corrections made after the independent implementation audit.

That historical record lists 552 slides, 402 weekly Word pages, 103 delivered diagram PNGs, 130 notebook images and 50 course pages, with empty PowerPoint layout reports and no open material findings at its closure. Changed images were viewed; unchanged images retained exact prior-view hashes. Preserve these counts as history rather than reusing them for the corrected build. The view reports are linked in `ITERATION3_2026-10-12.json`; findings remain under Iteration 3 in the audit.

The historical CPU measurements excluded installation/downloads, widgets, live providers and optional GPU/media stretches. The clean local installation has since been verified separately, as described below.

## Audit correction status (completed 13 October 2026)

The current correction record is `research/AUDIT_FIXES_2026-10-12.json`, separate from the historical Iteration 3 snapshot. A clean local Python 3.12 installation and `pip check` passed for the 14 exact direct pins documented in `research/weeks/platform_2026.md`. Use the correction record for the actual clean pinned CPU runs, CUDA-available/default-CPU regression, final gates and full rendered-review scope. Keep historical counts separate from that record; later changes require new checks and current image review.

Corrections preserve the shared 40-attempt provider cap including retries, explicit `.env` selection after opt-in, and local fallbacks for unavailable APIs or widgets. Week 8 develops on validation and freezes the final test; score reuse checks the actual adapter weights, training data, settings and seed. Every new result still flows through the measured asset and fill pipeline. Serialise full runs and Office renders, then record current hashes, actual views and any findings before closure.

Unverified: timed Colab runs, live Gemini/OpenAI calls, browser widget rendering, public deployment, optional stretches, publisher-blocked links and student usability testing. Local installation, regression checks and CPU execution do not establish those conditions.


34 release gates and the final affected-artifact checks passed; all twelve smoke runs and clean pinned full CPU cores passed. The longest core was Week 8: 728.7 seconds (12.15 minutes), within the 15-minute budget. Visual review covers 552 slides, 412 weekly Word pages, 103 diagram PNGs, 130 notebook images and 54 course pages. All twelve slide layout reports are empty; zero open findings. Current Office render inputs, inspected image hashes and full-run executable code match the distributed artifacts.

The correction findings are closed in IMPLEMENTATION_FIXES_2026-10-12.md; AUDIT_FIXES_2026-10-12.json binds the current checks and reviewed artifacts. Not verified: timed free Colab, live Gemini/OpenAI, browser widget interaction, actual GPU training, public deployment, optional stretches, three publisher-blocked reading pages, and beginner/classroom effectiveness. Local timings exclude installation and downloads. A clean local pinned installation and pip check were verified separately. No credentials were read, no hosted requests were sent and no commit or push was made.


## Iteration 3 — student-path retry, 13 October 2026

The retry found additional student-facing problems that solution-only execution had missed. Nineteen TODO checks now stop unfinished code before it becomes a result, plot, training run or optional demonstration. The error identifies the TODO and tells the student to complete it and rerun. The actual model algorithms and MSc content are retained.

All 50 final command groups passed, including all twelve CPU smoke runs, coverage at 100%, nine paired built-student/built-solution regression cases, shared setup checks, callbacks and SDK contracts. All five changed labs received successful new full CPU runs. The remaining seven full records retain exact current executable-code parity. The longest current core remains Week 8 at 728.7 seconds (12.15 minutes). Timings use the tested local CPU environment and cached models/data; installation, downloads, widgets and hosted routes are excluded.

Current replacement review: zero open findings. Source changes, archived/current metric wording and the 24-package version verification are detailed in VERIFICATION_RETRY_2026-10-13.md. Desktop/Colab/live API/classroom limits remain explicit; no new depth or course-order changes.


### VS Code widget pilot follow-up — 13 October 2026

The current Week 1 temperature widget was rerun in VS Code with Python 3.12.10 and ipywidgets 7.7.1. Its visible control and chart updated at T=1.0, 1.7 and 0.4. Four exact distributed code cells used a five-name invented fixture. No permission setting was changed by the agent. This closes the local Week 1 interaction check; other frontend checks and timed Colab remain unverified. Evidence: build/retry_20261013/widget_ui_verified.json.

## Pre-submission end-to-end check (Q6), 16 October 2026

Q5 (14–15 Oct) stopped before its final checks; Q6 completed them and reviewed all 12 weeks again. 18 findings fixed in sources (F1–F18 in `research/E2E_REVIEW_2026-10-16.md`), with no change to executable lab code or measured results. Final: 12/12 builds, 26/26 gates (links 204/209, 5 publisher-blocked), 12/12 CPU smokes, code parity 12/12, empty layout reports, `verify_render_inputs.py` 29/29, and every changed slide and Word page viewed. Unverified: timed Colab, live APIs, Colab widgets, blocked links, classroom use.
