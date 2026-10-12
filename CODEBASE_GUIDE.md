# Teaching pack codebase guide

The active project generates a twelve-week Generative AI course through a source-to-deliverable build pipeline. Edit curriculum sources, then regenerate the files in `deliverables/`. The `legacy/` folder contains an earlier independent generator and is retained for reference.

## End to end build flow

```mermaid
flowchart TD
  A["Section 7.3 YAML: course topics and tutorials"] --> B["build_week.py: load and validate a week"]
  C["Weekly YAML: core slides and speaker notes"] --> B
  D["Weekly Markdown: teaching notes"] --> B
  E["Beginner JSON: diagrams and explanations"] --> F["beginner.py: insert support slides and worked examples"]
  B --> F
  F --> G["spec.json: one resolved weekly specification"]
  H["Weekly figures and measured assets"] --> G
  E --> I["technical_diagrams.js: SVG and PNG"]
  I --> G
  G --> J["slides.js: editable PowerPoint"]
  G --> K["notes.js: Word teaching notes"]
  L["Lab Python with percent cells and solution markers"] --> M["Student and instructor notebook builders"]
  I --> M
  M --> N["Notebooks with embedded PNGs and commented Python exports"]
  J --> O["deliverables: week folder"]
  K --> O
  N --> O
```

## Source map

| Path | Responsibility | Edit when |
|---|---|---|
| `curriculum/section_7_3.yaml` | Course scope, week order, lecture topics and tutorials | The authorised descriptor content changes |
| `curriculum/weeks/week_XX.yaml` | Forty core slides, speaker notes and coverage terms | Lecture content changes |
| `curriculum/beginner/week_XX.json` | Prerequisites, five technical diagrams, recap cards, example traces, glossary and code symbols | A concept needs clearer explanation |
| `curriculum/glossary.json` | Canonical terms reused across all weeks and the course glossary | A definition needs correction |
| `curriculum/notes/week_XX.md` | Full teaching notes, exercises, answers and references | Detailed lecture or lab guidance changes |
| `curriculum/labs/week_XX_lab.py` | Student exercises and instructor solutions in one percent-format source | Code or code-reading guidance changes |
| `curriculum/figures/week_XX.py` | Existing conceptual figures and result visualisation | An existing figure changes |
| `curriculum/assets/` | Measured lab outputs and asset-generation scripts | New lab measurements are produced |
| `curriculum/course/` | Course-level guide and rationale sources | Course guidance changes |
| `tools/build_week.py` | Resolves content, validates it and coordinates all weekly outputs | The output pipeline changes |
| `tools/beginner.py` | Validates beginner content, inserts vocabulary and diagram slides, adds worked examples and remaps slide references | The shared teaching structure changes |
| `tools/technical_diagrams.js` | Renders the same diagram labels as native PowerPoint objects, SVG and PNG | Diagram layout changes |
| `tools/teaching_support.js`, `tools/arial_widths.json` | Shared course-map and recap scenes, with measured Arial wrapping | Course support layout changes |
| `tools/slides.js`, `tools/notes.js` | PowerPoint and Word layout engines | Office layout changes |
| `tools/build_course.py`, `tools/build_descriptor.py` | Course guide, rationale and descriptor builders | Course-level output generation changes |
| `tools/render.py`, `tools/render_office.ps1` | Microsoft Office PDF and PNG previews | Visual review is needed |
| `tools/verify_beginner.py` | Portable image, code, notebook and slide structure checks | Verify a rebuilt release |
| `tools/test_labs.py` | Executes solution notebooks with small smoke settings or full settings | Executable lab logic changes |
| `build/` | Rebuildable caches, specifications and review previews | Inspect a failed build |
| `deliverables/` | Files distributed to instructors and students | Use the generated pack |

## Beginner diagram contract

Block roles distinguish training losses/updates (`loss`) from evaluation, validation and checking (`metric`). Both use the orange objective palette, with explicit legends “Loss / update” and “Measure / check”. Measuring a frozen model does not change its weights. The diagram tests guard this distinction in frozen-model lab flows.

Each week defines `overview`, `mechanism`, `training`, `inference` and `lab` diagrams with four to seven numbered blocks. The `training` and `inference` IDs also cover preparing and using systems such as prompt tests, deployment and RAG. Each block names an operation, explains it in plain language and can name a real lab symbol. `arrows` contains one output label for each connection. An optional feedback route names the operation to repeat and `feedback_from` and `feedback_to` give its zero-based source and target. `insert_before_title` anchors each detailed diagram beside an exact existing slide title. IDs must be unique. The builder computes the final slide count from the authored diagrams.

All seven PNGs (five technical diagrams, course map and recap) attach directly to notebook Markdown cells. The Word notes use the same PNGs. The PowerPoint uses editable blocks, labels and connectors. Separate SVG and PNG files appear in each week's `Diagrams/` folder. SVG and PowerPoint share one geometry function, so every arrow starts and ends on a block boundary in both formats. A correction to an authored diagram therefore updates every format on the next build. Shared scenes use measured Arial word wrapping and explicit single-line boxes. Rounded-corner adjustments apply only to rounded rectangles; applying them to ellipses makes Office reject a deck.

The recap replaces the original summary and moves to the last slide, keeping 46 slides. The original detailed points remain in Word. Lecture-plan ranges include the relocated final recap. `recap` in each beginner JSON supplies four to six concept cards, an illustrative example, exit question/answer, next-class link and instructor script.

## Lab exercise boundaries

`### BEGIN SOLUTION` and `### END SOLUTION` enclose instructor code. `### STUB` supplies the student's starter code. Markdown answer blocks use `<!-- BEGIN ANSWER -->` and `<!-- END ANSWER -->`. Cells tagged `solution-only` are omitted from student notebooks. Preserve these markers when explaining or editing code.

Distribute the `.ipynb` notebooks. Developer Python exports live in `build/week_XX/`, retain the same TODOs or solutions, and translate `%pip` into subprocess calls for syntax and code comparisons. The builder injects `curriculum/labs/_setup.py` once after installation. It supplies `DEVICE` and `SMOKE`, defaults to CPU, fixes seeds, prints versions and provides `check()` and optional `ask_model()`. `USE_GEMINI = False` and `USE_OPENAI = False` prevent provider access. Enable exactly one for a synthetic comparison; unavailable requests return `None` for the local fallback. Week sources add targeted semantic `check_…()` helpers after TODOs and keep widget cells tagged `skip-test`. Install cells use the dated pins in `research/weeks/platform_2026.md`; local tests skip installation and record their actual versions.

## Measured results

Numbers from real runs never go straight into slide or note text. They flow through one path:

1. A full lab run with `GENAI_RESULTS_PATH` set writes its export to a chosen scratch path. Compare the measured export and its settings before promoting a separate `curriculum/assets/week_XX/results_cpu_core.json`. Original `results.json`, `metrics.json`, attention tables and their plots remain archived evidence unless explicitly replaced. Asset scripts retain their own measured schemas.
2. `tools/fill_results.py --weeks N` turns those files into the sentences, tables and numbers in `fill.json`.
3. Slides and notes contain `{{PLACEHOLDER}}` names; `build_week.py` substitutes them and refuses to build while one is unfilled.

Record successful full CPU cores under `iteration_3_full_runs` in `research/lab_runs_2026-10.json`, with device, measured seconds, date, environment and source/executed notebook hashes. `build_course.py` selects the latest successful full/core CPU record for every week; missing current evidence fails the guide build. Its current `LAB_TABLE` is computed from those records, while `LAB_INFO` supplies an explicitly archived earlier-experiment table. Smoke outputs never feed `fill.json`. Installation, downloads, interactive cells and optional stretches need separate evidence.

For weeks 9–12, `harvest_figures.py --week N --dry-run` validates the complete figure set without changing assets. Harvesting reads `executed_solutions_full.ipynb` and its successful `lab_test_full.json`, checks code against the built notebook in order, and rejects missing or damaged images before writing any. Later smoke runs cannot replace this source. New run records also hash the executed notebook; older records lack that field and retain their original evidence limits. `test_lab_evidence.py` exercises these guards in temporary folders.

`compare_lab_code.py` compares ordered Python syntax trees, including string contents. Comment or formatting edits are allowed; reordered or duplicated cells are changes. Missing or failed full-run records return a failure status. Shared setup defaults `USE_GEMINI` and `USE_OPENAI` to false. Enable exactly one for optional `ask_model()` calls; credentials alone do not enable requests, and smoke mode skips both. The helpers cap both providers together at 40 attempts, cache responses, back off on HTTP 429 and never display credentials. Local users can select a private file through `LOCAL_ENV_FILE`; it is read only after API opt-in. OpenAI API usage may incur charges separately from ChatGPT subscriptions; local completion needs no API spending. Quotas and eligibility remain account-specific.

`check_links.py` makes at most two GET attempts for temporary server/transport failures and records each attempt. HTTP 403/429 remains blocked and unverified; HTTP 404 remains a failure. `test_link_checks.py` exercises these distinctions without network requests.

## Text conventions

Speaker notes require exactly one ordered `In short:`, `Explain:`, `Example:`, `Ask:`, `Answer:` and `Watch out:`. Total whitespace-delimited length is 110–180 words including labels. Explain contains three to six sentences of at most twenty words. `tools/speaker_notes.py` validates the template; legacy notes now fail the ordinary build and prose audit. `audit_prose.py --strict-speaker-notes` explicitly selects the same strict policy. Quiz notes retain `Answer: X` before the explanation, matching the correct option.

- Inline maths is written `$…$` in the notes and in slide text. The build converts it to Unicode with real sub- and superscripts and wraps it in `<var>…</var>`, which `notes.js` uses to set single-letter variables in italics and digits, operators and Greek letters upright. A `$` directly before a digit never closes maths, so prices such as "$0.10 input/$0.40 output" stay as text.
- In the Word notes, companion headings are `## Slide N: Title`, where N and Title are those of the 46-slide deck; `verify_beginner.py` fails on a mismatch.
- Bare `<tags>` in Markdown are dropped by Word and Jupyter; put them in `code`. Use →, −, ×, en dashes and "≈ 80%" (with a space); `audit_prose.py` checks these.
- Quiz slides store the correct option as `answer` (0–3), and the speaker notes say "Answer: X" (or "Answer X") with the same letter; `audit_prose.py` checks this and the spread of letters.

## Build and check

```powershell
.venv/Scripts/python.exe tools/build_week.py --weeks 1-12
.venv/Scripts/python.exe tools/build_course.py
.venv/Scripts/python.exe tools/verify_beginner.py
.venv/Scripts/python.exe tools/test_speaker_notes.py
.venv/Scripts/python.exe tools/test_lab_helpers.py
.venv/Scripts/python.exe tools/test_lab_setup.py
.venv/Scripts/python.exe tools/test_lab_runtime.py
.venv/Scripts/python.exe tools/test_lab_evidence.py
.venv/Scripts/python.exe tools/test_link_checks.py
.venv/Scripts/python.exe tools/test_output_files.py
node tools/test_output_files.js
.venv/Scripts/python.exe tools/test_beginner_support.py
.venv/Scripts/python.exe tools/test_technical_figures.py
node tools/test_teaching_notes.js
node tools/test_technical_diagrams.js
.venv/Scripts/python.exe tools/audit_inline_math.py
.venv/Scripts/python.exe tools/audit_coverage.py
.venv/Scripts/python.exe tools/audit_slide_text.py
.venv/Scripts/python.exe tools/audit_prose.py
.venv/Scripts/python.exe tools/audit_flat_bullets.py
.venv/Scripts/python.exe tools/lint_figures.py --weeks 1-12
.venv/Scripts/python.exe tools/audit_figure_text.py --weeks 1-12
.venv/Scripts/python.exe tools/verify_pilot_artifacts.py
.venv/Scripts/python.exe tools/check_links.py
.venv-labs/Scripts/python.exe tools/test_labs.py --weeks 1-12
.venv/Scripts/python.exe tools/render.py --weeks 1-12
.venv/Scripts/python.exe tools/build_revision_review.py
```

Use `--weeks 12` for an individual week, or `--labs-only` to rebuild notebooks and developer exports after a lab-only edit. After changing executable lab logic, execute the affected solution notebook with `tools/test_labs.py --weeks N --full --device cpu`. The runner owns `build/.busy` until it finishes, records the runtime environment and preserves mode-specific executed notebooks. Do not build or render the same week during execution. Rendering requires the local Microsoft Office installation and Poppler used by this repository.

Iteration 3 closure (12 October 2026): all 26 release gates, all 12 final smoke runs, all 12 measured CPU cores and complete rendered review passed for the recorded release. `research/ITERATION3_2026-10-12.json` binds the actual checks and views to current artifacts. A later source change needs the relevant rebuild, checks and changed-output review; prior evidence cannot certify new content.

The renderer also writes `slides.layout.json` with text boxes that exceed PowerPoint's measured text bounds. Review those boxes and the rendered images before distributing a rebuild. Structural checks confirm correct attachments and editable objects; visual review checks the actual page layout.

Generated PowerPoint, Word, notebook and Python exports are written to unique adjacent temporary files, then replaced when complete. Bounded retries allow brief Windows reader locks to clear; an exhausted retry fails the build and preserves the previous destination. Atomic-writer tests cover replacement, temporary locks and failure preservation.

## Revision loop

The full routine, with the rules, commands and a table of mistakes to look for, is in [research/REVIEW_PROCESS.md](research/REVIEW_PROCESS.md). Edit slide YAML with `tools/yaml_edit.py` (`edit_slides`), which changes only the keys you name and checks that nothing else moved.

Revise one week from its sources, rebuild, run the lab smoke check, render with Office, and inspect every individual slide/page/diagram. Record a defect, explain the correction, then rebuild and inspect the affected output again. Keep source and image hashes in the visual manifests so old previews cannot certify a new build. Measured results stay separate from illustrative arithmetic and smoke outputs.
