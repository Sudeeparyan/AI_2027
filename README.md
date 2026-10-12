# Generative AI (MSc in Artificial Intelligence): 12-week teaching pack

Everything needed to teach the **Generative AI** module described in `context/Descriptor - Generative AI-old.docx`, with a proposed revision of descriptor **Section 7.3** only.

## Start here: `deliverables/`

| Folder | Contents |
|---|---|
| `deliverables/00_Course/` | **Instructor_Guide.docx** (how to use the pack, course map, MIMLO matrix, labs, licences, maintenance) · **Course_Glossary.docx** (consistent definitions across all twelve weeks) · **Section_7.3_Rationale.docx** (what changed in 7.3 and why, original vs proposed text) · **Descriptor - Generative AI (7.3 revised).docx** (clean) · **Descriptor - Generative AI (7.3 tracked changes).docx** (same edit as Word tracked changes) |
| `deliverables/Week_01_…` to `Week_12_…` | `Week_XX_Lecture_Slides.pptx` (46 slides with speaker notes) · `Week_XX_Teaching_Notes.docx` · student and solution `.ipynb` labs · `Diagrams/` (PNG and editable SVG) |

Only the Detail and Tutorials columns of Section 7.3 differ from the original descriptor; every other part of the Word file is byte-identical (checked by `tools/build_descriptor.py`). Lecture topics and week order are unchanged.

## Teaching students who are new to the subject

Each week shows its place on the twelve-week course map, introduces starting vocabulary and includes five technical block diagrams: the overall flow, the core mechanism, how to build or prepare the system, how to use it, and the lab sequence. Blocks name their operations and outputs and connect to actual functions or variables in the lab. Solid arrows carry data. Dashed arrows show a step that repeats. The Word notes include worked examples, questions with instructor explanations, common points of confusion and a code map. Technical terms and equations remain, with plain explanations before the detail. The final slide is a recap infographic with an illustrative example, an exit question and the connection to the next class. Use the answers to decide which explanation to revisit.

Start with the vocabulary slide, ask students to trace one example through the overall flow, then introduce the detailed theory. Use each detailed diagram beside its related topic. In the lab, students follow the embedded diagrams and the step-to-code table before running cells. The notebook embeds all seven PNGs (five technical diagrams, the course map and the final recap), so uploading it to Colab needs no separate image files. Open the notebooks directly in Colab or VS Code. Developer Python exports live in `build/week_XX/` for code comparison. Student files retain TODO exercises; solution files are for the instructor. Complete a TODO before running a cell that depends on it.

Speaker notes use **In short**, **Explain**, **Example**, **Ask**, **Answer** and **Watch out**, in that order, with 110–180 words including the labels. Explain has three to six short sentences. Each TODO has a hint and a semantic `check_…()` helper; each lab part offers a core task, one setting to change, common errors and an optional stretch.

See [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for the source map and build flow. Edit `curriculum/beginner/week_XX.json` to change the explanations and diagrams together, then rebuild that week.

See the [visual review gallery](research/revision_review.html) for all twelve recaps, shared diagrams and rendered slides. The [October revision report](research/REVISION_2026-10.md) records changes by week, verification evidence and remaining execution limits. The [audit](research/AUDIT_2026-10.md) records the defects found and their fixes. To improve a week again, follow [the review routine](research/REVIEW_PROCESS.md): it lists the rules, the commands in order and the mistakes found so far.

## How the pack is built

The Office files and notebooks are **generated** from plain-text sources, so edit the sources, not the outputs.

```
curriculum/
  section_7_3.yaml        revised 7.3 (single source of truth for the descriptor and all weeks)
  weeks/week_XX.yaml      40 core slides per week with speaker notes
  beginner/week_XX.json   vocabulary, five diagrams, recap infographic, worked examples and code map
  glossary.json          one course vocabulary, reused in weekly glossaries and Course_Glossary.docx
  notes/week_XX.md        teaching notes (Markdown with $$maths$$, figures, tables)
  labs/week_XX_lab.py     lab source (percent format; solutions marked for the student version)
  figures/week_XX.py      diagrams (matplotlib)
  assets/                 real results: asset scripts make_week_XX.py, metrics/results JSON, harvested lab figures
  course/                 instructor guide and 7.3 rationale sources
  sources.yaml            reference ledger (verification results in build/link_check.json)
tools/                    builders and checks (see below)
```

| Command (from the project folder) | What it does |
|---|---|
| `.venv/Scripts/python.exe tools/build_descriptor.py` | Builds and verifies the two descriptor copies |
| `.venv/Scripts/python.exe tools/build_week.py --weeks 1-12` | Builds slides, notes and notebooks for the given weeks (comma list or range), with validation |
| `.venv/Scripts/python.exe tools/build_course.py` | Builds the instructor guide, shared glossary and 7.3 rationale |
| `.venv-labs/Scripts/python.exe tools/test_labs.py --weeks 1-12` | Executes the **built** solution notebooks in quick smoke mode. Use `--full --device cpu` for measured required cores and a scratch `GENAI_RESULTS_PATH` for new exports. Rebuild after a lab edit |
| `.venv/Scripts/python.exe tools/harvest_figures.py --week 12` | Copies figures from a successful mode-specific full run (weeks 9–12); rejects stale code or missing images before changing assets. Add `--dry-run` to validate without writing |
| `.venv/Scripts/python.exe tools/fill_results.py --weeks 1-12` | Turns measured assets, including separate `results_cpu_core.json` records, into the `{{PLACEHOLDER}}` values on slides and in notes; retains archived result provenance |
| `.venv/Scripts/python.exe tools/render.py --weeks 1-12` | Renders slides and notes to PNG contact sheets for visual review (needs Microsoft Office) |
| `.venv/Scripts/python.exe tools/check_links.py` | Checks every link in the ledger, slides, notes and labs |
| `.venv/Scripts/python.exe tools/audit_inline_math.py` | Checks that inline maths in the notes converts cleanly |
| `.venv/Scripts/python.exe tools/audit_coverage.py` | Checks that every Section 7.3 Detail item appears in that week's slides or notes |
| `.venv/Scripts/python.exe tools/verify_beginner.py` | Checks shared diagrams, notebook attachments, Python syntax, student/solution separation and editable slide objects |
| `node tools/test_technical_diagrams.js` | Checks routes, text fit, readable fonts and shared native/SVG scenes, including course maps and recap infographics |
| `.venv/Scripts/python.exe tools/test_technical_figures.py` | Checks U-Net, DiT, transformer, multimodal and agent connections, decision branches, text fit and diagram borders |
| `.venv/Scripts/python.exe tools/audit_slide_text.py` | Checks the actual PowerPoint instructional text for the 14-point minimum |
| `.venv/Scripts/python.exe tools/test_beginner_support.py` | Checks consistent glossary definitions and references to the moved recap slide |
| `.venv/Scripts/python.exe tools/build_revision_review.py` | Updates the local twelve-week visual gallery after rendering |
| `.venv/Scripts/python.exe tools/audit_prose.py` | Checks reader-facing text for hyphen minus signs, ranges and spaced hyphens, lost symbols, lower-case sentence starts, doubled words, `->` arrows, `<tags>` that Word would drop, repeated speaker-note sentences, class questions without instructor answers, and quiz answer patterns |
| `.venv/Scripts/python.exe tools/audit_flat_bullets.py` | Finds slide steps or lab boxes printed as loose, unlabelled bullets in the Word notes |
| `.venv/Scripts/python.exe tools/lint_figures.py --weeks 1-12` | Measures every drawn figure for text outside its box, overlapping labels and legends lying over data |
| `.venv/Scripts/python.exe tools/audit_figure_text.py --weeks 1-12` | Checks that text inside projected pictures is at least 14 pt |
| `.venv/Scripts/python.exe tools/notes_sheets.py --weeks 6` | Tiles the rendered Word pages four to a sheet for review; `--weeks 1-12 --slide 44` tiles one slide from every week |
| `.venv/Scripts/python.exe tools/compare_lab_code.py -v` | Compares executable cells in order with a successful full run; detects changed, reordered or duplicated code while ignoring comments and formatting |
| `.venv/Scripts/python.exe tools/test_lab_evidence.py` | Checks full-run evidence, safe figure harvesting and explicit provider opt-in without downloads or real provider calls |
| `.venv/Scripts/python.exe tools/test_link_checks.py` | Checks bounded retries for temporary link errors; missing pages still fail and blocked pages remain unverified |
| `.venv/Scripts/python.exe tools/test_output_files.py` and `node tools/test_output_files.js` | Checks complete output replacement, brief Windows locks and preservation of the previous file when writing fails |
| `.venv/Scripts/python.exe tools/verify_pilot_artifacts.py` | Checks that `context/`, `sources/` and `legacy/` are unchanged (SHA-256 baseline) and re-checks the Week 5 pilot outputs |

Set `PYTHONIOENCODING=utf-8` on Windows consoles.

## Setting up the build environment

1. **Python 3.12** virtual environments: `.venv` for the tools (`pip install -r tools/requirements-tools.txt`) and `.venv-labs` for executing labs (install PyTorch, then `pip install -r tools/requirements-labs.txt`). A CUDA environment (`.venv-gpu`) is optional and only speeds up the instructor runs.
2. **Node.js 20+**: `cd tools && npm install` (pptxgenjs, docx, marked, react-icons, sharp).
3. **Microsoft Office** (optional) for rendering previews with `tools/render.py`.

Students use the notebooks in free Colab with its CPU runtime, or in local VS Code/Jupyter. Every required core defaults to CPU; GPU, longer training and remote deployment are optional stretches. Week 8 includes real bounded LoRA fine-tuning. The instructor guide computes its current runtime table from the latest successful CPU records in `research/lab_runs_2026-10.json` → `iteration_3_full_runs`. Those cached laptop measurements exclude installation, downloads and interactive/provider stretches. Earlier GPU and larger-model plots retain separate archived labels and results. No clean-install or timed Colab claim is made.

Upgraded notebooks offer optional Gemini or OpenAI API access through `USE_GEMINI` and `USE_OPENAI`; both default to `False`. Enable one provider to compare an invented teaching prompt with the local result. Keys come from Colab Secrets, environment variables, or the explicitly selected private local `.env` file after opt-in. OpenAI API usage is billed separately from a ChatGPT subscription. Every core task has a local fallback and needs no API key. Provider documentation and checked SDK versions are recorded in `research/weeks/platform_2026.md`. Shared `ask_model()` returns text or `None`; exercises keep a local answer or rubric when the optional route is unavailable.

Iteration 3 is complete (12 October 2026): all 26 release gates and all 12 final lab smoke runs passed. Successful full CPU-core evidence covers every week (all below 15 minutes), and current executable code matches those runs. The final visual review has no open material findings. See `research/REVISION_2026-10.md` and `research/ITERATION3_2026-10-12.json` for current hashes, view coverage and unverified classroom conditions.

## Other folders

* `context/` – the head professor's descriptor template and the original course document (inputs; not modified).
* `sources/` – the eleven supplied source documents (inputs; not modified).
* `research/` – research notes and the verified-source list used while revising the curriculum.
* `legacy/` – the previous generator, its content and outputs (including the earlier Codex revision), kept for reference only. They do not follow Section 7.3 and are not used by the new pipeline. The previous README is `legacy/README_legacy.md`.
