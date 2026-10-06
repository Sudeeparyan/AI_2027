# Generative AI (MSc in Artificial Intelligence): 12-week teaching pack

Everything needed to teach the **Generative AI** module described in `context/Descriptor - Generative AI-old.docx`, with a proposed revision of descriptor **Section 7.3** only.

## Start here: `deliverables/`

| Folder | Contents |
|---|---|
| `deliverables/00_Course/` | **Instructor_Guide.docx** (how to use the pack, course map, MIMLO matrix, labs, licences, maintenance) · **Section_7.3_Rationale.docx** (what changed in 7.3 and why, original vs proposed text) · **Descriptor - Generative AI (7.3 revised).docx** (clean) · **Descriptor - Generative AI (7.3 tracked changes).docx** (same edit as Word tracked changes) |
| `deliverables/Week_01_…` to `Week_12_…` | `Week_XX_Lecture_Slides.pptx` (44 slides with speaker notes) · `Week_XX_Teaching_Notes.docx` · student and solution labs in `.ipynb` and `.py` · `Diagrams/` (PNG and editable SVG) |

Only the Detail and Tutorials columns of Section 7.3 differ from the original descriptor; every other part of the Word file is byte-identical (checked by `tools/build_descriptor.py`). Lecture topics and week order are unchanged.

## Teaching students who are new to the subject

Each week now introduces starting vocabulary and three technical block diagrams: the overall flow, the core mechanism, and the lab sequence. Blocks name their inputs and outputs and connect to actual functions or variables in the lab. The Word notes include worked examples, questions with instructor explanations, common points of confusion and a code map.

Start with the vocabulary slide, ask students to trace one example through the overall flow, then introduce the detailed theory. Use the mechanism diagram beside its related topic. In the lab, students read the two embedded diagrams and the step-to-code table before running cells. The notebook embeds the PNGs, so uploading it to Colab needs no separate image files. The Python exports use percent-format cells for VS Code and link to the adjacent `Diagrams/` folder. Student files retain TODO exercises; solution files are for the instructor.

See [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for the source map and build flow. Edit `curriculum/beginner/week_XX.json` to change the explanations and diagrams together, then rebuild that week.

## How the pack is built

The Office files and notebooks are **generated** from plain-text sources, so edit the sources, not the outputs.

```
curriculum/
  section_7_3.yaml        revised 7.3 (single source of truth for the descriptor and all weeks)
  weeks/week_XX.yaml      40 core slides per week with speaker notes
  beginner/week_XX.json   vocabulary, three diagrams, worked examples and code map
  notes/week_XX.md        teaching notes (Markdown with $$maths$$, figures, tables)
  labs/week_XX_lab.py     lab source (percent format; solutions marked for the student version)
  figures/week_XX.py      diagrams (matplotlib)
  assets/                 real results: asset scripts make_week_XX.py, metrics/results JSON, harvested lab figures
  course/                 instructor guide and 7.3 rationale sources
  sources.yaml            reference ledger (all links checked)
tools/                    builders and checks (see below)
```

| Command (from the project folder) | What it does |
|---|---|
| `.venv/Scripts/python.exe tools/build_descriptor.py` | Builds and verifies the two descriptor copies |
| `.venv/Scripts/python.exe tools/build_week.py --weeks 1-12` | Builds slides, notes and notebooks for the given weeks (comma list or range), with validation |
| `.venv/Scripts/python.exe tools/build_course.py` | Builds the instructor guide and the 7.3 rationale |
| `.venv-labs/Scripts/python.exe tools/test_labs.py --weeks 1-12` | Executes the **built** solution notebooks in quick smoke mode (`--full` for real settings; set `GENAI_RESULTS_PATH` to save results). Rebuild a week after editing its lab |
| `.venv/Scripts/python.exe tools/harvest_figures.py --week 12` | Copies figures from an executed full run (weeks 9–12) into `curriculum/assets/` |
| `.venv/Scripts/python.exe tools/fill_results.py --weeks 4,6,7,8,9,10,11,12` | Turns measured results into the values shown on "real" slides |
| `.venv/Scripts/python.exe tools/render.py --weeks 1-12` | Renders slides and notes to PNG contact sheets for visual review (needs Microsoft Office) |
| `.venv/Scripts/python.exe tools/check_links.py` | Checks every link in the ledger, slides, notes and labs |
| `.venv/Scripts/python.exe tools/audit_inline_math.py` | Checks that inline maths in the notes converts cleanly |
| `.venv/Scripts/python.exe tools/audit_coverage.py` | Checks that every Section 7.3 Detail item appears in that week's slides or notes |
| `.venv/Scripts/python.exe tools/verify_beginner.py` | Checks shared diagrams, notebook attachments, Python syntax, student/solution separation and editable slide objects |

Set `PYTHONIOENCODING=utf-8` on Windows consoles.

## Setting up the build environment

1. **Python 3.12** virtual environments: `.venv` for the tools (`pip install -r tools/requirements-tools.txt`) and `.venv-labs` for executing labs (install PyTorch, then `pip install -r tools/requirements-labs.txt`). A CUDA environment (`.venv-gpu`) is optional and only speeds up the instructor runs.
2. **Node.js 20+**: `cd tools && npm install` (pptxgenjs, docx, marked, react-icons, sharp).
3. **Microsoft Office** (optional) for rendering previews with `tools/render.py`.

Students need none of this: the notebooks run on Google Colab.

## Other folders

* `context/` – the head professor's descriptor template and the original course document (inputs; not modified).
* `sources/` – the eleven supplied source documents (inputs; not modified).
* `research/` – research notes and the verified-source list used while revising the curriculum.
* `legacy/` – the previous generator, its content and outputs (including the earlier Codex revision), kept for reference only. They do not follow Section 7.3 and are not used by the new pipeline. The previous README is `legacy/README_legacy.md`.
