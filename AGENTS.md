# AGENTS.md: Generative AI teaching pack (read first, every session)

Read this file, then `research/TASKS.md`. Update the board after each task.
Teacher override (10–11 Oct 2026): continue through all tasks without approval stops; optional
Gemini or OpenAI API access is permitted, with a complete local fallback and no required API spending.
`improvements.md` is an older, superseded prompt: do not read it.

## What this is
A 12-week MSc AI module, "Generative AI" (2 h lecture + 2 h lab per week). Each week the students get
a PowerPoint (used to teach), Word teaching notes (reference) and a Jupyter lab (student + solutions).
Students are graduates from other fields: new to AI, beginners in Python. Keep MSc depth (equations,
real implementation, fine-tuning), but explain it the way a great teacher does at a whiteboard.

## Hard rules
1. Edit sources only (`curriculum/`, `tools/`), then rebuild. Never hand-edit `deliverables/`.
   Never change `context/`, `sources/` or `legacy/`.
2. Binding template: `context/Descriptor - Generative AI-old.docx`. Only Section 7.3 (Detail, Tutorials)
   may change. Week topics and order come from `curriculum/section_7_3.yaml`.
   `tools/audit_coverage.py` stays at 100%.
3. `context/01_Generative_AI_2027.docx` is a different proposal. Borrow only the teaching ideas listed in
   TASKS.md; never its week order, outcomes or assessment.
4. Results shown on slides or in notes come only from measured runs:
   `curriculum/assets/week_XX/*.json` → `tools/fill_results.py` → `fill.json` → `{{PLACEHOLDER}}`.
   Every quoted runtime needs a record in `research/lab_runs_2026-10.json`. Never invent a number.
5. About 46 slides per deck. To add a slide, merge or cut one and move the detail to the Word notes.
6. No git commit or push unless the teacher asks. Never revert changes you did not make.
7. Say a check passed only if you ran it in this session and saw it pass.

## Teaching standard
- Order for every topic: why it matters (real example) → intuition or analogy → worked example with real
  numbers or tensor shapes → diagram → precise term → equation, with every symbol explained in words.
- Slides: one idea each; short phrases; ≤ 40 words of body text; text ≥ 14 pt. Prefer an informative
  visual (example strip, shape diagram, mini-table) to bullets or a decorative icon.
- Speaker notes serve the teacher AND a student who missed the class. Template, 110–180 words:
  `In short:` one sentence. `Explain:` 3–6 plain sentences (≤ 20 words each), defining every new term.
  `Example:` real numbers, shapes or an analogy, and where the analogy breaks.
  `Ask:` a class question. `Answer:` its answer. `Watch out:` one common mistake.
- Word notes use the same words and go deeper: step-by-step worked examples, misconceptions/FAQ,
  self-check questions with answers. `curriculum/glossary.json` is the single glossary for all weeks.
- Diagrams: each block is a real operation. Each arrow is labelled with what flows, with shapes where
  useful. Solid arrows carry data; dashed arrows show feedback or repetition. Flow runs left→right or
  top→bottom, with no crossings and at most 7 blocks. Colour blocks by role (colourblind-safe) and add
  a legend. The PPT, SVG, PNG, Word and notebook copies must be identical.
- Interactive: predict-then-reveal questions on slides (answer in the notes). Labs get ipywidgets
  sliders for the key setting; tag widget cells `skip-test`.

## Lab standard
- Notebooks (.ipynb) only, for labs and practice. They must run in free Google Colab and in local VS Code,
  on the CPU by default. A GPU is optional, never required. The core path should take ≤ 15 min of
  waiting on a CPU (measure it). Longer training is an optional stretch, with stored results shown instead.
- Optional APIs: Gemini free tier via `google-genai`, or OpenAI via its official SDK. The latter
  may incur charges, separately from a ChatGPT subscription; API spending is never required.
  Students enable exactly one of `USE_GEMINI = False` / `USE_OPENAI = False` in shared setup.
  Keys come from Colab Secrets, environment variables, or an explicitly selected local `.env` file;
  read them only after opt-in and never print or embed them. Keep a complete local fallback.
  Cap both providers together at 40 attempts per lab, including bounded HTTP 429 retries.
  Send invented examples only and explain the relevant provider data-use terms.
- Python-beginner support:
  - Before every code cell, a short Markdown cell: what the cell does, why, and the expected output.
  - Plain-English comments, and printed tensor shapes.
  - Every TODO has a hint and is followed by a `check_…()` that prints ✅ or a helpful message.
  - Each part has a "common errors" box.
- Structure: read → predict → run → change → check. Each part has a core task, a "change one thing"
  task and an optional stretch. End with 3–5 self-check questions (answers in the solutions notebook).
- Keep the `### BEGIN/END SOLUTION`, `### STUB` and `<!-- BEGIN/END ANSWER -->` markers.

## Where things are
Week N sources: `curriculum/weeks/week_NN.yaml` (40 core slides + speaker notes),
`curriculum/beginner/week_NN.json` (vocabulary, 5 diagrams, recap, code map),
`curriculum/notes/week_NN.md` (Word notes), `curriculum/labs/week_NN_lab.py` (percent-format lab),
`curriculum/figures/week_NN.py` (matplotlib). Engine: `tools/build_week.py`, `beginner.py`,
`technical_diagrams.js`, `slides.js`, `notes.js`. More detail: `CODEBASE_GUIDE.md`. The review loop and
the table of past defects are in `research/REVIEW_PROCESS.md`; read that table before any review task.

## Commands (project folder, PowerShell, `$env:PYTHONIOENCODING="utf-8"`)
- Build: `.venv/Scripts/python.exe tools/build_week.py --weeks N` (add `--labs-only` after a lab-only edit)
- Smoke-test the BUILT lab: `.venv-labs/Scripts/python.exe tools/test_labs.py --weeks N` (`--full` for real settings)
- Render to PNG: `.venv/Scripts/python.exe tools/render.py --weeks N`, then `tools/notes_sheets.py --weeks N`
- Text audits: `audit_prose.py`, `audit_inline_math.py`, `audit_slide_text.py`, `audit_flat_bullets.py`, `audit_coverage.py`
- Full gate list: "Build and check" in `CODEBASE_GUIDE.md`

## Pitfalls (each one cost hours)
- Sources are CRLF and contain backslashes. Edit them with the Edit tool or a Python script file, never a
  shell heredoc. Edit slide YAML with `tools/yaml_edit.py` (`edit_slides`). In single-quoted YAML, write
  an apostrophe as `''`.
- `test_labs.py` runs the notebook in `deliverables/`, so rebuild before testing. A smoke run overwrites
  `build/week_NN/executed_solutions.ipynb`, so run `harvest_figures.py` only after a full run.
- Only one `render.py` at a time (it drives PowerPoint/Word). Never build a week while it is being
  rendered or tested. After a crashed render, kill stale POWERPNT/WINWORD processes.
- matplotlib mathtext has no `\nolimits` or `\textstyle`. fp16 Stable Diffusion gives black images on
  the local GTX 1650. Qwen3 thinking mode needs sampling (T 0.6, top-p 0.95, top-k 20).
- Product and model names change (e.g. Vertex AI → Gemini Enterprise Agent Platform). Check the vendor's
  page for every model name, price or rate limit; never rely on memory.

## Parallel work and token budget
- Each task in TASKS.md lists the files it owns; touch only those. Shared files belong to Engine (E) and
  Merge (M) tasks: `tools/`, `glossary.json`, `sources.yaml`, `figures/_style.py`, `curriculum/course/`,
  `AUDIT_2026-10.md`, `lab_runs_2026-10.json`, README and guides. Week tasks write proposed changes to
  shared files in their own `research/weeks/wNN_log.md`.
- Renders and full lab runs are machine-wide and run one at a time. Before one, create `build/.busy`
  containing your task ID, and delete it afterwards. If the file exists, wait or do other work.
- Read only the files your task names; use `grep`, line ranges and `| tail -20`. Don't re-explore the repo.
- Review images cheaply: contact sheets first, and open single images only where something looks wrong.
- Write tersely: findings as table rows, logs as bullets.
- Update your task's handoff line after each sub-step, not only at the end, so another session can
  resume if this one hits a usage limit.
