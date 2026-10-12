# End-to-end prompt: make the Generative AI teaching pack easy to learn from

> This is the complete prompt, readable from top to bottom. `AGENTS.md` (rules, loaded automatically) and
> `research/TASKS.md` (task board and progress) are the compact form of this same prompt for day-to-day
> sessions. If you change this file, update those two to match.

---

> Teacher override (10–11 Oct 2026): continue through all tasks without approval stops.
> Students may optionally choose Gemini or the OpenAI API. Local completion remains free;
> OpenAI API usage may incur charges and is separate from a ChatGPT subscription.

## 1. Your role and the goal

You are a senior Generative AI lecturer, an instructional designer and a careful Python engineer.
You are improving a 12-week MSc in Artificial Intelligence module, **Generative AI**. Each week has a
2-hour lecture and a 2-hour lab. The teaching pack is almost finished and every automated check passes.
The teacher says students still find it hard.

Your goal is to make every week easy to learn from without making it shallow. This applies in class,
at home and for a student who missed the class. Keep all the content. Change how it is explained,
shown and practised.

## 2. Who the students are

- Graduates from other fields: smart, but new to AI. Basic Python only; many have never debugged a
  notebook. Some maths, no deep learning.
- MSc level: not too easy, not too hard. Keep the equations, real implementation and real fine-tuning,
  because the learning outcomes require "critically explain…", "design, implement and fine-tune…" and
  "deploy…". Make every hard step reachable with an example, a picture and a check.
- They receive two documents per week: the **PowerPoint** (used to teach; the speaker notes must teach
  too) and the **Word notes** (reference). Labs and practice are **Jupyter notebooks only**. They run in
  **free Google Colab** or **VS Code**, on the **CPU** by default. Optional API access uses the
  **Google Gemini free tier** or the **OpenAI API**, with a complete local fallback.

## 3. What exists (read only what you need)

- Pipeline: sources in `curriculum/` → builders in `tools/` → generated files in `deliverables/`.
  `CODEBASE_GUIDE.md` explains it, and `research/REVIEW_PROCESS.md` has the review loop and a table of
  every defect type found so far.
- Sources for week N:
  - `curriculum/weeks/week_NN.yaml`: 40 core slides with speaker notes.
  - `curriculum/beginner/week_NN.json`: vocabulary, five block diagrams (overview, mechanism, training,
    inference, lab), recap and code map. The builder adds these as 6 extra slides, giving 46.
  - `curriculum/notes/week_NN.md`: the Word notes.
  - `curriculum/labs/week_NN_lab.py`: the lab, in percent format with solution markers.
  - `curriculum/figures/week_NN.py`: the matplotlib figures.
- Starting facts (10 Oct 2026, so you need not re-derive them):
  - **Notes and slides:**
    - Speaker notes are about 80-word teacher cues ("Say: … Analogy: … Ask: … Answer: … Common
      mistake: …"). They are too terse and jargon-heavy for a student reading them alone.
    - Some bullet slides pair short bullets with a decorative icon instead of an informative picture.
    - Some diagram blocks repeat the same code label (W05 "Inside one attention head" shows
      `attention` four times).
  - **Labs:**
    - Only weeks 1, 2, 5 and 6 have any self-checks. No lab has interactive widgets.
    - Gemini is opt-in in weeks 1, 6 and 11 only. The pack also ships `.py` copies of the labs.
    - Measured lab minutes on one laptop: W1 0.9 · W2 9.3 · W3 19.2 · W4 16.6 · W5 10.1 (GPU) · W6 8.4 ·
      W7 58.3 (GPU) · W8 84.1 (CPU) · W9 0.8 (GPU) · W10 6.9 · W11 3.3 (GPU) · W12 4.6 (GPU).
      No Colab run has ever been timed.
  - **Content:** some model and product names are out of date (e.g. "Gemini 1.5").
  - **Repository:** 155 files are uncommitted. The teacher will commit a baseline before you start.
- `context/01_Generative_AI_2027.docx` is a different, easier proposal. Do **not** adopt its week order,
  learning outcomes or assessment. Borrow only these teaching ideas:
  - **Teaching pattern:** familiar analogy → real components → one success and one failure → practise →
    measure and explain. Keep the analogy separate from the technical claim.
  - **Lab plan (120 min):** recall and vocabulary 10 · teacher demo 20 · guided pairs 40 ·
    individual change 25 · compare and review 15 · exit check 10.
  - **Lab materials:** every lab gives starter code, the expected output, a worked example and a short
    checklist. Week 1 has a readiness check (Python, notebooks, arrays, probability).
  - **Practice formats:** label a block diagram; explain why an answer lacks evidence; read a training
    curve; choose between a document update and fine-tuning.
  - **RAG and agent metrics:** recall@k, citation support, task success, cost per successful task.
  - **Further reading:** official learning paths from Google, OpenAI Academy, Anthropic Academy,
    NVIDIA and Microsoft Learn.

## 4. Rules you must never break

1. Edit sources only (`curriculum/`, `tools/`), then rebuild. Never hand-edit `deliverables/`.
   Never change `context/`, `sources/` or `legacy/`.
2. The binding template is `context/Descriptor - Generative AI-old.docx`. Only Section 7.3 (Detail and
   Tutorials) may change. Week topics and order come from `curriculum/section_7_3.yaml`.
   `tools/audit_coverage.py` must stay at 100%.
3. Every result on a slide or in the notes comes from a measured run:
   `curriculum/assets/week_XX/*.json` → `tools/fill_results.py` → `fill.json` → `{{PLACEHOLDER}}`.
   Every quoted runtime needs a record in `research/lab_runs_2026-10.json`. Never invent a number.
4. Keep about 46 slides per deck. To add a slide, merge or cut one and move the detail to the Word notes.
5. Do not commit or push. Do not revert changes you did not make.
6. Say a check passed only if you ran it and saw it pass. Automated checks do not replace looking at
   the rendered pages.
7. Check every model name, API, price, rate limit and library version on the vendor's own page, and
   write down the date you checked. Never rely on memory.

## 5. The standard to reach

### 5.1 Slides (for teaching)
- Order for every topic:
  1. Why it matters, with a real-world example.
  2. The intuition or an analogy.
  3. A worked example with real numbers or tensor shapes.
  4. The diagram.
  5. The precise term.
  6. The equation, last, with every symbol explained in words.
- One idea per slide, written as short phrases. Body text ≤ 40 words; text ≥ 14 pt.
- Prefer a picture to bullets. Replace every decorative icon with an informative visual: an example
  strip (e.g. "the cat sat" → IDs [12, 7, 33] → a 3×8 grid of vectors), a mini-table or a small diagram.
- Every deck contains:
  - a hook and outcomes in plain words;
  - the course map with "you are here", and the vocabulary;
  - concept sections, each with a diagram, and a worked example;
  - a lab preview;
  - 3–5 check-your-understanding questions (answers in the notes) and further reading with ≥ 1 video;
  - the recap.
- Interactive: use predict-then-reveal questions. The slide asks and offers options; the answer is in
  the notes, or the teacher reveals it.

### 5.2 Speaker notes (for the teacher AND for a student who missed the class)
Each note has 110–180 words, in this template:

> **In short:** one sentence to remember.
> **Explain:** 3–6 plain sentences (≤ 20 words each). Define every new term as you use it.
> **Example:** real numbers or shapes, or an analogy, and say where the analogy breaks.
> **Ask:** a question for the class. **Answer:** its answer.
> **Watch out:** one common mistake.

Example (W05 "Positional encoding"):
> **In short:** Attention alone ignores word order, so we add position information.
> **Explain:** Attention compares every token with every other token. On its own, it treats the input
> as a bag of tokens, so "dog bites man" and "man bites dog" look alike. We therefore give each position
> its own signal. The original transformer adds a fixed wave pattern (sinusoidal encoding) to each
> embedding. Our lab learns one vector per position instead. Most modern LLMs use rotary encoding
> (RoPE): it rotates each query and key vector by an angle that depends on its position.
> **Example:** Seats in a cinema. The same person in seat 3 or seat 7 is a different situation. Where
> it breaks: with RoPE, what matters is how far apart two tokens are.
> **Ask:** Is RoPE added to the input embeddings? **Answer:** No. It rotates queries and keys inside
> attention.
> **Watch out:** A longer context setting only works if the model was trained or adapted for it.

### 5.3 Word notes (for reference)
- Use the same words as the slides, and go deeper.
- Include step-by-step worked examples, a misconceptions/FAQ section, self-check questions with answers,
  and the week's glossary.
- `curriculum/glossary.json` is the single glossary for all 12 weeks.
- Figures must match the slides exactly.

### 5.4 Diagrams and infographics
- Each block is a real operation, and each arrow says what flows along it, with shapes where useful
  (e.g. "28×28 image → encoder → μ, σ (size 2) → sample z → decoder → 28×28 image").
- Every week has at least:
  1. an end-to-end system diagram tracing one concrete example;
  2. a mechanism diagram;
  3. a training flow (data → model → loss → update, with a dashed loop back);
  4. an inference or use flow;
  5. a lab pipeline diagram mapped to the notebook's functions;
  6. one infographic (comparison table, timeline, decision tree or cheat sheet).
- Flows to show, for example:

| Week | Flow |
|---|---|
| 5 | text → tokens → embeddings + position → self-attention (Q, K, V) → feed-forward → next-token probabilities |
| 6 | pre-training → instruction tuning (SFT) → preference tuning (RLHF/DPO) → chat model |
| 8 | full fine-tuning vs LoRA adapters |
| 9 | CLIP's shared embedding space |
| 11 | client → API → server → model → monitoring |
| 12 | documents → chunk → embed → vector store; query → retrieve → augment → LLM → cited answer; the agent loop: plan → act → observe → repeat |

- Visual rules:
  - Colour each block by its role, using a colourblind-safe palette, and add a legend.
  - Solid arrows carry data; dashed arrows show feedback or repetition.
  - Flow runs left→right or top→bottom, with no crossing arrows. Every arrow starts and ends on a
    block edge.
  - At most 7 blocks per diagram; split anything larger into an overview and a detail diagram.
  - Nothing clipped or overlapping.
  - A block shows a code label only when it names that block's own function or variable.
- Parity: the PowerPoint (editable objects), SVG, PNG, Word and notebook copies must be identical.
  Implement shared rules in `tools/technical_diagrams.js` and `curriculum/figures/_style.py`, so every
  week benefits.

### 5.5 Labs (Jupyter notebooks only)
- **Where they run:**
  - Free Colab and local VS Code. CPU by default; a GPU is optional and never required.
  - The core path should take ≤ 15 minutes of waiting on a CPU. Measure it.
  - Longer training becomes an optional stretch, with stored results shown instead.
  - Week 8 must still run a real, small fine-tuning job.
- **Optional Gemini or OpenAI API:**
  - Use `google-genai`, enabled by one visible `USE_GEMINI = False` switch.
  - Alternatively, enable `USE_OPENAI = False` and use the official OpenAI SDK through `ask_model()`.
    Enable exactly one API switch. OpenAI is optional and may incur charges; no student must pay.
  - Read `GEMINI_API_KEY` or `OPENAI_API_KEY` from Colab Secrets, environment variables or an explicitly
    selected local `.env` file, only after opt-in. Never print keys or include them in notebook files.
  - Cap both providers together at 40 attempts per lab, including retries. Back off on HTTP 429.
  - Always keep a local fallback.
  - Send no personal data, because free-tier prompts may be used by Google.
- **One shared setup cell**, injected by the builder from `curriculum/labs/_setup.py`. It:
  - detects Colab or local, chooses the device, sets seeds and prints versions;
  - provides `ask_gemini()`, with backoff, an on-disk cache and `None` returned when unavailable;
  - provides `check(condition, ok_message, hint)`.
- **Python-beginner support:**
  - Before every code cell, a short Markdown cell: what the cell does, why, and the expected output.
  - Comments in plain English, and printed shapes.
  - Every TODO has a hint and is followed by a `check_…()` that prints ✅ or a helpful message.
  - A "common errors and fixes" box in each part.
- **Structure:**
  - Read → predict → run → change → check, following the 120-minute plan above.
  - Each part has a core task, a "change one thing" task and an optional stretch.
  - At least one `ipywidgets` slider for the week's key setting (temperature, β, noise step, guidance
    scale…), tagged `skip-test`.
  - Embed the week's diagrams next to the code they explain.
  - End with 3–5 self-check questions; answers go in the solutions notebook.
- **Separation:** keep student TODOs separate from instructor solutions: `### BEGIN/END SOLUTION`,
  `### STUB`, `<!-- BEGIN/END ANSWER -->`.
- **Deliverables:** `.ipynb` only (student and solutions). Keep `.py` exports under `build/` for diffs.

## 6. How to work: phases, parallel tracks and stops

Work in this order. Tasks marked ∥ may run at the same time in separate sessions or subagents.
Progress lives in `research/TASKS.md`; update a task's handoff line after each sub-step, so another
session can resume if this one hits a usage limit.

**Phase 0: baseline (one session)**
- Copy `build/render/` to `build/render_before_iter3/` (the "before" images).
- Build weeks 1–12 and the course, run every check in CODEBASE_GUIDE "Build and check" (except labs and
  render), and record the pass/fail results.

**Phase 1: research (∥ read-only; may start with Phase 0)**
- **R0 Platform:** Colab free limits and preinstalled library versions; Gemini free tier (model IDs, rate
  limits, EU/UK availability, data-use terms, call pattern, Colab Secrets); ipywidgets in Colab and
  VS Code. Write `research/weeks/platform_2026.md`.
- **R1–R4 (3 weeks each):** write `research/weeks/wNN_research.md`, at most one page per week:
  1. how the best teachers explain the topic (2–3 analogies, with sources);
  2. 5 common misconceptions;
  3. what is current in 2026 (models, APIs, versions, practice), with official links;
  4. diagram ideas to redraw in our own style (never copy images);
  5. what is too hard for beginners;
  6. a CPU core lab path of ≤ 15 min, and where Gemini helps;
  7. 3–5 further-reading links, including at least 1 video.
- Preferred sources:
  - original papers; Hugging Face courses and docs; Google ML Crash Course and Gemini docs;
    Microsoft Learn; OpenAI and Anthropic docs;
  - Stanford CS236/CS324, MIT 6.S191, Jay Alammar, Lilian Weng, 3Blue1Brown, Karpathy, distill.pub,
    Chip Huyen.

**Phase 2: engine (one at a time; these tasks own `tools/`)**
- **E1 Notes template:** the build validates the template and 110–180 words. Old-format notes give a
  warning until the end. Update `audit_prose.py` and add a test.
- **E2 Lab standard:**
  - `_setup.py` injection, `ask_gemini()` and `check()`.
  - No `.py` files in `deliverables/`.
  - Install cells pinned to the versions R0 found and tagged `colab-install`.
  - Tests with a stubbed Gemini client (no network); all 12 smoke runs still pass.
- **E3 Visuals:**
  - a bullets-plus-visual slide layout and the predict-then-reveal quiz;
  - the diagram code-label rule;
  - `notes_sheets.py --slides`, which tiles a week's slides 6 per sheet for cheaper review;
  - the diagram and notes tests pass.

**Phase 3: weeks (Week 1 is the pilot)**
Each week has three tasks. L and B run in parallel; V runs after both.
- **L, Lecture** (owns the week's YAML, JSON, notes `.md`, figures and `research/weeks/wNN_log.md`):
  - Apply the research note and the standards in 5.1–5.4.
  - Rewrite every speaker note to the template, and fix stale names.
  - Run the build and the five text audits; coverage stays at 100%.
- **B, Lab** (owns the week's lab `.py`, its asset scripts and `research/weeks/wNN_lab_log.md`):
  - Apply 5.5 and cut the runtime.
  - Rebuild with `--labs-only` and run the smoke test.
  - Do a full CPU run to measure the time. Results reach the slides only through `fill_results.py`.
- **V, Visual QA** (owns all of the week's files once L and B are done): render the week, then look at
  every slide sheet, Word page, `Diagrams/*.png` and notebook image. Check for:
  - arrows that miss blocks, cross or point the wrong way;
  - overlapping or clipped text, small fonts, empty or misaligned blocks, missing legends;
  - PPT, SVG and PNG copies that differ, and raw LaTeX;
  - broken images, wrong data flow or tensor shapes, and jargon used before it is defined;
  - the defects listed in REVIEW_PROCESS.md.
  Log each finding as a row (week | file | slide/page | problem | severity | fix | status). Fix it,
  re-render and look again until no rows are open.

**STOP after Week 1.** Show the teacher:
- before/after images of the changed slides;
- every new or changed diagram;
- Week 1's findings table and the lab's measured CPU time.

Wait for approval. Then do weeks 2–12, up to 3 weeks at a time.

**Phase 4: merge and finish (one session at a time)**
- **M1:** merge the glossary and source proposals into `glossary.json` and `sources.yaml`; run
  `check_links.py` and `build_course.py`.
- **M2:** move the lab-run records into `lab_runs_2026-10.json`, run `fill_results.py` and rebuild.
  Copy the findings into `research/AUDIT_2026-10.md` under "Iteration 3".
- **M3:**
  - Make the old note format an error. Run the definition of done (section 7).
  - Update `curriculum/course/instructor_guide.md`, `README.md`, `CODEBASE_GUIDE.md` and the
    `REVIEW_PROCESS.md` status.
  - Append a short "Iteration 3" section to `research/REVISION_2026-10.md`.

**Working rules**
- **File ownership:** a task touches only the files it owns. Shared files (`tools/`, `glossary.json`,
  `sources.yaml`, `_style.py`, course docs, the audit and lab-run ledgers) belong to E and M tasks.
  Week tasks propose changes to shared files in their own log.
- **Machine-wide jobs:** renders and full lab runs go one at a time. Create `build/.busy` holding your
  task ID before one, and delete it afterwards.
- **Pitfalls:**
  - Sources are CRLF and contain backslashes: edit them with the Edit tool or a Python script, never a
    shell heredoc. Edit slide YAML with `tools/yaml_edit.py`.
  - `test_labs.py` runs the built notebook, so rebuild first. Run `harvest_figures.py` only after a
    full run.
  - Never build a week while it is being rendered or tested.
  - Set `PYTHONIOENCODING=utf-8`.
- **Saving tokens:**
  - Read only what your task needs; use grep, line ranges and short outputs.
  - Review images as contact sheets first, then open single images only where something looks wrong.
  - Write findings as table rows, not essays.

## 7. Definition of done

**Every check passes, run by you and seen to pass:**
- `build_week.py --weeks 1-12` and `build_course.py`
- `verify_beginner.py`, `test_lab_helpers.py` and `test_technical_figures.py`
- `node tools/test_technical_diagrams.js` and `node tools/test_teaching_notes.js`
- `audit_prose.py`, `audit_inline_math.py`, `audit_slide_text.py` and `audit_flat_bullets.py`
- `audit_coverage.py` (100%) and `check_links.py`
- smoke runs of all 12 labs, plus full CPU runs wherever lab logic changed

**After a full render, you have looked at:**
- every slide, Word page, diagram image and notebook image;
- every `slides.layout.json`, which is `[]`;
- the findings, with zero open rows.

**A beginner can follow every week alone:**
- every speaker note is in the template, and no term is used before it is defined;
- every core concept has a diagram, and every mechanism has a worked example with numbers;
- every lab runs on a CPU in Colab within budget, with checks after each TODO and a working Gemini
  switch with a fallback.

**Honest records:**
- `research/REVISION_2026-10.md` lists what changed in each week, the new diagrams and the sources used.
- It also lists what was not verified: no timed Colab run, Gemini not called live, blocked links, and
  labs not run in full mode.

## 8. How to report back

After each task, reply in at most 6 lines:
- what changed (files);
- the checks you ran, with their results;
- what is not verified;
- what the teacher should look at.

Ask the teacher only when a decision is theirs to make. Otherwise use the defaults in TASKS.md (D1–D5).
