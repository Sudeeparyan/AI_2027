# How to review and improve a week

This is the working routine used for the October 2026 revision. Follow it when you (a person, Claude or
Codex) improve the pack again, so each pass finds the same kinds of mistakes and leaves the same
evidence behind. The plan and its rules are in `improvements.md`; the open and closed findings are in
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
   # full run on the CPU path (hides the GPU)
   $env:CUDA_VISIBLE_DEVICES=""; $env:HF_HUB_OFFLINE="1"; .venv-labs/Scripts/python.exe tools/test_labs.py --weeks N --full
   # full run on the GTX 1650, exporting results for the slides; raise the per-cell limit for long cells
   $env:GENAI_RESULTS_PATH="<scratch>/results.json"; .venv-gpu/Scripts/python.exe tools/test_labs.py --weeks N --full --timeout 7200
   ```

   `test_labs.py` executes the built notebook in `deliverables/`, so run `build_week.py --weeks N` after
   any lab edit and before the run. On the 4 GB GTX 1650, Week 8 (fp32 LoRA) and Week 10 (Qwen3-VL-2B,
   set `GENAI_VLM`) must run on the CPU. Compare a new export with the stored one before replacing it,
   then run `.venv/Scripts/python.exe tools/fill_results.py --weeks N`. Run `tools/harvest_figures.py`
   only after a full run, never after a smoke run.

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
| Speaker-note boilerplate pasted on many slides | the same Ask/Analogy/Common mistake on 8 slides | `audit_prose.py` (REPEAT) |
| An answer labelled as a mistake | "Ask: …? Common mistake: No." | `audit_prose.py` (LABEL) |
| Every Ask has an Answer that fits the slide | an Ask about RoPE on a slide about decoding | reading |
| Counts in the notes match the slide | "Recap the eight points" on a six-point slide | reading |
| Notes describe what the figure really shows | notes describe an invalid-answer chart the figure leaves out | reading |
| Negative numbers and ranges | `-0.5` → −0.5, `5-8` → 5–8 | `audit_prose.py` (MINUS, RANGE) |
| Maths typed as words or ASCII | `epsilon_theta`, `abar = 0.64`, `x0` | `audit_inline_math.py` |
| Sentences that start with a lower-case code name | ". avg_nll measures" → ". The avg_nll function measures" | `audit_prose.py` (LOWER) |
| Unmeasured runtimes and sizes | "Downloads ≈ 3 GB" (it was 4.5 GB) | reading + lab run |
| Claims that only hold on one path | Flan-T5-small on the CPU path answers "ireland" | executed notebook |
| Picture text smaller than 14 pt when projected | chart tick labels | `audit_figure_text.py` |
| Legends over data or axis labels | figure-level legends | `lint_figures.py` + eye |
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
| A line that ends with a bare option label | "… (c)" with "[4, 4]" on the next line | `tools/notes.js` joins the label to its option; `pdftotext -enc UTF-8 -layout build/render/week_NN/notes.pdf - \| grep -E "\((a\|b\|c\|d)\)\s*$"` finds any left |

## Helpers

- `tools/yaml_edit.py` changes chosen keys of chosen slides and leaves the rest of the YAML file
  byte for byte, then checks that nothing else moved:

  ```python
  import sys; sys.path.insert(0, "tools")
  from yaml_edit import edit_slides
  edit_slides(7, {16: {"notes": "Say: ..."}})
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
  notebook. If it reports a code change, re-run the lab before trusting its stored results or time. (Week 1
  reports two metadata fields added to the results export after its 9 October run; the run record explains.)
- Notebooks are byte-identical when their sources have not changed, so a hash shows whether a smoke or full
  run still applies. PowerPoint and Word files are not: each build writes a new timestamp, and Word
  hyperlinks get random relationship ids. Compare their contents, not their hashes.
- Swapping quiz options: move the correct option and one distractor only, then rewrite every letter the
  explanation names by hand. A blind letter replacement is unsafe ("A bigger model" uses A as a word).

## Status (10 October 2026)

Every week has been through the loop once. The evidence for the final build is in the table at the end of
`research/REVISION_2026-10.md`.

| Week | Text audits | Evidence checked | Slides viewed | Word pages viewed | Audit section |
|---|---|---|---|---|---|
| 1–12 | pass (0 problems) | full run 9 Oct; smoke run 10 Oct on the delivered notebooks | yes | yes | yes, no open rows |

Not yet checked, and worth doing in the next pass: a timed run on a Colab T4, a clean installation with
downloads, and classroom feedback from the exit questions.
