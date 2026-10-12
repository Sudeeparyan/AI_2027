# Task board: iteration 3 (started 10 Oct 2026)

**How to use.** Take the first task whose status is `todo`, whose needs are `done` and that no session is
doing, unless the teacher named one. Set it to `doing (tool, date)`. Work to its "Done when" list and
update the handoff line after each sub-step. Finish with `done` or `blocked: reason`.
∥ = can run at the same time as other ∥ tasks. Rules and standards: `AGENTS.md`.

**Teacher override (2026-10-10):** "complete everything dont wait for my command".
Continue across tasks and past the Week 1 pilot without approval stops. Preserve the uncommitted
baseline in `build/source_before_iter3/` and the existing renders before rebuilding; do not commit
or push. All source ownership, measured-evidence and verification rules still apply.
**API override (2026-10-11):** the teacher supplied a private Google key in `.env`, to use only if
required, and permits students to choose Gemini or the OpenAI API. Local completion stays free.
Keys are never embedded or printed; OpenAI charges and separate API billing must be explicit.

## Starting facts (checked 10 Oct 2026; don't re-derive)
- Pipeline works. 46 slides per week; 5 block diagrams, a course map and a recap per week.
  All audits pass; coverage 214/214; links 178/182 (4 publisher pages blocked).
  155 files uncommitted on branch `version1`.
- Speaker notes are about 80-word teacher cues ("Say: … Analogy: … Ask: … Answer: … Common mistake: …").
  They are too terse and jargon-heavy for a student who missed class (example below).
- Some bullet slides pair 4 short bullets with a decorative icon, e.g. W05 "From text to vectors".
  Diagram code labels repeat: W05 "Inside one attention head" shows `attention` under four blocks.
- Labs: only W01, W02, W05 and W06 have any self-check asserts. No lab uses widgets. Gemini is opt-in in
  W01, W06 and W11 only (`RUN_GEMINI`/`RUN_CLOUD`, `gemini-flash-latest`). The pack ships `.py` exports too.
- Recorded full-run minutes (one laptop): W1 0.9 · W2 9.3 · W3 19.2 · W4 16.6 · W5 10.1 GPU · W6 8.4 ·
  W7 58.3 GPU · W8 84.1 CPU · W9 0.8 GPU · W10 6.9 · W11 3.3 GPU · W12 4.6 GPU.
  W3, W4, W7 and W8 are over budget; weeks timed only on the GPU need a CPU time. No Colab run has been timed.
- Stale names are likely, e.g. "Gemini 1.5" and many GPT-3 mentions. Verify every name against R0–R4.

## Teacher decisions (an agent treats the default as agreed until the teacher edits it)
| # | Question | Default |
|---|---|---|
| D1 | Commit the current tree as a baseline before parallel work? | Yes. The teacher commits; agents never do. |
| D2 | Pilot week, reviewed before rolling out | Week 1 (it sets up Colab, Gemini and Python habits) |
| D3 | Ship `.py` lab exports? | No. Notebooks only in `deliverables/`; keep `.py` under `build/` for diffs |
| D4 | Gemini in LLM-heavy weeks (6, 7, 10, 11, 12) | Optional switch, recommended in the lab text; the local CPU path always works |
| D5 | Extra "slides with notes" PDF for absent students | Not now |
| D6 | Optional hosted model access | Exactly one Gemini/OpenAI switch; both off by default; complete local fallback |

## Borrow from `context/01_Generative_AI_2027.docx` (teaching style only)
- Pattern: familiar analogy → real components → one success and one failure → practise → measure and
  explain. Keep the analogy separate from the technical claim.
- 120-min lab: recall and vocabulary 10 · teacher demo (one success, one common error) 20 · guided
  pairs 40 · individual change 25 · compare and review 15 · exit check 10.
- Every lab gives starter code, the expected output, a worked example and a short checklist.
  Week 1 includes a readiness check (Python, notebooks, arrays, probability).
- Practice and exam formats: label a block diagram; explain why an answer lacks evidence; read a
  training curve; decide between a document update and fine-tuning. Simple calculations, no long derivations.
- RAG and agent metrics: recall@k, citation support, task success, cost per successful task.
- Further reading from official learning paths: Google (Gen AI beginner guide, ML Crash Course),
  OpenAI Academy, Anthropic Academy and "Building effective agents", NVIDIA, Microsoft Learn.
- Disclosure of AI-assisted work: tools used, prompts, checks, and the student's own explanation.

## Speaker-note example (W05 "Positional encoding")
Before: "Say: Plain unrestricted self-attention has no built-in coordinates. Shuffling input rows shuffles
output rows in the same way. …"

After:
> **In short:** Attention alone ignores word order, so we add position information.
> **Explain:** Attention compares every token with every other token. On its own, it treats the input as
> a bag of tokens, so "dog bites man" and "man bites dog" look alike. We therefore give each position its
> own signal. The original transformer adds a fixed wave pattern (sinusoidal encoding) to each embedding.
> Our lab learns one vector per position instead. Most modern LLMs use rotary encoding (RoPE): it rotates
> each query and key vector by an angle that depends on its position.
> **Example:** Seats in a cinema. The same person in seat 3 or seat 7 is a different situation. Where it
> breaks: with RoPE, what matters is how far apart two tokens are.
> **Ask:** Is RoPE added to the input embeddings? **Answer:** No. It rotates queries and keys inside
> attention.
> **Watch out:** A longer context setting only works if the model was trained or adapted for it.

## Phase 0: baseline (sequential)
**B0 · Baseline check** · owns: `build/`, creates `research/weeks/` · needs: D1 (local snapshot authorised by teacher override) · status: done (Codex, 2026-10-10)
Done when:
- `build/render/` is copied to `build/render_before_iter3/` first, so the "before" images survive.
- `build_week.py --weeks 1-12` and `build_course.py` succeed.
- Every check in CODEBASE_GUIDE "Build and check" has run, except labs and render. Pass or fail is
  recorded here, one line per check.
Handoff: Source and existing render snapshots preserved in build/source_before_iter3 and build/render_before_iter3. 22/23 requested baseline gates passed; logs/results in build/baseline_iter3. Link check had three temporary GitHub 503 failures and four publisher 403 blocks; retry/fix in M1. Labs/render excluded as specified.
- build_week.py --weeks 1-12: PASS.
- build_course.py: PASS.
- verify_beginner.py: PASS.
- test_lab_helpers.py: PASS.
- test_lab_runtime.py: PASS.
- test_lab_evidence.py: PASS.
- test_link_checks.py: PASS.
- test_output_files.py: PASS.
- node test_output_files.js: PASS.
- test_beginner_support.py: PASS.
- test_technical_figures.py: PASS.
- node test_teaching_notes.js: PASS.
- node test_technical_diagrams.js: PASS.
- audit_inline_math.py: PASS.
- audit_coverage.py: PASS (100%).
- audit_slide_text.py: PASS.
- audit_prose.py: PASS.
- audit_flat_bullets.py: PASS.
- lint_figures.py --weeks 1-12: PASS.
- audit_figure_text.py --weeks 1-12: PASS.
- verify_pilot_artifacts.py: PASS.
- check_links.py: FAIL (175/182 reachable; 4 blocked; 3 HTTP 503).
- build_revision_review.py: PASS.

## Phase 1: research (∥ all; read-only except your own output file; may start before B0)
**R0 · Platform facts** · owns: `research/weeks/platform_2026.md` · status: done (Codex, 2026-10-10)
Answer from official pages, with the date checked:
- Colab free tier: CPU and GPU limits, and preinstalled versions of torch, transformers, diffusers,
  peft, trl, sentence-transformers and ipywidgets.
- Gemini API free tier: current model IDs, rate limits, region availability (including the EU and UK),
  data-use terms, the current `google-genai` call pattern, and Colab Secrets.
- ipywidgets in Colab and in VS Code.
Handoff: Complete: platform_2026.md has dated official sources for every R0 topic. PASS: 28 package entries matched against four pinned official CPU/GPU freezes; two Python examples parse; independent source review found no blocking issue. Embedding-ID source discrepancy recorded. Unverified: live Colab packages/timing, Gemini quotas/access and widget rendering. E2 needs regional free-API distribution clarification; D1 teacher baseline commit is outstanding. Stopped after R0.

**R1 W1–3 · R2 W4–6 · R3 W7–9 · R4 W10–12** · owns: `research/weeks/wNN_research.md` · status: R1/R2/R3/R4 done (Codex agents, 2026-10-10)
Per week, at most one page:
1. 2–3 explanations or analogies the best teachers use (source).
2. 5 common student misconceptions.
3. What is current in 2026 (model families, APIs, library versions, practice), with official links.
4. Diagram or infographic ideas to redraw in our own style. Never copy images.
5. What is too hard for AI and Python beginners. Read only the slide titles in `build/week_NN/spec.json`
   and the lab headings.
6. A CPU-only core lab path of ≤ 15 min, and where the Gemini free tier helps.
7. 3–5 further-reading links, including at least 1 video.

Preferred sources:
- Original papers; Hugging Face courses and docs; Google ML Crash Course and Gemini docs; Microsoft Learn;
  OpenAI and Anthropic docs.
- Stanford CS236/CS324, MIT 6.S191, Jay Alammar, Lilian Weng, 3Blue1Brown, Karpathy, distill.pub,
  Chip Huyen.
Handoff R1: W01–03 files written; sections/video/source checks passed. · R2: W04–06 files written; sections/video/source checks passed. · R3: W07–09 files written; sections/video/source checks passed. · R4: W10–12 files written; seven-section, source/video and size checks passed. All proposed CPU times remain unmeasured.

## Phase 2: engine (sequential, one at a time; these own `tools/`)
**E1 · Speaker-note template** · owns: notes checks in `tools/build_week.py`, `tools/audit_prose.py`, a new test
· needs: B0 · status: done (Codex agent, 2026-10-10)
Done when:
- The build validates the AGENTS.md template labels and 110–180 words. Old-format notes give a warning,
  not an error, until M3.
- REPEAT, LABEL and ANSWER understand `Watch out:`, and the quiz "Answer: X" letter rule still works.
- The new test passes, and all 12 weeks still build.
Handoff: Shared validator and audit updated; 14 regression tests and prose audit passed. Root all-week build passed for all 12 decks; existing minimum safeguards preserved, one legacy warning per week until M3.

**E2 · Lab standard** · owns: the lab section of `tools/build_week.py`, `tools/beginner.py` (python_export),
`tools/verify_beginner.py`, `tools/test_labs.py`, new `curriculum/labs/_setup.py`, lab tests, lab rows
in README/CODEBASE_GUIDE · needs: E1, R0 · status: done (Codex, 2026-10-11)
Done when:
- The builder injects one shared setup cell from `_setup.py` into every notebook. That cell:
  - detects Colab or local, uses the CPU unless the student opts into the GPU, sets seeds and prints versions;
  - provides the `USE_GEMINI` switch and `ask_gemini()`: the model ID is one constant taken from R0; the
    key comes from Secrets or env and is never printed; it backs off on 429, caches on disk, and returns
    `None` so the lab falls back to the local path;
  - provides `check(condition, ok_message, hint)`.
- No `.py` exports in `deliverables/` (D3), and the checks are updated to match.
- Install cells are tagged `colab-install` and use the versions R0 found.
- Stubbed-client tests pass: switch off → no call; no key → fallback; 429 → retry, then fallback.
- All 12 smoke runs pass.
Handoff: Shared CPU setup/API/cache/check helper injected once; dated install pins and notebook-only outputs implemented. All 12 CPU smoke runs passed (build/e2_smoke.log). Final insertion/pinning/environment/retry corrections rebuilt in all notebooks; verify_beginner passed 12/12 on 11 Oct, and 14 current setup regressions passed. Local SDK/config/widget construction passed. No live Gemini or Colab run; final all-week executions repeat after week rewrites in M3.

**E2p · Optional provider choice** · owns: shared setup, API setup tests, builder install/banner text, API policy docs · status: done (root, 11 Oct)
Handoff: Shared Gemini/OpenAI choice, private local `.env` selection, combined ceiling/cache/retries and local fallback implemented. Root observed PASS: 22 setup tests,25 evidence tests, actual OpenAI3.28 constructor/request arguments, W01 rebuild/fullCPU/smoke/eight audits. Policy and pilot Word/slide text aligned. Private `.env` remains ignored/untracked/unread; no live API call. All-week final runs follow rewrites in M3.

**E3 · Slide visuals and cheaper review** · owns: `tools/slides.js`, `tools/technical_diagrams.js`, `tools/teaching_support.js`,
the diagram parts of `tools/beginner.py`, `tools/notes_sheets.py`, their tests · needs: E2 · status: done (Codex agent, 2026-10-11)
Done when:
- New layout: ≤ 3 short bullets beside an informative visual (example strip, mini-table or small
  figure) instead of an icon.
- Predict-then-reveal quiz: the slide shows the question and options; the answer is only in the notes.
- A diagram block shows a lab symbol only when it names that block's own function or variable.
- `notes_sheets.py --slides --weeks N` tiles a week's slides 6 per sheet.
- `node tools/test_technical_diagrams.js`, `test_technical_figures.py` and
  `node tools/test_teaching_notes.js` pass.
Handoff: Editable example strips/mini-tables, hidden quiz answers, explicit code_owner, compact legends/map, authored modern notes, short vocabulary and six-slide sheets implemented. Root ran and saw PASS: test_slide_visuals.js, test_visual_support.py (5), test_beginner_support.py (4), test_technical_figures.py (10), test_technical_diagrams.js and test_teaching_notes.js. Rebuild/render of upgraded weeks follows.

**E3f · Complete flow parity** · owns: flow parts of `tools/build_week.py`, `tools/slides.js`, `tools/technical_diagrams.js`, `tools/verify_beginner.py` and their tests · status: done (root/agent, 11 Oct)
Pilot visual review found native flow slides still had unlabelled arrows and lacked matching exports. Adapt explicitly authored roles/arrows to one shared technical scene; export PNG/SVG and use the same PNG in Word/notebooks and editable scene in PPT. Root owns file-table correction in notes.js and W01 flow content. Legacy flows remain until their week author supplies truthful labels.
Handoff: One strict annotated flow scene supplies editable PPT, SVG/PNG, Word and notebook images/portable links. Default decorative card badges removed. Named teaching figures now embed through the same notebook path, including cached labs-only builds; generated ownership manifests clean stale exports. Root observed PASS: seven flow tests, technical diagrams, slide visuals, teaching notes, 5 visual/4 beginner support tests and 6 verifier self-tests; W01 rebuild/visual review passed. Legacy flows require author labels during their week upgrade; named-figure support awaits the W04 rebuild.

## Phase 3: weeks (pilot W01 first; teacher override permits continuing. Afterwards ≤ 3 weeks at once)
Each week has three tasks. L ∥ B; V runs after both. Needs: E1–E3, plus R0 and that week's R note.

**L · Lecture** · owns: `weeks/week_NN.yaml`, `beginner/week_NN.json`, `notes/week_NN.md`,
`figures/week_NN.py`, `research/weeks/wNN_log.md`. Done when:
- Every speaker note follows the template. Body text ≤ 40 words per slide; sentences ≤ 25 words.
- Every term is defined at first use or on the vocabulary slide. New glossary terms and sources are
  proposed in the log.
- Every mechanism has a worked example with numbers or shapes, and every concept section has a diagram.
- Every icon-plus-bullets slide uses the E3 visual layout or is merged.
- The deck (~46 slides) has: a hook, plain outcomes, the course map, vocabulary, concept sections, a
  worked example, a lab preview, 3–5 check questions with answers in the notes, further reading with
  ≥ 1 video, and the recap.
- Model and product names match the R notes. The Word notes have a misconceptions/FAQ section and
  self-checks with answers, and their figures match the slides.
- `build_week --weeks N` and the five text audits pass.

**B · Lab** · owns: `labs/week_NN_lab.py`, `assets/make_week_NN*.py`, `research/weeks/wNN_lab_log.md`. Done when:
- The AGENTS.md lab standard is applied, with duplicate setup code removed in favour of `_setup.py`.
  The lab has at least one widget, and a Gemini path that falls back cleanly.
- The CPU core path is measured at ≤ 15 min (full CPU run, using `build/.busy`), and the run record is
  proposed in the lab log. If over budget, cut it or explain why.
- Rebuilt with `--labs-only`; the smoke test passes; a full run was done if logic changed.
  Any changed slide numbers go through `fill_results`, never by hand.
- W08 still runs a real (small) fine-tuning job, as LO3 requires.

**V · Visual QA** · owns: all of week N's files (after L and B are done). Done when:
- The week has been rendered and `build/render/week_NN/slides.layout.json` is `[]`.
- You viewed the slide sheets, every Word page sheet, every `Diagrams/*.png` and the notebook images.
  Check for:
  - arrows that miss blocks, cross or point the wrong way;
  - overlapping or clipped text, small fonts, empty or misaligned blocks, missing legends;
  - PPT, SVG and PNG copies that differ, and raw LaTeX;
  - broken images, and wrong data flow or tensor shapes;
  - the "What to look for" table in REVIEW_PROCESS.md.
- Every finding is a row in the log (week | file | slide/page | problem | severity | fix | status),
  fixed, re-rendered and viewed again. Zero rows are open.

| Week | L | B | V | Handoff |
|---|---|---|---|---|
| 01 pilot | done | done | done (12 Oct) | 46 slides; 33 Word pages; 9 diagrams; 11 notebook images covered. Layout []; zero open findings. CPU core 44.6 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 02 | done | done | done (12 Oct) | 46 slides; 33 Word pages; 9 diagrams; 15 notebook images covered. Layout []; zero open findings. CPU core 23.8 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 03 | done | done | done (12 Oct) | 46 slides; 30 Word pages; 7 diagrams; 17 notebook images covered. Layout []; zero open findings. CPU core 279.3 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 04 | done | done | done (12 Oct) | 46 slides; 33 Word pages; 11 diagrams; 13 notebook images covered. Layout []; zero open findings. CPU core 39.4 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 05 | done | done | done (12 Oct) | 46 slides; 30 Word pages; 7 diagrams; 8 notebook images covered. Layout []; zero open findings. CPU core 248.7 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 06 | done | done | done (12 Oct) | 46 slides; 30 Word pages; 9 diagrams; 7 notebook images covered. Layout []; zero open findings. CPU core 33.0 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 07 | done | done | done (12 Oct) | 46 slides; 32 Word pages; 9 diagrams; 7 notebook images covered. Layout []; zero open findings. CPU core 439.0 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 08 | done | done | done (12 Oct) | 46 slides; 29 Word pages; 8 diagrams; 8 notebook images covered. Layout []; zero open findings. CPU core 797.3 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 09 | done | done | done (12 Oct) | 46 slides; 34 Word pages; 8 diagrams; 12 notebook images covered. Layout []; zero open findings. CPU core 180.8 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 10 | done | done | done (12 Oct) | 46 slides; 40 Word pages; 10 diagrams; 12 notebook images covered. Layout []; zero open findings. CPU core 175.4 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 11 | done | done | done (12 Oct) | 46 slides; 40 Word pages; 8 diagrams; 11 notebook images covered. Layout []; zero open findings. CPU core 39.5 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |
| 12 | done | done | done (12 Oct) | 46 slides; 38 Word pages; 8 diagrams; 9 notebook images covered. Layout []; zero open findings. CPU core 114.0 s; final smoke PASS; 26 gates PASS; coverage 100%. See ITERATION3_2026-10-12.json. |

## Phase 4: merge and finish (sequential, after all weeks)
**M1 · Glossary and sources** · needs: all L · status: done (Codex, 11 Oct)
Merge the proposals from the week logs into `glossary.json` and `sources.yaml`. Then run
`test_beginner_support.py`, `check_links.py` and `build_course.py`.
Handoff: Merged 19 terms (142 canonical total) and 37 named teaching sources; support tests and course build PASS. Links:205/209 reachable,4 publisher403 blocked,0 failures. Week glossary parity restored; compact vocabulary preserved. Final source freeze/build and visual review completed; see M3.

**M2 · Evidence** · needs: all B and V · status: done (Codex, 12 Oct)
- Move the run records into `lab_runs_2026-10.json` and run `fill_results`.
- Rebuild the affected weeks.
- Copy the finding rows into `AUDIT_2026-10.md` under "Iteration 3".
Handoff: All 12 measured CPU cores and their hashes recorded; fill/all-week builds complete. Archive/current results stay separate. Final findings copied to AUDIT Iteration 3; all week/course reviews closed. Current evidence: ITERATION3_2026-10-12.json.

**M3 · Finish** · needs: M1, M2 · status: done (Codex, 12 Oct)
- Make the old note format an error. Run the full gate list, all 12 smoke runs and a full render.
- Update `curriculum/course/instructor_guide.md`, README.md, CODEBASE_GUIDE.md and the REVIEW_PROCESS
  status.
- Append a short "Iteration 3" section to `REVISION_2026-10.md`: changes per week, new diagrams and
  sources used. Add an honest list of what was not verified (e.g. no timed Colab run, Gemini not
  called live, blocked links).
Handoff: Complete 12 Oct: 26 gates PASS; 12 final smokes PASS; full CPU evidence/current-code parity PASS. Reviewed 552 slides, 402 weekly Word pages, 103 diagrams, 130 notebook images and 50 course pages with retained view hashes. All layouts []; zero open findings. Docs/audit/revision updated. Unverified: Colab, live APIs, clean install, widgets, public deployment, optional stretches and four blocked links. No commit/push.

## Intake (any time)
**I0 · Teacher feedback → tasks.** Turn each point in `research/TEACHER_FEEDBACK.md` into a task row
here, with ID, owned files and a "done when". Don't do the work in the same session.
