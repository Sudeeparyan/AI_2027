# Independent student comprehension review — 14 October 2026

The pack has a strong teaching structure, but the previous completion report did not establish that every explanation was accurate or every preview current. This review traces the source-to-deliverable pipeline, reads all twelve weeks and checks the generated material from the perspective of graduates new to AI and Python. Confirmed problems are corrected in sources and regenerated. Classroom comprehension remains a separate test.

## Scope and constraints

- Preserve the binding `Descriptor - Generative AI-old.docx`, the approved week order, outcomes and assessment. The separate `01_Generative_AI_2027.docx` contributes only the teaching ideas permitted by TASKS.md.
- Sources and existing render baselines are retained in `build/student_review_20261014/`. Existing uncommitted work is preserved; no commit or push.
- Shared engine review covers loading/enrichment, source-to-slide numbering, Office layouts, notebook student/solution separation, shared opt-in setup, image reuse, measured-result filling and execution/render evidence.
- Week reviews: [01–04](weeks/q5_w01_04.md), [05–08](weeks/q5_w05_08.md), [09–12](weeks/q5_w09_12.md). These contain specific locations, fixes and actual visual-review manifests.

## Shared findings

| ID | Place | Problem | Correction | Status |
|---|---|---|---|---|
| Q5-E01 | Course guide preview | Distributed guide contains the newer W08 CPU time; retained render input and PNGs still show the earlier time. | Re-render course documents; add `verify_render_inputs.py` and regression fixtures. It compares package content, retaining explanations/images while ignoring timestamps and random relationship IDs. Initial probe: 28/29 inputs match; guide fails. | Source guard fixed; final render/check pending |
| Q5-E02 | Source catalogue | An independent Future of Life Institute resource was labelled “official text”. | Identify its publisher and independent status; retain European Commission primary article links. | Source fixed; rebuilt reading pages pending |
| Q5-E03 | Codebase guide | Legend documentation predates the existing merged Loss / Check key. | Describe actual legend text, preserve the loss-versus-evaluation semantic distinction, and document the preview-provenance gate. | Fixed |
| Q5-E04 | Projected text across weeks | A preliminary count flags 29 ordinary slide bodies above 40 words; including native tables exposes three additional cases. Repeated outcomes/lab labels contribute much of the excess. | Shorten shared projected summaries, remove the repeated duration card, trim affected examples and add `audit_body_words.py` as a release gate. Count native tables in both word/font audits, with three regression tests. Full outcomes, two-hour schedule and explanations remain in course/Word notes. | Source fixes in progress; final built audit pending |

## Current verification

- Initial engine run: 14 command groups passed; logs in `build/student_review_20261014/checks_engine/`.
- New render-input guard: 3 regression tests passed, including changed explanation/image/relationship detection and ignored volatile IDs.
- Weeks 05–12 rebuilt successfully at 46 slides each. Remaining builds, final checks, twelve CPU smokes, rendering and replacement image review are in progress.
- No executable lab algorithm or measured asset has been changed in this review. Confirm final full-run code parity before relying on retained CPU results.

## Source checks

Checked official pages on 14 October: [Gemini helper model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [EC Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) and [EC Article 111](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-111). Existing helper model choices and the dated transition explanation are supported. The [independent guide](https://artificialintelligenceact.eu/) identifies the Future of Life Institute as its publisher. Documentation support does not verify access in a student's account or a live provider request.

## Practical limits

No timed Colab session, live model-provider request, new GPU/stretch run or classroom beginner test is established by this review. Retain the documented widget/frontend and publisher-blocked-link limits. The original descriptor's comments and revisions outside Section 7.3 remain preserved as required. Before teaching, use a small beginner pilot: ask students to trace a diagram, predict one result, explain a symbol and complete one TODO without the solution notebook.
