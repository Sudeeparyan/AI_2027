# Generative AI 2027 teaching pack — release check

Date: 28 September 2026. Scope: the supplied 12-week module sequence, with two concept hours and two guided-practice hours each week.

## Delivered

- 12 classroom PowerPoint decks, 40 slides apiece; 12 editable PowerPoint source decks with native shapes and scripts in presenter notes.
- 12 Word teaching/study guides with four everyday analogies, their exact mappings and limits, worked micro-examples, formative questions and answers, exam practice, references and optional reading/video links.
- 12 executable Python notebooks with the weekly case, prediction/change task and transfer reflection. The demonstrations use small prepared or synthetic data and label simplified stand-ins.
- 480 currently referenced PNG and SVG diagram exports, indexed by stable slide IDs and content hashes. Older asset versions remain in the asset folder for revision history.
- Python sources for authored lesson data, slide compilation, rendering, handouts, notebook assembly, provider switching, checks, gallery and packaging.

## Checks run

- `python course.py improve --weeks all --max-passes 3 --force`: all 12 weeks passed slide count, notes length, shape bounds, word wrap and notebook execution checks.
- `python verify_release.py`: 12/12 weekly releases passed, including slide/source counts, embedded image hashes, PDF page bounds, guide sections, unique notebook cell IDs and zero unresolved build issues.
- `python -m unittest discover -s tests -v`: 12/12 regression tests passed, including failed-build release preservation, authoring resume, provider fallback and shared-code protection.
- Visual inspection: openings and concept examples from weeks 1, 2, 3, 9, 10, 11 and 12; Word handout pages and exam rehearsal. The release check is structural; a subject teacher should still review educational accuracy and suitability for their cohort.

## Revision route

Edit a single `content/week_XX/slides.json` record for a local improvement. For a change to the full teaching story, edit `content/teaching_design/author_lessons.py`, run it and then `build_lessons.py`. Rebuild one week or all weeks with `course.py improve`, inspect `outputs/current/index.html`, and record human feedback in `reviews/`. The compiler intentionally replaces canonical slide records only when explicitly run.

Official reading/video links were checked on 28 September 2026 and should be rechecked before a 2027 delivery. Videos are optional supporting material; no full transcript was reviewed. The ten final RAG evaluation cases remain in `instructor_only/` and should not be given to learners before the final check.
