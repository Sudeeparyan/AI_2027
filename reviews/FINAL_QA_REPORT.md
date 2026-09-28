# Final QA report — 28 September 2026

Status: no unresolved failures in the checks performed. This is a tested classroom release, not a claim that every future environment or provider will behave identically.

## Verified

- 12 application tests passed, including quota fallback, malformed output, resume after a model change, stale-source rejection, backups, concurrent-run locking, local HTTP transport, failed-build release protection and notebook reference validation.
- Rebuilt all 12 weeks after corrections. Every notebook executed successfully.
- 12 classroom decks, 28 slides each: 336 slides with speaker notes.
- Each classroom slide embeds exactly the corresponding saved PNG; 336 asset hashes and SVG companions checked.
- 12 Word study guides, 12 notebooks, editable decks, PDF previews and per-week reports present.
- All Word guides and slide decks rendered. Inspected all slide contact sheets and all handout page montages; opened corrected slides at larger size. No remaining visible clipping identified in that review.
- Exam-answer sections, resources, notebook cell IDs, source titles and PDF page bounds checked.
- Exported ZIP integrity passed. A fresh extraction passed the 12 tests, a 60-job offline authoring dry run and all 12 weekly structural checks. This reused the installed Python dependencies; it was not a fresh operating-system installation.

## Fixed during this review

1. Builds with unresolved automatic checks now fail before replacing the current weekly release.
2. Each successfully built week is checkpointed immediately.
3. Shared notebook code references are restricted to content/shared, and AI candidates must preserve those references.
4. Replaced a missing curriculum filename with the retained source path.
5. Removed an unsupported assessment-conflict statement; the supplied original descriptor specifies project 60%, examination 40%.
6. Corrected the week 5 cover invoice to match its worked example.
7. Corrected VAE and noising change diagrams, and shortened several wrapped diagram labels without removing their definitions.
8. Kept reference paragraphs together across page breaks and removed inherited title rules in Word guides.
9. Added verify_release.py for repeatable post-build verification.

## Remaining verification boundaries

Live cloud API calls require valid user credentials and were not run. Provider failures and switching were tested using mocks and a local HTTP server. External web links were not exhaustively rechecked during this final offline QA pass. Real PowerPoint/Jupyter behavior should be piloted in the institution's environment. The core labs intentionally use labelled simulations and prepared data; this review does not turn them into trained models or production services. Formal academic approval and student learning effectiveness require instructor review.

## Recheck after future edits

```bash
python -m unittest discover -s tests -v
python course.py improve --weeks all --max-passes 3
python verify_release.py
```

Evidence: FINAL_STRUCTURAL_CHECK.json, FRESH_EXTRACTION_TEST.json and the latest dated build reports.
