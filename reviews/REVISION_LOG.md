# Revision log

## Structured project revision

- Moved weekly content and diagrams into editable JSON files.
- Added stable slide IDs, separate notebook sources and reusable renderers.
- Generated a new course illustration and retained its generation prompt.
- Added rendered PNG/SVG diagram assets that the classroom slides actually reuse.
- Added source snapshots, image hashes, dated build reports and current outputs.
- Added bounded automatic layout repairs and a local watch command.
- Human review remains explicit: checks do not automatically rewrite or approve
  teaching claims.

## Review and rebuild

The initial week 2 run flagged a long-word wrap on slides s05 and s09. Changed the terminal VAE block from Reconstruction to Output image, retaining the full term in the explanation. Added explicit Incorrect claim and Correction labels to failure slides to prevent a beginner mistaking a false claim for the teaching point.

## Teaching-content improvement

Week 2 s05 now explains μ as centre, σ as spread and z as a sampled code. The handout and slide script were updated together. The script also clarifies that the lab does not train a VAE. This semantic improvement came from human review, not from the layout checker.

## Syllabus and research improvement pass

Added 36 original exam-practice questions with model answers, official reading/video references, meaningful weekly metric tables, a supplied VAE loss chart, image-output comparisons and masking practice. Replaced hardcoded week 11 checks with actual invalid-input, missing-source and access tests. Added shared Python application and separate teacher-held final cases. Videos were selected from official/indexed metadata; full video transcript review is not claimed.

## Final delivery verification — 28 September 2026

- Latest full build passed automatic checks for all 12 weeks; the final week 2 label-position improvement also passed.
- Verified 12 classroom decks, 28 slides per deck (336 total), speaker notes, and one embedded saved visual per slide.
- All 12 offline notebooks executed during the build. Word guides and decks rendered to PDF and preview images.
- Visually inspected representative slides for VAE loss, attention, RAG evaluation and delivery costs, plus the five-page week 2 guide. Moved training-curve labels below the line to separate them from validation labels.
- Educational review rows remain pending unless individually reviewed; automated checks are not academic approval.
- Official-source links and selected video descriptions were researched; full video transcripts were not reviewed.

## Final QA — 28 September 2026

See FINAL_QA_REPORT.md for corrected release checks, source references, invoice consistency, diagrams and document pagination. All 12 weeks rebuilt, 12 tests passed and the portable archive passed a fresh-extraction check.
