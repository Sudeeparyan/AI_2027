# w07_log

## Iteration 3 (Claude, 11 Oct 2026)

### L · Lecture
- 40 core notes rewritten to the six-label template (all pass `speaker_notes.validate_speaker_notes` after filling).
- Icon bullets replaced by example strips or mini-tables (slides 7, 14, 19, 21, 30, 32, 37); card and compare icons removed; flows given roles and labelled arrows (22, 27).
- Results slides relabelled as the stored GPU run (Qwen2.5-1.5B, 40 messages); CPU-core results added to notes through new `W07_CORE_*` fill values.
- Beginner support rewritten: three prerequisites, five diagrams with real data flow (the old lab diagram chained independent experiments), compact recap.
- Word notes: lab guide for the CPU core, worked examples (few-shot scoring, CoT parsing, vote by hand, Wilson interval for 16 of 20), FAQ, five self-checks, "Limitations and challenges" heading for coverage.
- Glossary proposal for M1: add "Development set" and "Held-out set" (used in slides, lab and notes).

### V · Visual QA
| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 07 | weeks/week_07.yaml | 26 views | Body text over 40 words | High | Shortened tables, cards, code and quiz; compact beginner labels | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 07 | beginner/week_07.json | 43 (lab) | Arrows implied data flowed between independent experiments | High | Dev/held-out flow with a real branch for the chosen prompt | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 07 | figures/week_07.py | 27 | Context-engineering arrows unlabelled; labels then crossed boxes | Medium | Labelled candidates/kept items/context; widened layout | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 07 | figures/week_07.py | 11 | Anatomy figure text over budget | Low | Shorter part details, title removed | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 07 | notes/week_07.md | Word p2 | "slides 16, 20 and 35" only first number remapped | Low | "slide 16, slide 20 and slide 35" | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 07 | weeks/week_07.yaml, fill | 21, 37, 40 | Example strips and injection bullet wrapped awkwardly | Low | Shorter strip text; INJECTION_SUMMARY shortened | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 07 | notes/week_07.md | Word p2, p11–16 | Stored GPU results described as "from this week's lab" | Medium | Labelled stored run; CPU core values added | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

## Final coordinated closure — 12 October 2026
- Final current coverage: 46 slides, 32 Word pages, 9 delivered diagram PNGs and 7 notebook images; actual views plus byte-identical previously viewed images are documented in the final review manifests. Layout report [], zero open findings.
- Latest successful cached local CPU full core: 439.0 s (7.3 min), recorded in `research/lab_runs_2026-10.json`; final smoke PASS (96.5 s). Current executed core code matches the successful full-run evidence.
- All 26 current release gates passed, including coverage 100%, strict notes, body ≤ 40, fonts ≥ 14, semantic helpers and shared diagram/attachment parity. See `research/ITERATION3_2026-10-12.json`.
- No timed Colab, clean-install/download, live-provider, browser-widget or optional GPU/stretch claim. Earlier pending handoffs above describe intermediate states.
