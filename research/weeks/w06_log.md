# Week 06 log

## V · Visual QA and activation (Claude, 11 Oct 2026)

| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 06 | labs/week_06_lab.py | Part 1 | transformers 5 apply_chat_template returns BatchEncoding: KeyError 'shape' | High | return_dict=True and ['input_ids']; dtype= | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 06 | weeks/week_06.yaml | 24 | Duplicate 'callout' key silently broke edit_slides | Medium | Removed duplicate (parsed content unchanged) | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 06 | weeks/week_06.yaml, figures | 8, 26, 30, 43 | Native figure text pushed slide body over 40 words; reasoning figure arrows unlabelled | Medium | Shortened figure text; labelled arrows (prompt IDs, answer, trace + answer); caption removed | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 06 | beginner/week_06.json | 7, 43 | Code labels exceeded blocks; check answer mismatched | Medium | Removed non-owner code labels; fixed answer | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 06 | notes/week_06.md | Word p18 | 'about one twentieth of the 7 B example' is wrong (8.4e22 / 1.6e21 ≈ 52) | Medium | 'about one fiftieth' with the division shown | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 06 | notes/week_06.md | Word p17 | Cumulative top-p mass 0.77 (sum of rounded values) contradicted 0.76 on p13 | Low | 0.76 from unrounded probabilities | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 06 | weeks/week_06.yaml | 30 | Notes 206 words; example_strip used 'value' instead of 'text' | Low | Trimmed; renamed key | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

## Final coordinated closure — 12 October 2026
- Final current coverage: 46 slides, 30 Word pages, 9 delivered diagram PNGs and 7 notebook images; actual views plus byte-identical previously viewed images are documented in the final review manifests. Layout report [], zero open findings.
- Latest successful cached local CPU full core: 33.0 s (0.6 min), recorded in `research/lab_runs_2026-10.json`; final smoke PASS (36.0 s). Current executed core code matches the successful full-run evidence.
- All 26 current release gates passed, including coverage 100%, strict notes, body ≤ 40, fonts ≥ 14, semantic helpers and shared diagram/attachment parity. See `research/ITERATION3_2026-10-12.json`.
- No timed Colab, clean-install/download, live-provider, browser-widget or optional GPU/stretch claim. Earlier pending handoffs above describe intermediate states.
