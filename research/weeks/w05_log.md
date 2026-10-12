

## W05 L candidate handoff — 11 October 2026

- Prepared only under `build/w05_candidate`; five original source SHA-256 hashes guard activation. `apply_candidate.py` defaults to a draft and uses `edit_slides`; `--apply` requires root's activation signal.
- Preserved 40 core slide types, official header/objectives/coverage and equations. Authored 47 core/support notes: six labels, 137–161 words, 3–6 Explain sentences of at most 20 words. Visible predictions/checks: core 4, 15, 22, 27, 37.
- Replaced modified decorative bullet icons with worked example strips; compacted real operation flows, shapes and labels. Code appears only at a named operation with `code_owner:true`; every hidden code remains an empty string for schema compatibility. Target branches bypass representation paths; independent pretrained inspections remain outside training flow.
- Vocabulary canonical meanings preserved; six concise displayed meanings. Single-glossary anchor retained in Word, no new local glossary table. Shared-glossary proposals: verify entries for training target, attention mask conventions and cache; do not duplicate definitions per week.
- Word retains derivations/advanced material, adds numerical attention and multi-head shape walkthroughs, misconceptions, five self-checks with answers, archived/core distinctions and official/video reading. BERT reference matrix remains byte-for-byte function-equivalent; both complete archived GPT-2 matrices retained with numbered ticks and an explicit token-index key.
- New fill adapter proposal **required before build**: `w05_gpt2_future_weight` ← `curriculum/assets/week_05/gpt2_attention.json` → `max_future_weight`. Used in Word instead of a hardcoded archived zero. Existing fill keys and asset files unchanged; figure captions obtain archived values directly from their existing JSON readers.
- Model IDs checked against publisher pages on 11 October 2026: `google-bert/bert-base-uncased`, `openai-community/gpt2`. They are historical optional inspection examples. References: https://huggingface.co/google-bert/bert-base-uncased ; https://huggingface.co/openai-community/gpt2 ; https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html ; https://www.youtube.com/watch?v=eMlx5fFNoYc .
- API help uses shared `ask_model` and the injected Gemini/OpenAI opt-ins. Local critique remains available; no personal input or API calls used during preparation. Retained Gemini regional caveat; paid client distribution in EEA/UK/Switzerland is outside the free-core route.
- Author-only note/schema/AST/diagram/vocabulary/recap/contract/hash checks passed (`precheck.json`); no builds, renders, notebook runs or runtime claims. Actual PPT body/layout and complete CPU timing remain root's post-activation gates.
- Handoff: **candidate ready, not activated**. Root owns board, fill adapter, source activation and machine validation.

## V · Visual QA and activation (Claude, 11 Oct 2026)

| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 05 | labs/week_05_lab.py | lab core | Candidate core ran a tiny model on bundled text; samples were noise | High | Tiny Shakespeare core 3 layers/4 heads/width 128/context 64/2,000 updates; full CPU 248.7 s; ledger + {{W05_CORE_TIME}} | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 05 | figures/week_05.py | 8, 27, 28, 39, 41 | Unlabelled residual start, label overlaps, crossing in three_archs, KV cache had no cache block, vit labels exceeded blocks | Medium | Redrew block, kv_cache, attn_worked; wrapped labels; canvas 6.7 (text ≥14 pt) | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 05 | beginner/week_05.json | 14 (mechanism) | Details repeated code labels ('Scores/scores', 'Weights/weights') | Low | Details name the operation: 'q·k ÷ √2', 'Row softmax' | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 05 | beginner/week_05.json, lab self-check 2 | Word p3; notebook end | Causal-mask answer 'it predicts the following target' did not answer 'is the current position allowed?' | Low | 'Yes; only later positions are blocked' (lab: attends to itself and earlier positions) | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 05 | notes/week_05.md | Word p20, p30 | Two GPU runs looked contradictory; 'Both enabled returns local guidance' unclear | Low | 'A separate, later full run'; both-switch behaviour stated as in _setup.py | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 05 | beginner/week_05.json | lab/inference checks | Check answers did not match the diagram question | Medium | Rewrote answers | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

## Final coordinated closure — 12 October 2026
- Final current coverage: 46 slides, 30 Word pages, 7 delivered diagram PNGs and 8 notebook images; actual views plus byte-identical previously viewed images are documented in the final review manifests. Layout report [], zero open findings.
- Latest successful cached local CPU full core: 248.7 s (4.1 min), recorded in `research/lab_runs_2026-10.json`; final smoke PASS (20.2 s). Current executed core code matches the successful full-run evidence.
- All 26 current release gates passed, including coverage 100%, strict notes, body ≤ 40, fonts ≥ 14, semantic helpers and shared diagram/attachment parity. See `research/ITERATION3_2026-10-12.json`.
- No timed Colab, clean-install/download, live-provider, browser-widget or optional GPU/stretch claim. Earlier pending handoffs above describe intermediate states.
