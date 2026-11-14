# w08_lab_log

## Iteration 3 (Claude, 11 Oct 2026)

- New CPU core: Qwen2.5-0.5B-Instruct, 5 Banking77 intents, 40 / 6 / 10 messages per intent (train / validation / test), rank-8 LoRA on attention projections, one epoch (25 steps).
- TODOs with checks: `to_chat` (`check_to_chat`), LoRA config (`check_lora` recomputes `sum(RANK × (in + out))` from the model), training, comparison (`check_comparison`), adapter size. Rank slider tagged skip-test; optional hosted comparison on five invented messages; `GENAI_RESULTS_PATH` export.
- Fairness fix after the first full run: a 12-token generation cap left 78% of zero-shot replies unparsed, so the baseline measured truncation, not the model. Cap raised to 32 tokens for all three models; re-run.
- Full CPU runs (ledger): 662.2 s with the 12-token cap; 797.3 s (13.3 min) with 32 tokens, within the 15-min budget. The latter feeds `fill.json`.
- Measured core: base zero-shot 42%, base few-shot 86%, LoRA 98% on 50 test messages; 1,081,344 trainable parameters (0.22%); adapter 4.4 MB; training 6.8 min.
- Self-check 2 answer corrected: with the adapter on, the run's free-text reply added confident unsupported details, so the answer now says to check every factual claim rather than predicting a label-style reply.
- Not verified: Colab, live API path, widgets in a browser.

## Final coordinator handoff — 12 October 2026
- Latest measured cached local CPU core 797.3s (13.3min), within 900 s; final smoke PASS (158.1s), zero failed feedback checks. Recorded full-run code matches current executable core.
- Final image review closed; canonical evidence is `research/ITERATION3_2026-10-12.json`. Earlier unmeasured/pending handoffs are historical. No timed Colab, live API, clean-install/download, browser-widget or optional-stretch verification.

## Independent audit repairs — 12 October 2026
- A01/A03: hosted_prompt uses a separate fully invented HOSTED_EXAMPLES list and invented queries; local dataset-backed SHOTS never reaches ask_model. Hosted cap is 16 tokens.
- A02: SFTConfig now receives use_cpu=(DEVICE == "cpu"); model placement is checked against shared DEVICE before training.
- A07: evaluate requires an explicit dataset. Prompt and adapted exploration use one validation item per intent; final Part 5 freezes settings and scores all three models on test_ds. Unchanged reruns reuse predictions; changed frozen settings raise a useful error.
- A07: development subset is deliberately tiny and labelled qualitative; final test remains 50 messages. Split disjointness and comparison checks now validate all scored texts and all metric rows.
- A12: all six parts now have read/predict/core/change/check/stretch/common-error support; Part 4 changes stay on validation and Part 5 changes concern reporting only.
- Static verification: Python AST parses with %pip removed; 4 solution blocks and CRLF preserved. Exports retain the comparison/run schema. Added 15 development generations require a fresh full CPU time; previous runtime is historical.
- Handoff: root should rebuild W08, run privacy/device/frozen-test fixtures, smoke then full CPU with fresh results export, update measured ledger/fill, and inspect notebook output. Colab/live providers/browser widgets remain unverified here.

## Source fallback review — 12 October 2026
- Final-test cache strengthened: adapter_fingerprint hashes only named trainable tensor bytes, shapes and dtypes; training_fingerprint hashes training examples and SFT configuration. The final settings include the actual adapter checksum and training seed.
- The same adapter and training choices reuse saved final predictions; changed weights, messages, seed or configuration reject final-test reuse before any new test request.
- Widget fallback correction: the rank callback and its counts are prepared before the optional ipywidgets import.
- PASS: build/review_lab_fallbacks_20261012.py ran in .venv-labs: 21 source fixtures, including unchanged/changed adapter cache tests, changed data/seed rejection, four forced missing-widget paths, and all 13 callbacks at slider boundaries. Synthetic tiny inputs only; no model download, secrets, API calls or notebook execution.
- Result JSON: build/lab_source_regressions_20261012.json. Browser widget rendering and Colab remain unverified; root owns current rebuilt notebook smoke/full evidence.

## Clean pinned CPU compatibility — 12 October 2026
- Fresh pinned TRL 1.15.0/Transformers 5.18.0/torch 2.11.0+cpu full run failed after 75.6s: SFTTrainer unconditionally calls its fused LM head, whose optional Triton function is None on Windows. Original failed log/notebook/JSON will be archived before retry.
- Installed TRL source checked: sft_trainer.py 1137 and 1609 attach/use fused head; utils.py 505–509 handles absent Triton but utils.py 1638 calls the missing kernel. No SFTConfig switch disables this path.
- PortableSFTTrainer uses the supported compute_loss override: adapted backbone context, one-position target shift, completion/padding mask, ordinary PyTorch cross-entropy and num_items_in_batch normalization. Only scored positions reach the vocabulary head; real LoRA training is retained.
- Official public hook and completion-only objective checked 12 October: https://huggingface.co/docs/trl/sft_trainer and https://huggingface.co/docs/transformers/main/en/trainer (subclassing guide). Regression/full-run evidence pending.
- PASS: pinned tools/test_lab_repairs.py ran four tests, including tiny random Qwen loss parity, completion masking/shift, LoRA gradient parity, num_items_in_batch normalization, returned outputs and finite all-masked loss.
- PASS: W08 labs-only rebuild. Full pinned CPU retry started 12 October 2026 at 22:26:04 UTC; output staged under build/audit_fixes_20261012/results/week_08. Original failed attempt retained in full_runs.json and failed_week_08_before_portable_loss/. Budget/result pending; coordinator holds W08 renders/builds.
- Official TRL SFT documentation explicitly recommends subclassing SFTTrainer and overriding compute_loss: https://huggingface.co/docs/trl/sft_trainer (checked 12 October 2026). Exact compatibility evidence is the installed pinned TRL 1.15.0 and Transformers 5.18.0 source plus the numerical/gradient regression; the current main Transformers page documents 5.19.0.
- Coordinator-authorised Markdown-only clarification added during the full retry: five tokens give four next-token positions; two non-ignored labels project [2,d] hidden vectors into [2,V] vocabulary scores. The class is supplied support code. Executable code unchanged; root final rebuild will include this explanation.
- PASS: pinned full CPU retry finished in 728.7s (12.145min), within the 900s budget. Real rank-8 SFT used 200 training messages, 30 validation messages, 25 optimiser updates and 50 frozen final-test messages; no hosted API call.
- PASS: inspected the executed notebook: zero error outputs and no failed check message; TODO checks succeeded. Retry record: build/audit_fixes_20261012/full_week_08_retry.json; measured export: build/audit_fixes_20261012/results/week_08/results_cpu_core.json; executed notebook: build/week_08/executed_solutions_full.ipynb. Original failed aggregate is byte-for-byte unchanged; build/.busy released.
- Handoff: root can promote this exact staged export and record, rebuild to include the Markdown-only helper explanation, then render/review W08. Colab timing, live providers and browser widget rendering remain unverified.
