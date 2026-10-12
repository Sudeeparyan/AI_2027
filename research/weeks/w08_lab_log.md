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
