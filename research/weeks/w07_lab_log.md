# w07_lab_log

## Iteration 3 (Claude, 11 Oct 2026)

- New lab: Qwen2.5-0.5B-Instruct on CPU; 40 synthetic tickets split 20 development / 20 held-out; six TODOs each with a `check_…()`; a prompt-size slider tagged skip-test; optional hosted judge through `ask_model` (eight calls at most, synthetic pairs only); `GENAI_RESULTS_PATH` export.
- Review fixes before the first run: transformers 5 `apply_chat_template` returns a BatchEncoding, so the slider counts `["input_ids"]`; judge consistency uses tuple membership; hosted table uses `display`.
- Full CPU run 463.4 s (7.7 min) PASS, recorded in the ledger; smoke PASS after the final rebuild.
- Measured core (11 Oct): zero-shot 20%, one-shot 25%, few-shot 50%, role + few-shot 50% on 20 development messages; chosen role + few-shot 45% held out; JSON 100% valid but 33% fully correct (invented dates); judge picked A in both orders for all four pairs. These are genuine small-model behaviours and are used as teaching examples.
- Proposal for M2: keep `results.json` (stored 1.5B GPU run) and `results_cpu_core.json` (CPU core) as separate, labelled evidence.

## Final coordinator handoff — 12 October 2026
- Latest measured cached local CPU core 439.0s (7.3min), within 900 s; final smoke PASS (96.5s), zero failed feedback checks. Recorded full-run code matches current executable core.
- Final image review closed; canonical evidence is `research/ITERATION3_2026-10-12.json`. Earlier unmeasured/pending handoffs are historical. No timed Colab, live API, clean-install/download, browser-widget or optional-stretch verification.
