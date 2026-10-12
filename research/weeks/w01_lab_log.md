# W01 lab handoff — iteration 3

- 2026-10-11: source-to-candidate diff reviewed before applying; no intervening edits detected. Previous source SHA256: `E3944A66EDB404E45FA48483557636918E3AB79CC5C84C78D3C726C3F924364C`.
- Applied the prepared candidate to `curriculum/labs/week_01_lab.py`. Shared setup now owns seed, device, smoke mode and Gemini access. The local route calls `ask_gemini()` and retains the local response when it returns `None`.
- Applied source SHA256: `C401B5BF8CA3DF0DE1F6470DCE36ECF826A1345AC6FF1670A0555305AA3BE3D1`. `git diff --check` passed for the source and handoff log in this session.
- CPU core: real `HuggingFaceTB/SmolLM2-135M-Instruct`; ordinary completions capped at 32 tokens and fictional-paper responses at 64 (smoke cap remains 16). The larger 360M checkpoint is an optional stretch.
- Preserved all five scientific TODOs, their solution markers, the original STUB, classifiers, likelihood/scoring/sampling mechanisms, `GENAI_RESULTS_PATH` hook and exact original JSON export fields.
- Added readiness checks, printed shapes, TODO hints and assert-backed `check_*()` helpers, per-part predict/core/change/stretch/common-errors support, a temperature slider tagged `skip-test`, and five final answer-marked self-checks.
- Fixed the data fallback: its name list omits `q`, but the NLL examples score `xqzz`; the vocabulary now includes all 26 letters. The sampler check restores RNG state so validation does not consume experimental draws.
- Removed unmeasured general claims about hosted-model reliability. Gemini remains optional, synthetic-data-only, with the shared regional/account caveat and complete local fallback.
- Evidence untouched: no assets, stored result JSON, fill placeholders or run-ledger entries changed. Any retained 360M slide/notes results must identify that stored checkpoint and settings. Root will export the new 135M full run to scratch and compare before deciding evidence replacement through `fill_results`/M2.
- **Pending:** root rebuild, smoke run, full CPU core measurement and run-record proposal. No build, notebook execution, download or provider call was performed for this handoff; widget rendering in Colab/VS Code is unverified.

Handoff: root rebuilt and saw smoke/fullCPU pass; latest serial fullCPU core43.2seconds recorded in research/lab_runs_2026-10.json (cached setup/models; exact environment/hashes retained). New scoring diagram checked. Authored slider construction and callback passed locally at three temperatures (build/w01_widget_probe.log); browser rendering, live Gemini and Colab execution remain unverified. E3f adds two matching lecture-flow images before final pilot build.

| Week | File | Cell | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 01 | lab | TODO5 | bfloat16 probability sum compared with float32 value; correct solution failed its check | High | Compute softmax probabilities in float32; updated hint/comment | Closed: rebuilt smoke/fullCPU passed |

- Final handoff (11 Oct): default optional comparison now calls ask_model(), selecting exactly one shared Gemini/OpenAI route and reporting its actual checkpoint. Root observed 22setup/25evidence tests and actual SDKconstructor/argument probe pass. Latest built fullCPUcore42.5seconds and smoke26.7seconds passed; full-run hash/environment/settings recorded in central ledger. Models/data cached locally; live APIs, clean-install/Colab timings and browser widget rendering remain unverified. Both added lecture-flow images matched the reviewed slide/Word/diagram copies.

## Final coordinator handoff — 12 October 2026
- Latest measured cached local CPU core 44.6s (0.7min), within 900 s; final smoke PASS (65.7s), zero failed feedback checks. Recorded full-run code matches current executable core.
- Final image review closed; canonical evidence is `research/ITERATION3_2026-10-12.json`. Earlier unmeasured/pending handoffs are historical. No timed Colab, live API, clean-install/download, browser-widget or optional-stretch verification.
