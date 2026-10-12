# W07 research: prompt engineering with evidence

Checked **2026-10-10**. R3 scope: built slide titles and lab headings only; no lab execution.

**Teaching explanations (adapt, do not copy):**
- Brief a capable new colleague: supply task, context, examples and output constraints ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). Limit: a model has no human understanding or guaranteed compliance.
- Few-shot examples are solved exercises: show the pattern before asking a new question ([Google](https://ai.google.dev/gemini-api/docs/prompting-strategies)). Limit: examples condition this response; they do not update weights.
- A JSON schema is a form with required boxes ([structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)). Limit: valid structure does not establish factual correctness. These last two analogies are teaching adaptations.

**Five misconceptions:** (1) more examples always improve accuracy; (2) a role prompt grants expertise; (3) visible reasoning faithfully explains internal computation; (4) valid JSON means correct extraction; (5) a model judge or ten test cases proves superiority. Counter each with one failure, then a held-out check.

**Current practice:** define success criteria before tuning; separate development and held-out examples; compare identical inputs and decoding settings. Modern thinking controls differ by model: teach CoT as a historical technique, not a universal instruction ([Anthropic guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) supports `enable_thinking=False`; thinking mode requires sampling, not greedy decoding. It is a teaching checkpoint, not a claim of newest/best. Use R0's dated package inventory and Gemini constant; E2 must validate compatibility.

**Redraw:** prompt components -> chat template -> tokenizer -> model -> parser -> rubric, with text/token IDs/JSON/score arrows; dashed feedback returns rubric to prompt editing. Add a four-example prediction table, a JSON parse-versus-content mini-table and a paired judge-order example. Use role colours and a legend; no copied art.

**Beginner barriers:** five experiments introduce generation, sampling, JSON, injection and judges together. Teach one routing example end to end first. Explain lists of message dictionaries, decoding, exceptions and the denominator in accuracy before code; move ROUGE details, self-consistency and statistical inference to worked notes/stretch.

**Proposed CPU core, target <=15 minutes (unmeasured):** 12 synthetic routing messages, two prompts, bounded short completions using [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct); print failures and check labels/JSON locally. Slider changes the number of examples without launching generation. Keep larger Qwen/self-consistency/judge runs optional; inspect stored measured outputs. Record cold-download and warm execution separately. Optional Gemini: <=24 initial calls for the same pairs, leaving retry budget below the course ceiling; do not compare different models as a pure prompting effect. `USE_GEMINI=False`, synthetic data, R0 regional terms caveat and complete local fallback remain mandatory; free eligibility is not promised.

**Further reading (5):** [Google prompt guide](https://ai.google.dev/gemini-api/docs/prompting-strategies); [Anthropic prompting](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output); [reasoning faithfulness research](https://assets.anthropic.com/m/71876fabef0f0ed4/original/reasoning_models_paper.pdf); [Google Cloud Tech video: Prompt engineering for developers](https://www.youtube.com/watch?v=I0DBxnTlaMw) (2024; qualify its CoT advice for current models).

Handoff: source research complete; video metadata verified in search, direct playback untested. Runtime, downloads, Gemini access and widget rendering unmeasured.
