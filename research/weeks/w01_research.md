# R1 · Week 01: landscape, probability and sampling

Checked **2026-10-10**. Read: built slide titles and lab headings only. Proposals below are teaching designs, not measured results.

**Explanations to adapt (redraw; do not copy).**
- A name-maker learns a table of next-letter counts before it draws letters. Karpathy's [makemore lesson](https://karpathy.ai/zero-to-hero.html) connects counts, tensors, sampling and negative log likelihood. Adapt as a bag containing coloured counters; unlike a physical bag, the next distribution depends on the preceding token.
- Tokens are the model's text pieces, not necessarily whole words. Start with one word split into pieces, then replace each with an integer; [Google MLCC](https://developers.google.com/machine-learning/crash-course/llm) supports this context-first explanation. A token ID is an index, not a meaning or probability.
- A temperature slider changes how concentrated the next-token lottery is. Link the concrete bar chart to [Transformers generation parameters](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation); the slider changes sampling, not learned parameters or factual knowledge.

**Five misconceptions to confront.** (1) A fluent response is evidence of truth. (2) One token always equals one word. (3) Inference updates model weights. (4) Lower temperature guarantees a correct answer. (5) Open weights imply unrestricted use or complete access to training data. Put corrections beside the first relevant example.

**Current practice and versions.** An existing compact local example is [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), whose publisher documents CPU loading and factual limitations; use the base model for next-token probabilities, distinguishing an instruction-tuned assistant. Avoid a claim that it is the newest model. Keep dated historical model names in the history section. R0's official Colab inventory records torch `2.11.0+cpu`, transformers `5.18.0`, ipywidgets `7.7.1`; these are published snapshots, not a tested dependency lock. Gemini uses `google-genai`; see [R0](platform_2026.md) for the verified catalogue and API pattern.

**Own-style visuals and beginner barriers.** Replace a landscape icon with a model/application/tool mini-table. Draw counts → row-normalised probabilities → next-token draw → append token, labelling the count/probability shapes. Separate a training loop from one inference step. The title sequence introduces Bayes, likelihood, chain rule and latent variables quickly; explain a three-event calculation first. Lab barriers: indexing a count table, normalising the correct axis, model downloads, dictionaries and token decoding. Add a readiness check and printed shapes before abstractions.

**CPU core proposal, target ≤15 minutes of waiting; unmeasured.** Count a small synthetic name list, normalise, predict and sample; change temperature with a slider. Inspect one short local-model prompt with capped generation; keep stored measured outputs available if download fails. Check probability sums, shapes and the student's evidence table. Optional Gemini: a few synthetic prompts to compare fluency and unsupported claims; `USE_GEMINI=False` and complete local fallback. R0 notes an unresolved EEA/Swiss/UK distributed-client paid-service condition: do not promise free regional access or enable billing.

**Further reading:** [Google LLM introduction](https://developers.google.com/machine-learning/crash-course/llm) · [makemore video](https://www.youtube.com/watch?v=PaCmpygFfXo) · [Transformers decoding](https://huggingface.co/docs/transformers/generation_strategies) · [SmolLM2 publisher card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

Handoff: research complete; no build, runtime, API call or widget-rendering claim.
