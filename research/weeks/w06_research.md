# R2 · Week 06: language models and model choice

Checked **2026-10-10**. Scope: built slide titles and lab headings. Proposals need implementation and timing.

**Teaching explanations (adapt, then state the limit).**
- Autocomplete appends one sampled token and asks again. Continuations need fact-checking; tokens need not be words. [3Blue1Brown](https://www.3blue1brown.com/lessons/mini-llm/).
- Training adjusts many dials; inference reads their settings. Prompt demonstrations change context, leaving weights fixed. Separate training/inference diagrams make the source's dial analogy precise. [3Blue1Brown](https://www.3blue1brown.com/lessons/mini-llm/).
- Choose by job: encoders represent inputs; decoders continue sequences; encoder-decoders generate using encoded input. These architecture families do not rank intelligence. [HF architectures](https://huggingface.co/learn/llm-course/en/chapter1/6).

**Five misconceptions to correct.**

| Misconception | Correction |
|---|---|
| More parameters guarantee the best choice | Evaluate the actual task and constraints. |
| A longer context means reliable recall | Capacity does not establish task accuracy. |
| Few-shot prompting updates model weights | Ordinary inference leaves weights fixed. |
| Temperature controls truthfulness | It changes the sampling distribution. |
| Perplexity ranks every model fairly | Tokenisation and evaluation setup affect it. |

For the last point, use [HF perplexity guidance](https://huggingface.co/docs/transformers/en/perplexity); it is primarily for autoregressive models, not a direct masked-BERT score.

**Current practice.** Label BERT/T5/GPT-2 historical. Official `Qwen/Qwen3-0.6B` is a small causal checkpoint labelled Apache-2.0 with thinking/non-thinking modes, without implying newest-model status. Thinking-mode guidance: sample with temperature 0.6, top-p 0.95, top-k 20; avoid greedy decoding. [Qwen card](https://huggingface.co/Qwen/Qwen3-0.6B). Use checkpoint-specific chat templates, explicit CPU placement and short output caps. [Chat templates](https://huggingface.co/docs/transformers/en/chat_templating). Verify licences/limitations. R0 provides a dated stable Gemini ID and published Transformers snapshot; avoid `latest` aliases and universal quota claims. [R0](platform_2026.md).

**Redraw.** Text → tokenizer → token IDs → decoder → logits → sampler → appended token; use dashed repetition, labelled shapes and role legend. A separate task/input/output strip compares three architectures without implying identical objectives.

**Beginner barriers.** Scaling laws, tokenisation, decoding, licences, APIs and reasoning appear together. Lab headings imply several downloads and perplexity calculations. Start with one next-token table and one API response; isolate dictionary access, slicing, log probabilities and unsupported arguments in short worked cells.

**CPU core proposal: target ≤15 minutes, unmeasured.** Local toy tokenisation and probability exercises; a small local causal-model inference with capped text/output; held-out perplexity using one tokenizer; compare stored measured encoder/decoder/encoder-decoder outputs. Temperature/top-p widgets change a probability plot (`skip-test`). Full multi-checkpoint comparisons and Qwen thinking runs are stretches. Include download/setup waiting in the measured budget; keep the local probability exercises available offline.

**Gemini role.** Recommended optional hosted comparison on three synthetic prompts, scored against a visible answer rubric; preserve local completion. Shared `USE_GEMINI=False`, caching and bounded retry. R0 flags EEA/UK distributed-client terms: do not promise free regional eligibility or enable billing. [R0](platform_2026.md).

**Further reading:** [HF architectures](https://huggingface.co/learn/llm-course/en/chapter1/6), [generation parameters](https://huggingface.co/docs/transformers/en/main_classes/text_generation), [perplexity](https://huggingface.co/docs/transformers/en/perplexity), [Qwen card](https://huggingface.co/Qwen/Qwen3-0.6B), [Karpathy tokenizer video](https://www.youtube.com/watch?v=zduSFxRajkE).

Handoff: research complete; primary sources checked. No lab/API/widget execution; CPU timing and Colab compatibility await B/V.
