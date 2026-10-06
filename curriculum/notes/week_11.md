# Lecture plan at a glance

This week moves from models to systems (MIMLO 3, 4, 5). Students should leave able to map the tool stack, use API platforms safely (keys, tokens, pricing, rate limits, streaming), compare cloud, self-hosted and local deployment, explain quantisation, batching and caching, build a small app with Gradio and FastAPI packaged in Docker, and design monitoring and guardrails. The measurements on the "real" slides come from running this week's lab in advance (`curriculum/assets/week_11/`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | From notebook to 500 users: what could go wrong? |
| 5–25 min | 5–10 | 1 · Frameworks and the stack | Layers; frameworks table; OpenAI-compatible APIs; SDK vs framework; quiz. |
| 25–47 min | 11–17 | 2 · API platforms | Platform concepts; key safety; request anatomy; real latency; cost estimate; streaming and retry code. |
| 47–72 min | 18–25 | 3 · Deployment and efficiency | Cloud vs self-hosted vs local; quantisation (equation, real results); batching (real); serving engines and caching; what fits where; quiz. |
| 72–79 min | – | Break | |
| 79–93 min | 26–30 | 4 · Building applications | Architecture; FastAPI; Gradio; Docker, versions, CI. |
| 93–114 min | 31–37 | 5 · Monitoring and guardrails | What to monitor; real monitoring summary; layered guardrails; LLMOps tools; comparison table; checklist. |
| 114–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Run the lab's Gradio app with `share=True` at the start of the lecture, put the link on screen and let students use it. At the start of section five, open the log file and show their requests (latency, tokens, blocked attempts: someone always tries a prompt injection). Monitoring becomes concrete immediately.

<!-- pagebreak -->

# Lecture notes

## 1. Frameworks and the tool stack

![The generative AI stack](fig:stack)

Think in **layers**: hardware (cloud GPUs/TPUs, servers, laptops and phones); **models** (proprietary via API; open-weight from the Hugging Face Hub); **access and serving** (cloud APIs, self-hosted engines such as vLLM, local runners such as Ollama); **orchestration** (LangChain/LangGraph, LlamaIndex); **applications and UI** (Gradio, Streamlit, FastAPI back ends). **LLMOps** (logging, tracing, evaluation, guardrails, versioning) spans all layers.

| Tool | Layer | Use it for |
|---|---|---|
| Hugging Face (Hub, Transformers, Inference Providers) | Models, serving | Open models and datasets; local inference or hosted providers |
| OpenAI / Azure OpenAI | Cloud API | GPT models; Azure adds private networking, regional data residency, compliance |
| Google Gemini API / Vertex AI | Cloud API | Gemini models; free tier in AI Studio; enterprise route via Vertex AI |
| vLLM (also SGLang, TGI) | Self-hosted serving | High-throughput GPU serving with an OpenAI-compatible API |
| Ollama (also llama.cpp, LM Studio) | Local | Quantised open models on a laptop; offline and private |
| LangChain / LangGraph | Orchestration | Chains, tools, stateful agent graphs (week 12) |
| LlamaIndex | Orchestration | Data connectors, indexing and retrieval for RAG (week 12) |

**The OpenAI-compatible API.** Most platforms accept the OpenAI chat-completions format: Gemini's compatibility endpoint (`generativelanguage.googleapis.com/v1beta/openai/`), Hugging Face's router (`router.huggingface.co/v1`), vLLM (`http://<server>:8000/v1`), Ollama (`http://localhost:11434/v1`), and the FastAPI server students build. The same client code works everywhere; only `base_url`, `api_key` and `model` change. This reduces lock-in and makes fair comparisons easy.

![One client, many back ends](fig:openai_compat)

**Plain SDK or framework?** Frameworks help when you need connectors, retrievers, state management for agents or tracing integrations; they add abstraction and change quickly. Start with the SDK and plain code; adopt a framework when it solves a concrete problem (the guidance in Anthropic's and OpenAI's agent-building guides).

## 2. API platforms

| Concept | What to know |
|---|---|
| Authentication | API keys, or cloud identities (Microsoft Entra ID for Azure) that avoid long-lived secrets |
| Tokens and pricing | Billed per million input and output tokens; output is usually several times dearer; images and audio are tokens too |
| Rate limits | Requests and tokens per minute; HTTP **429** when exceeded; limits rise with usage tier |
| Streaming | Tokens sent as generated (server-sent events); lower perceived latency |
| Batch APIs and caching | Offline jobs at a discount; prompt (prefix) caching discounts repeated prefixes |
| Structured output and tools | JSON schemas, function calling (weeks 7 and 12) |

**Key safety.** Never put keys in notebooks, code or Git; use Colab Secrets, environment variables or a secrets manager (Azure Key Vault, Google Secret Manager). Keys stay on the server: browsers and mobile apps call your back end, never the provider directly. Separate keys per project, spending limits and alerts, regular rotation. If a key leaks, revoke it first.

**Where the time goes.** Network and authentication, then queueing under load, then **prefill** (the whole prompt processed in parallel; grows with prompt length), then **decoding** one token at a time.

$$\text{latency} \approx \text{TTFT} + \frac{\text{output tokens}}{\text{tokens per second}}$$

![Anatomy of a request](fig:request_anatomy)

**Lab measurement:** {{LAT_NOTES}}

**Cost estimate.** For $R$ requests with $n_{in}$ input and $n_{out}$ output tokens at prices $p_{in}$ and $p_{out}$ per million tokens: cost $= R\,(n_{in} p_{in} + n_{out} p_{out}) / 10^6$. For 500 students × 20 requests a day × 30 days = 300,000 requests with 800 input and 300 output tokens, illustrative prices give:

| Option (illustrative prices) | Price per 1M tokens (in / out) | USD per month |
|---|---|---|
| Small hosted model | \$0.10 / \$0.40 | 60 |
| Mid-size hosted model | \$0.50 / \$2.00 | 300 |
| Frontier hosted model | \$2.00 / \$10.00 | 1,380 |
| Self-hosted small GPU, always on | \$0.60 per hour | 432 + staff time |

> **Key idea:** Prices change often; teach the method, and ask students to look up current prices for their project.

**Retries.** On HTTP 429 or transient errors, retry with **exponential back-off and jitter** (wait $b \cdot 2^{k}$ plus a random fraction), cap the number of retries, and set `max_tokens` and a timeout on every call.

## 3. Deployment options and efficient serving

| | Cloud API | Self-hosted (vLLM on GPUs) | Local (Ollama) |
|---|---|---|---|
| Models | Frontier proprietary and hosted open models | Open-weight, including your fine-tunes | Small, quantised open models |
| Cost profile | Pay per token; no fixed cost | Fixed GPU cost; cheap per token at high, steady load | Free after hardware |
| Data | Leaves your network (check terms, region) | Stays in your infrastructure | Stays on the device |
| Operations | Provider scales and updates | You run, scale, patch, monitor | Per-device setup |
| Risks | Price/model changes, rate limits, lock-in | Capacity planning, skills | Limited quality and speed |

Hybrids are common: a small local or self-hosted model for routine requests, a cloud model for hard ones.

### Quantisation

$$\text{memory} \approx N_{\text{params}} \times \frac{\text{bits}}{8}\ \text{bytes}$$

$$s = \frac{\max |w|}{2^{b-1} - 1}, \qquad q = \mathrm{round}\left(\frac{w}{s}\right), \qquad \hat{w} = q \cdot s$$

A 7 B model needs about 14 GB in 16-bit, 7 GB in 8-bit and 3.5–4 GB in 4-bit, plus the KV cache. **Absmax** quantisation stores integers and one scale per group; smaller groups (e.g. 64 weights) keep more precision, the basis of GPTQ, AWQ and GGUF (used by vLLM, llama.cpp and Ollama). `bitsandbytes` loads models in 8-bit or 4-bit NF4 in one line (QLoRA, week 8).

**Lab measurement:** {{QUANT_NOTES}}

![Size and perplexity after quantisation (lab)](fig:lab_quant)

### Batching and serving engines

Decoding is limited by memory bandwidth: each step reads all weights to produce one token per sequence. **Batching** reuses each weight read for many sequences, raising throughput. **Continuous batching** (vLLM, SGLang, TGI) adds and removes requests at every step. **PagedAttention** (Kwon et al., 2023) stores the KV cache in fixed-size pages, reducing waste and allowing larger batches. Larger batches raise throughput but can raise each user's latency.

**Lab measurement:** {{BATCH_NOTES}}

![Throughput vs batch size (lab)](fig:lab_batch)

### Caching and other savings

* **KV cache** (week 5) and **prefix caching**: reuse computation for shared prompt prefixes; providers discount cached input tokens.
* **Response caching**: identical requests answered from a cache (milliseconds instead of seconds in the lab).
* **Speculative decoding**: a small draft model proposes tokens that the large model verifies in parallel.
* **Right-sizing**: a fine-tuned or distilled small model is often the biggest saving.

**What fits where (weights only):** 0.5 B in 16-bit ≈ 1 GB (any laptop); 7–8 B in 4-bit ≈ 4 GB (laptop with Ollama, or a free Colab T4); a T4 has 16 GB (≈ 7 B in 16-bit, ≈ 20–30 B in 4-bit); 70 B in 16-bit ≈ 140 GB (several data-centre GPUs).

## 4. Building applications and interfaces

![Proposed integrated architecture of a small generative AI application](fig:app_architecture)

This figure shows a proposed integrated design. In the lab the two routes are separate: Gradio calls `respond` and `generate` directly, while the SDK and LangChain call the FastAPI endpoint. The endpoint does not inherit the checks or logging in `respond`. The exported `app.py` is also a smaller packaging scaffold; it omits the notebook's personal-data redaction, output check and metadata logging.

* **FastAPI** turns Python functions into a web API; Pydantic validates requests; interactive docs at `/docs`. Returning the OpenAI response format (with a `usage` block) makes the server usable by the OpenAI SDK and LangChain.
* **Gradio** builds interfaces from Python functions; `gr.ChatInterface(respond)` gives a full chat UI. `share=True` creates a temporary public link: anyone with it can use your quota. **Hugging Face Spaces** hosts Gradio apps with secrets support. Streamlit is an alternative.
* **Docker** packages code, pinned dependencies and the start command; runs on laptops, Hugging Face Spaces, Google Cloud Run or Azure Container Apps. Pass secrets as environment variables at run time.
* **Versioning and release:** version the model, prompts, decoding settings and retrieval index; run the evaluation set in CI and compare with the current version; roll out gradually and keep a rollback path.

## 5. Monitoring, guardrails and comparison

| Area | Metrics |
|---|---|
| Performance | TTFT, p50/p95/p99 latency, throughput, timeouts |
| Cost | Tokens per request, per user, per day; budget alerts |
| Reliability | Error rates, HTTP 429s, provider outages, fallbacks |
| Quality | User feedback, sampled human or LLM-judge review, regression tests, drift |
| Safety | Guardrail blocks, redactions, flagged outputs, injection attempts |
| Versions | Model, prompt and index behind each answer |

**Lab measurement:** {{MONITOR_NOTES}}

![Monitoring summary from the lab app's logs](fig:lab_monitor)

**Guardrails in layers:** input checks (length, rate limits, PII redaction, injection and topic filters) → model setup (system prompt, least-privilege tools, safe decoding) → output checks (moderation classifier, schema validation, leak and policy checks) → human oversight → monitoring and red-teaming. Regex rules are only a start; production systems add trained safety classifiers (e.g. Llama Guard-style models), moderation APIs or libraries such as NeMo Guardrails and Guardrails AI.

**LLMOps tooling:** tracing records each step of a request with timings and tokens. OpenTelemetry is the open standard; Langfuse (open source), LangSmith, MLflow and cloud consoles (Azure AI Foundry, Vertex AI) provide tracing, evaluation and dashboards.

| Need | Good choice | Why |
|---|---|---|
| Best quality, quick prototype | Cloud API (free tier) + Gradio | No infrastructure; strongest models |
| Private data, no budget | Ollama or Transformers locally | Data stays on the machine |
| Fine-tuned model for a demo | Hugging Face Spaces or Colab + Gradio | Free hosting of your own model |
| Many users, steady load | vLLM on a cloud GPU | Batching gives low cost per token |
| Reproducible hand-in | Docker image + pinned versions | Examiners can run it |

# Common misconceptions

| Misconception | Correction |
|---|---|
| "You need LangChain to build an LLM app." | Many apps need only an SDK and plain code; frameworks help with connectors, agent state and tracing. |
| "Putting the API key in a private notebook is fine." | Notebooks get shared and committed; use secrets managers and keep keys server-side. |
| "Input tokens and output tokens cost the same." | Output tokens are usually several times more expensive and also dominate latency. |
| "Self-hosting is always cheaper." | Only at high, steady load; an idle GPU costs money, and operations need staff time. |
| "Quantisation always ruins quality." | 8-bit is usually almost lossless and 4-bit with small groups often close; but always re-evaluate. |
| "Average latency is a good metric." | Averages hide slow requests; use percentiles (p95/p99) and TTFT. |
| "Guardrails make the model safe." | They reduce risk; each layer can be bypassed, so combine layers with least privilege, monitoring and human oversight. |
| "Log everything to debug later." | Logging raw conversations can breach data protection; log metadata, redact, limit retention. |

# Responsible AI lens: deploying to real users

* **Transparency:** tell users they are interacting with an AI system and label generated content (EU AI Act Art. 50, applicable from 2 August 2026).
* **Data protection:** know where prompts go and how long providers keep them; minimise and redact; data-processing agreements; regional hosting where required (GDPR).
* **Security:** OWASP Top 10 for LLM applications (prompt injection, sensitive information disclosure, improper output handling, excessive agency, unbounded consumption, …); authentication and rate limiting; secrets management.
* **Accountability:** version and log which model and prompt produced each answer; keep an escalation path to a human.
* **Fairness and quality over time:** monitor quality by user group and topic; models and data drift.
* **Environment and cost:** right-size models, cache, batch and switch off idle resources.

# Lab guide and answers

**Runtime:** Colab T4 GPU (≈ 20 min) or CPU (slower; the notebook works unchanged). Model: Qwen2.5-0.5B-Instruct (SmolLM2-135M in the automated smoke test). Cloud API keys (`GEMINI_API_KEY`, `HF_TOKEN`) are optional and read only from Colab Secrets. **Hand-in:** latency, cost, quantisation, batching and monitoring tables; app link or screenshot; three written answers.

## TODOs

* **TODO 1** in `measure`: set `ttft` at the first non-empty chunk; `tok_per_s = (n_tokens - 1) / (total - ttft)`.
* **TODO 2** FastAPI endpoint: call `generate(req.messages, ...)` and return the OpenAI response format with `choices` and `usage`.
* **TODO 3** `monthly_cost = requests × (in_tokens × price_in + out_tokens × price_out) / 1e6`.
* **TODO 4** `fake_quantize`: per-group scale `amax / qmax`, round, clamp to ±qmax, multiply back.
* **TODO 5** `check_input`: block > 2,000 characters and injection phrases; redact e-mails and phone numbers; return `(allowed, cleaned, reason)`.

## Measured results (instructor run)

{{RESULTS_NOTES}}

## Model answers to the written questions

1. **Cloud API or self-hosting for 500 students?** At illustrative prices a small hosted model costs far less than an always-on GPU and needs no operations, so a hosted API is the sensible default; a frontier model costs more than the GPU. Self-hosting wins with high, steady volume (batching gives high throughput), data that must stay on premises, a fine-tuned open model, or strict latency needs. Assumptions that change the answer: volume and peaks (deadline rushes), output length, scaling the GPU to zero out of hours, current prices and free tiers, required quality, and staff time.
2. **Guardrails.** The regex rules blocked the literal injection phrase and redacted the e-mail and phone number. They are bypassed by paraphrases, other languages, encodings, typos, indirect injection in retrieved documents and PII in unusual formats. Add trained moderation or safety classifiers on inputs and outputs, least privilege, schema validation, per-user rate limits and authentication, human escalation, red-teaming and monitoring of block rates, and AI disclosure.
3. **Monitoring design.** Metrics: volume, latency percentiles and TTFT, errors and 429s, tokens and cost per user/day, guardrail rates, model and prompt versions, quality signals (feedback, sampled review, regression tests). Alerts: p95 above target for several minutes, error-rate spikes, daily cost above budget, sudden rise in blocks, falling feedback. Do not log raw personal data, secrets or full conversations by default; redact, restrict access, set retention and document it (GDPR).

## Troubleshooting

* *Port already in use* when re-running Part 2: restart the runtime, or change `PORT`.
* *Gradio link does not appear:* make sure the cell ran on Colab with internet access; `share=True` needs outbound connections.
* *`bitsandbytes` error on CPU:* expected; the 4-bit cell runs only on a CUDA GPU.
* *Cloud API errors:* check the key is stored as a Colab Secret with notebook access enabled; model names change, so list models with `client.models.list()`; free tiers have strict rate limits (HTTP 429).
* *Different numbers from the slides:* latency depends on hardware and load; the patterns (TTFT ≪ total, batching gains, int8 ≈ lossless) should match.

# Practice questions with model answers

## Multiple choice

1. A client using the OpenAI SDK can call a vLLM server by changing: **(a)** the prompt; **(b)** `base_url`, `api_key` and `model`; **(c)** the tokenizer; **(d)** nothing. *Answer: (b).*
2. Which mostly determines total latency for long answers? **(a)** network time; **(b)** prefill; **(c)** the number of output tokens; **(d)** the API key. *Answer: (c).*
3. HTTP 429 means: **(a)** invalid key; **(b)** rate limit exceeded; **(c)** server crash; **(d)** content blocked. *Answer: (b).*
4. A 7 B model in 4-bit needs roughly how much memory for weights? **(a)** 0.9 GB; **(b)** 3.5–4 GB; **(c)** 14 GB; **(d)** 28 GB. *Answer: (b).*

## Short answer

1. **Estimate the monthly cost of 100,000 requests with 1,000 input and 200 output tokens at \$0.50 / \$2.00 per million tokens.** (3 marks) *Model answer:* 100,000 × (1,000 × 0.5 + 200 × 2) / 10⁶ = 100,000 × 0.0009 = \$90.
2. **Explain why batching increases throughput on a GPU and one trade-off.** (3 marks) *Model answer:* decoding is memory-bandwidth-bound; a batch reuses each weight read for many sequences. Trade-off: larger batches can increase per-request latency and need more KV-cache memory.
3. **Why should an API key never be placed in a web page's JavaScript?** (2 marks) *Model answer:* anyone can read client-side code and use the key at your expense; calls must go through a back end that holds the key.
4. **Compare cloud APIs and self-hosting on three criteria.** (4 marks) *Model answer:* cost (per token vs fixed GPU; self-hosting cheaper only at high steady load), data control (leaves network vs stays in-house), operations (provider-managed vs your responsibility), model choice (frontier proprietary vs open and fine-tuned).

## Exam-style question

**"A university wants a course assistant chatbot for 2,000 students. Propose a deployment architecture, justify the choice between a cloud API and self-hosting with a cost estimate, and describe the monitoring and guardrails you would put in place."** (20 marks)

*Marking guide:* architecture (UI, API, model, guardrails, logging) (4); deployment choice with cost arithmetic, data protection and quality considerations (6); efficiency measures (streaming, caching, quantisation or batching as relevant) (3); monitoring metrics and alerts (4); guardrails, transparency and data protection (3).

# Glossary

| Term | Meaning |
|---|---|
| API platform | Service providing model access over HTTP with keys, quotas and billing |
| OpenAI-compatible API | Endpoint accepting the OpenAI chat-completions format |
| TTFT | Time to first token: delay before the first output token arrives |
| Tokens per second | Decoding speed after the first token |
| Prefill / decode | Processing the prompt in parallel / generating tokens one by one |
| Rate limit (HTTP 429) | Cap on requests or tokens per minute |
| Exponential back-off | Retrying with waits that double each attempt, plus jitter |
| Quantisation | Storing weights with fewer bits (int8, int4) |
| Absmax / group-wise | Scaling by the largest magnitude, per group of weights |
| Continuous batching | Adding and removing requests at every decoding step |
| PagedAttention | vLLM's paged management of the KV cache |
| Prefix (prompt) caching | Reusing computation for repeated prompt beginnings |
| Speculative decoding | Draft model proposes tokens, large model verifies |
| Gradio / FastAPI | Python libraries for web interfaces / web APIs |
| Docker image | Packaged application with dependencies |
| Tracing | Recording each step of a request with timings |
| Guardrail | Check that blocks or modifies unsafe inputs or outputs |
| p95 latency | Latency below which 95% of requests complete |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Tunstall, L., von Werra, L. and Wolf, T. (2022) *Natural Language Processing with Transformers*: chapter 8, "Making Transformers Efficient in Production".

**Papers**

* Kwon et al. (2023) [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180); Dettmers et al. (2022) [LLM.int8()](https://arxiv.org/abs/2208.07339); Leviathan et al. (2023) [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192)

**Documentation and courses**

* [Microsoft: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners)
* [Microsoft Foundry: Azure OpenAI and models sold directly by Azure](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure)
* [Gemini API: OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) · [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/get-started)
* [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) · [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
* [vLLM documentation](https://docs.vllm.ai/en/latest/) · [Ollama](https://ollama.com/)
* [LangChain documentation](https://docs.langchain.com/) · [LlamaIndex documentation](https://developers.llamaindex.ai/python/framework/)
* [Gradio quickstart](https://gradio.app/guides/quickstart) · [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
* [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

# Link to the group project

Every project must be runnable by someone else. Minimum: a Gradio (or similar) interface, keys in secrets, pinned dependencies (ideally a Dockerfile), a log of latency and tokens per request, basic input/output guardrails, and a short deployment section in the report that justifies cloud vs local with measured latency, an estimated monthly cost and a quality check.
