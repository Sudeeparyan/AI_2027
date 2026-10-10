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

Start with the vocabulary and overall flow. For each section below, read the simple explanation first, trace the worked example aloud, and ask students to name the input and output. Introduce the equation only after the operation makes sense. The detailed text retains the full syllabus, while the lab and checks show what each method can and cannot establish.

## 1. Frameworks and the tool stack

**Start here.** A model predicts output; a server accepts requests; application code prepares input and handles errors; a user interface lets people interact. An SDK is a library for calling a service. A framework connects steps and tools. Start by tracing one request through these layers.

**Small worked example.** A browser sends text to your app server. The server keeps the provider key and makes the model call. The browser receives an answer; it should not receive the secret key.

![The generative AI stack](fig:stack)

Think in **layers**: hardware (cloud GPUs/TPUs, servers, laptops and phones); **models** (proprietary via API; open-weight from the Hugging Face Hub); **access and serving** (cloud APIs, self-hosted engines such as vLLM, local runners such as Ollama); **orchestration** (LangChain/LangGraph, LlamaIndex); **applications and UI** (Gradio, Streamlit, FastAPI back ends). **LLMOps** (logging, tracing, evaluation, guardrails, versioning) spans all layers.

| Tool | Layer | Use it for |
|---|---|---|
| Hugging Face (Hub, Transformers, Inference Providers) | Models, serving | Open models and datasets; local inference or hosted providers |
| OpenAI / Azure OpenAI | Cloud API | GPT models; Azure adds private networking, regional data residency, compliance |
| Google Gemini API / Agent Platform | Cloud API | Gemini models; check current AI Studio access and pricing; enterprise route via Gemini Enterprise Agent Platform (formerly Vertex AI) |
| vLLM (also SGLang, TGI) | Self-hosted serving | High-throughput GPU serving with an OpenAI-compatible API |
| Ollama (also llama.cpp, LM Studio) | Local | Quantised open models on a laptop; offline and private |
| LangChain / LangGraph | Orchestration | Chains, tools, stateful agent graphs (week 12) |
| LlamaIndex | Orchestration | Data connectors, indexing and retrieval for RAG (week 12) |

**The OpenAI-compatible API.** Most platforms accept the OpenAI chat-completions format: Gemini's compatibility endpoint (`generativelanguage.googleapis.com/v1beta/openai/`), Hugging Face's router (`router.huggingface.co/v1`), vLLM (`http://<server>:8000/v1`), Ollama (`http://localhost:11434/v1`), and the FastAPI server students build. Basic client code can often be reused by changing `base_url`, credentials and `model`. Supported fields, streaming, tools and structured output can differ; check the chosen endpoint. This reduces lock-in and makes fair comparisons easy.

![One client, many back ends](fig:openai_compat)

**Plain SDK or framework?** Frameworks help when you need connectors, retrievers, state management for agents or tracing integrations; they add abstraction and change quickly. Start with the SDK and plain code; adopt a framework when it solves a concrete problem (the guidance in Anthropic's and OpenAI's agent-building guides).

## 2. API platforms

**Start here.** An API connects programs. A request has credentials, input fields and limits; a response has answer data or an error. Streaming returns pieces before the whole answer is complete. First-token wait and total wait measure different parts. Prices, rate limits and supported fields vary by service.

**Small worked example.** With TTFT = 1 second and 100 output tokens at 20 tokens/second, the rough total is 1 + 100/20 = 6 seconds. At illustrative prices $0.10 input/$0.40 output per million, 800 input and 300 output tokens cost $0.0002 per request.

| Concept | What to know |
|---|---|
| Authentication | API keys, or cloud identities (Microsoft Entra ID for Azure) that avoid long-lived secrets |
| Tokens and pricing | Billed per million input and output tokens; output is usually several times dearer; images and audio are tokens too |
| Rate limits | Requests and tokens per minute; HTTP **429** when exceeded; limits rise with usage tier |
| Streaming | Tokens sent as generated (server-sent events); lower perceived latency |
| Batch APIs and caching | Offline jobs at a discount; prompt (prefix) caching discounts repeated prefixes |
| Structured output and tools | JSON schemas, function calling (weeks 7 and 12) |

**Key safety.** Treat an access key like a password. Keep it out of code, notebooks and Git. Load it from Colab Secrets, an environment variable or a secrets manager. App browsers call your server, which holds the provider key. Use limited permissions, separate keys and spending alerts where supported. If a key leaks, revoke it first, then inspect its use.

**Where the time goes.** A request travels over the network and may wait in a queue. **Prefill** processes the prompt; longer prompts generally add work. **Decoding** then produces new tokens in sequence. **TTFT** measures the wait to the first token. The formula below is a rough model of total waiting time, not a guarantee for every serving system.

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

**Retries.** A rate limit or temporary failure may justify a retry. **Exponential back-off** waits longer each time, for example $b \cdot 2^k$; **jitter** adds a random delay so clients do not all retry together. Respect service retry guidance, cap attempts and set a timeout and output limit. Do not retry permanent errors indefinitely.

## 3. Deployment options and efficient serving

**Start here.** Hosting chooses where the model runs and who maintains it. Quantisation reduces the number of bits storing weights. Batching shares model work across requests. Caching reuses a previous computation or answer. Each optimisation needs both performance measurements and a quality check on the actual task.

**Small worked example.** A billion weights at 16 bits use about 1,000,000,000 × 16/8 = 2 GB, before overhead. At 4 bits the ideal weight storage is 0.5 GB. Runtime caches and temporary tensors still need memory.

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

For 7 B weights, ideal storage is about 14 GB at 16 bits, 7 GB at 8 bits or 3.5 GB at 4 bits, before scales and runtime overhead. **Absmax quantisation** scales and rounds each group of weights to integers. Smaller groups use extra scales but may keep more detail. GPTQ, AWQ and GGUF use related, differing quantisation designs; this lab's simple rounding is not an implementation of all of them. `bitsandbytes` supports loading several low-bit formats, including NF4 used with QLoRA.

**Lab measurement:** {{QUANT_NOTES}}

![Size and perplexity after quantisation (lab)](fig:lab_quant)

### Batching and serving engines

Generation often spends substantial time reading model weights. **Batching** shares those reads across several sequences. **Continuous batching** lets new requests join as others finish. vLLM's **PagedAttention** (Kwon et al., 2023) manages attention-cache memory in pages, reducing wasted space. Measure aggregate throughput and individual waiting time: a larger batch can improve one while worsening the other.

**Lab measurement:** {{BATCH_NOTES}}

![Throughput vs batch size (lab)](fig:lab_batch)

### Caching and other savings

* **KV cache** (week 5) and **prefix caching**: reuse computation for shared prompt prefixes; providers discount cached input tokens.
* **Response caching**: identical requests answered from a cache (milliseconds instead of seconds in the lab).
* **Speculative decoding**: a small draft model proposes tokens that the large model verifies in parallel.
* **Right-sizing**: a fine-tuned or distilled small model is often the biggest saving.

**Weight storage examples.** A 0.5-billion-parameter model in 16-bit uses about 1 GB for weights. A 7–8-billion-parameter model in 4-bit uses about 3.5–4 GB before quantisation metadata. A 70-billion-parameter model in 16-bit uses about 140 GB for weights. A T4 has 16 GB of GPU memory, but fitting the weights is only the first check. Leave room for the runtime, intermediate calculations and the KV cache. Longer prompts and larger batches use more memory. Measure the complete workload before deciding that a model fits.

## 4. Building applications and interfaces

**Start here.** FastAPI exposes Python functions as HTTP endpoints. Gradio wraps a Python handler in a browser UI. Docker records the app environment. The proposed architecture diagram is a design target: the notebook's API and Gradio routes are separate, and checks on one route do not protect the other automatically.

**Small worked example.** Send the same message through the API and the Gradio handler. Inspect which function receives each request and where checks and logs run. Compare the exported packaging scaffold before assuming it includes every notebook feature.

![Proposed integrated architecture of a small generative AI application](fig:app_architecture)

This figure shows a proposed integrated design. In the lab the two routes are separate: Gradio calls `respond` and `generate` directly, while the SDK and LangChain call the FastAPI endpoint. The endpoint does not inherit the checks or logging in `respond`. The exported `app.py` is also a smaller packaging scaffold; it omits the notebook's personal-data redaction, output check and metadata logging.

* **FastAPI** turns Python functions into a web API; Pydantic validates requests; interactive docs at `/docs`. Returning the OpenAI response format (with a `usage` block) makes the server usable by the OpenAI SDK and LangChain.
* **Gradio** builds interfaces from Python functions; `gr.ChatInterface(respond)` gives a full chat UI. `share=True` creates a temporary public link: anyone with it can use your quota. **Hugging Face Spaces** supports Gradio apps and secrets. Current compute-Space creation has plan requirements; check eligibility, hardware and quotas in the [official Spaces overview](https://huggingface.co/docs/hub/spaces-overview). Streamlit is an alternative.
* **Docker** can package code, exact dependency versions and the start command; the lab export contains broad version ranges that need tested pins; runs on laptops, Hugging Face Spaces, Google Cloud Run or Azure Container Apps. Pass secrets as environment variables at run time.
* **Versioning and release:** version the model, prompts, decoding settings and retrieval index; run the evaluation set in CI and compare with the current version; roll out gradually and keep a rollback path.

## 5. Monitoring, guardrails and comparison

**Start here.** Monitoring records what happens after requests arrive: timings, tokens, failures and versions. Guardrails are checks around model calls; they may miss attacks or block harmless input. Use repeated test cases and inspect both kinds of mistake. Monitoring complements pre-release evaluation rather than replacing it.

**Small worked example.** In 100 requests, a p95 latency of 4 seconds means about 95% finished within 4 seconds. A mean of 1 second can still hide several long waits. Count blocked and allowed requests separately when interpreting the log.

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

**Guardrails in layers:** check input length and permissions; redact supported personal-data patterns; restrict tool access; validate output fields and apply relevant safety checks; send uncertain or consequential cases to a person; log and retest. Simple regular expressions miss varied attacks and data forms. Safety classifiers and libraries such as NeMo Guardrails or Guardrails AI add checks, but also need task-specific evaluation.

**LLMOps** means operating language-model applications through testing, versions, releases and monitoring. A **trace** records steps, timings and token use for a request. OpenTelemetry is a standard for telemetry; Langfuse, LangSmith, MLflow and cloud dashboards provide related inspection tools. Choose tools according to the information you need and protect any recorded personal data.

| Need | Good choice | Why |
|---|---|---|
| Best quality, quick prototype | Cloud API (check available tier) + Gradio | No infrastructure; strongest models |
| Private data, no budget | Ollama or Transformers locally | Data stays on the machine |
| Fine-tuned model for a demo | Hugging Face Spaces or Colab + Gradio | Check account requirements, hardware availability and cost |
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

* **Transparency:** identify AI interaction clearly. Article 50 distinguishes interaction notices, provider marking and deployer disclosure; exceptions apply. Most duties apply from 2 August 2026. Systems placed on the market before that date have until 2 December 2026 for the Article 50(2) marking duty only. [Official Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50), [Commission timing guidance](https://digital-strategy.ec.europa.eu/en/factpages/quick-facts-transparency-rules-ai-systems).
* **Data protection:** know where prompts go and how long providers keep them; minimise and redact; data-processing agreements; regional hosting where required (GDPR).
* **Security:** OWASP Top 10 for LLM applications (prompt injection, sensitive information disclosure, improper output handling, excessive agency, unbounded consumption, …); authentication and rate limiting; secrets management.
* **Accountability:** version and log which model and prompt produced each answer; keep an escalation path to a human.
* **Fairness and quality over time:** monitor quality by user group and topic; models and data drift.
* **Environment and cost:** right-size models, cache, batch and switch off idle resources.

# Lab guide and answers

**Runtime:** Colab T4 GPU or CPU (slower; the notebook works unchanged). On the 4 GB GTX 1650 laptop GPU used to check this pack (9 October 2026), the full notebook ran in about 3 minutes with the models and data already downloaded. Model: Qwen2.5-0.5B-Instruct (SmolLM2-135M in the automated smoke test). Cloud API keys (`GEMINI_API_KEY`, `HF_TOKEN`) are optional and read only from Colab Secrets. **Hand-in:** latency, cost, quantisation, batching and monitoring tables; app link or screenshot; three written answers.

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
* *Packed loading skipped on CPU:* this lab enables its optional route only on CUDA. Current bitsandbytes supports additional backends; check its hardware requirements.
* *Cloud API errors:* check the key is stored as a Colab Secret with notebook access enabled; model names change, so list models with `client.models.list()`; free tiers have strict rate limits (HTTP 429).
* *Different numbers from the slides:* latency depends on hardware and load; the patterns (TTFT ≪ total, batching gains, int8 ≈ lossless) should match.

# Practice questions with model answers

## Multiple choice

1. A client using the OpenAI SDK can call a vLLM server by changing: **(a)** the prompt format of each request; **(b)** the tokenizer the client loads; **(c)** `base_url`, `api_key` and `model`; **(d)** nothing, as the SDK finds the server. *Answer: (c).*
2. Which mostly determines total latency for long answers? **(a)** the number of output tokens; **(b)** prefill of the prompt; **(c)** network round-trip time; **(d)** the type of API key used. *Answer: (a).*
3. HTTP 429 means: **(a)** invalid key; **(b)** content blocked; **(c)** server crash; **(d)** rate limit exceeded. *Answer: (d).*
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

# Detailed explanations behind the short slides

The slides show one idea at a time. Use the detail below for follow-up questions, technical terms and further study.

## Slide 1: Title

Welcome to week 11, deployment and frameworks. So far we have used models inside notebooks. This week we turn a model into something other people can use: an application with an interface and an API, running somewhere sensible, at an acceptable cost and speed, with guardrails and monitoring. Every number on the "real" slides was measured by running this week's lab in advance, on a small open model, so students can reproduce them on Colab.

## Slide 2: Outcomes

Six outcomes matching the descriptor's week 11: frameworks; API platforms; deployment options; building applications and interfaces; deploying and monitoring with standard tools; and comparison. The final outcome is the one examiners and employers care about: justifying a deployment decision with evidence about latency, cost and quality rather than habit. This week practises MIMLO 5, deploying generative AI with contemporary frameworks, together with MIMLO 3 and 4.

## Slide 3: Agenda

The plan follows the revised descriptor. Sections two and three contain real measurements from the lab: streaming latency, quantisation and batching. Take the break after section three. Sections four and five show the code students will write in the lab and how to monitor and protect the application.

## Slide 4: Warm-up

Two minutes in pairs. Collect answers under the four headings on the slide; together they are the agenda for today. Students often mention speed first. Push them on cost: a public link to a paid API with no limits can produce a large bill overnight. Push them on keys: API keys pasted into notebooks or GitHub are routinely stolen by automated scanners. And on operations: without logs you do not know that the app is failing until users complain.

- Your notebook chatbot works well for you. Tomorrow 500 students will use it. What could go wrong?

- Speed: many requests at once, long answers, slow first token.

- Cost: who pays for the tokens or the GPU? Is there a budget cap?

- Safety and privacy: prompt injection, personal data, harmful answers, leaked API keys.

- Operations: crashes, model updates, knowing when something breaks.

## Slide 5: Frameworks and the tool stack

Section one maps the tool landscape into layers, compares the main frameworks named in the descriptor, introduces the OpenAI-compatible API pattern that connects them, and discusses when an orchestration framework helps and when plain code is better. Ask students which of these tools they have already used: most know a chat assistant, few have called an API directly. About twenty minutes.

- Which tools exist, and which do you need?

## Slide 6: The generative AI stack

Think in layers. At the bottom is hardware: cloud GPUs and TPUs, servers, laptops and phones with neural processing units. Models are proprietary, reached only through APIs, or open-weight, downloaded from the Hugging Face Hub. The access and serving layer is how you run a model: cloud APIs, self-hosted serving engines such as vLLM, or local runners such as Ollama. Orchestration frameworks compose prompts, tools, retrieval and agents. At the top are applications and interfaces. Running alongside every layer is LLMOps: logging, tracing, evaluation, guardrails and versioning.

## Slide 7: Frameworks and platforms at a glance

Walk through the table by layer. Hugging Face is the hub for open models and also offers hosted inference through partner providers. OpenAI's API and Microsoft's Azure OpenAI serve GPT models; Azure adds enterprise features such as private networking, regional data residency and compliance certifications, which matter to universities and companies. Google's Gemini API has a free tier through AI Studio, and Gemini Enterprise Agent Platform (formerly Vertex AI) is its enterprise route. vLLM is the most widely used open-source serving engine for GPUs. Ollama makes local models easy. LangChain, LangGraph and LlamaIndex are orchestration frameworks, used heavily next week.

| Tool | Layer | Use it for |
| --- | --- | --- |
| Hugging Face | Models and hosting | Find models and datasets; run Transformers code or call a host |
| OpenAI / Azure OpenAI | Provider API | Request hosted models; compare endpoint, region and data terms |
| Gemini API / Gemini Enterprise Agent Platform | Provider API | Request hosted Gemini models; check model features and limits |
| vLLM / SGLang / TGI | Model server | Run open models on your own GPUs for many requests |
| Ollama / llama.cpp / LM Studio | Local runner | Run suitable compressed open models on your device |
| LangChain / LangGraph | Request coordination | Connect model calls, tools and tracked workflow steps |
| LlamaIndex | Retrieval pipeline | Load, index and search documents for RAG (week 12) |

## Slide 8: One client, many back ends: the OpenAI-compatible API

Several providers and serving tools accept the basic OpenAI chat-completions format. Examples include compatible endpoints from Gemini, Hugging Face, vLLM and Ollama. The lab builds a small compatible endpoint with FastAPI. Reuse the basic client pattern by changing its base URL, key and model name. Then check supported fields, streaming behaviour and model features: compatibility does not mean every endpoint has identical capabilities. Provider-specific features may need a native SDK.

## Slide 9: Plain SDK or orchestration framework?

Students often assume they must use LangChain. Frameworks are useful when you need their connectors to data sources, retrievers, state management for multi-step agents, or tracing integrations, and LangGraph in particular is designed for controllable agent workflows. But they add abstraction layers, and their APIs change quickly, which makes debugging harder. Guides from Anthropic and OpenAI on building agents recommend starting with direct API calls and simple code, and adding frameworks only when they solve a real problem. In the lab, students write the plain version first, then see LangChain drive the same model.

| Plain SDK + your own code | Framework (LangChain, LangGraph, LlamaIndex) |
|---|---|
| Few moving parts; easy to debug and test | Ready-made connectors, retrievers, memory, tools |
| Full control over prompts and retries | Stateful graphs, retries, human-in-the-loop for agents |
| Best for simple apps, classification, extraction | Tracing integrations (e.g. LangSmith) |
| Anthropic and OpenAI guides recommend starting simple | Extra abstraction; APIs change quickly |

An SDK is a library for calling a service. Start with it and plain code; add a framework for a specific need.

## Slide 10: Quiz

**Question.** Student data must stay on university servers. What fits the existing compatible client?

- **A.** Self-host a suitable compatible model; change URL and re-test
- **B.** Rewrite every component for a new local framework
- **C.** Send data to the external API, then request deletion
- **D.** Switch to a larger external model with better accuracy

Answer: A. Because vLLM exposes an OpenAI-compatible API, the application code stays the same; only the base URL, key and model name change. You must then re-evaluate quality, because the open model will behave differently, and plan capacity for the GPUs. B is unnecessary work. C does not satisfy a requirement that data never leaves. D changes nothing about where data goes.

## Slide 11: API platforms

Section two covers what every developer must know about API platforms: authentication and key management, how tokens determine cost, rate limits and error handling, and streaming. We look at real latency measurements and a cost calculation. The same skills apply to OpenAI, Azure, Google and Hugging Face. About twenty-two minutes.

- Keys, tokens, pricing, rate limits and streaming

## Slide 12: What an API platform gives you, and what it expects

Six concepts cover most API platforms. Authentication uses API keys, or for enterprise clouds, identities such as Microsoft Entra ID, which avoid long-lived secrets. Pricing is per million tokens, with output tokens usually several times more expensive than input, so long answers cost more than long prompts. Rate limits cap requests and tokens per minute. Streaming sends tokens as they are generated. Batch APIs process large offline jobs asynchronously at a discount, and prompt caching discounts repeated prefixes such as long system prompts. Structured outputs and tool calling were covered in week 7 and return in week 12.

- **Authentication:** An API key identifies a caller. Use separate credentials for each app and environment.

- **Tokens and pricing:** Providers may charge separately for input, output and media. Read the current price units.

- **Rate limits:** The provider caps requests or tokens over time. HTTP 429 usually means "too many requests".

- **Streaming:** Return parts of the answer as they are generated, so users can start reading sooner.

- **Batch and caching:** Batch services process offline jobs together. Prefix caches may reuse repeated input; support and pricing vary.

- **Structured output and tools:** Schemas describe data fields; function calling describes tool requests (weeks 7 and 12).

## Slide 13: Keeping API keys safe

Key security is a practical skill every student needs. Keys committed to public repositories are found by automated scanners very quickly and used to run up bills. Store keys in Colab Secrets, environment variables or a cloud secrets manager. Never embed a key in front-end code, because anyone can read it in the browser; the front end must call your own back end, which holds the key. Use separate keys per project with spending limits and alerts. If a key leaks, revoke it first and investigate afterwards. The lab reads keys only from Colab Secrets.

- Keep access keys out of code, notebooks and Git. Treat a key like a password.

- Load keys from Colab Secrets, environment variables or a secrets manager.

- For an application, the server holds the provider key; the browser talks to your server.

- Give each app only the access it needs; set spending alerts and limits where supported.

- If a key leaks, revoke it immediately and review its use and billing.

## Slide 14: Anatomy of a request: where the time goes

Break latency into parts. Network and authentication add tens of milliseconds. Under load, requests wait in a queue. Prefill processes the whole prompt in parallel, so it grows with prompt length, which matters for RAG with long contexts. Then decoding generates one token at a time, which is why output length dominates total time. Time to first token is network plus queue plus prefill; total latency is roughly time to first token plus the number of output tokens divided by the decoding speed. Streaming shows the first words early, which greatly improves perceived speed.

- **TTFT** means time to first token: how long before the first text appears.

- **Decoding** creates further tokens in sequence. Longer answers take more time.

- Streaming shows each part as it arrives; it does not guarantee faster model computation.

## Slide 15: Measured: streaming latency of a small open model

This is the lab's measurement for Qwen2.5-0.5B-Instruct. The dark part of each bar is time to first token, the light part the rest of the answer. The numbers are quoted in the bullet. Time to first token is a small fraction of the total, which is why streaming is standard in chat interfaces. Hosted frontier models run on far larger hardware, but they are also much larger, so the same pattern holds.

- {{LAT_NOTES}}

- Users see text after the first token; without streaming they wait for the whole bar.

## Slide 16: Estimating cost: 500 students, 20 requests a day, one month

Walk through the arithmetic. Five hundred students times twenty requests a day times thirty days is three hundred thousand requests. Each has eight hundred input and three hundred output tokens. For the small model, that is three hundred thousand times eight hundred tokens at ten cents per million, plus three hundred thousand times three hundred tokens at forty cents per million: twenty-four plus thirty-six, sixty dollars. The same formula gives $300 for the mid-size model and $1,380 for the frontier model. A GPU running all month costs about four hundred and thirty dollars before staff time. The prices are illustrative; the method is what students must learn.

| Option (illustrative prices) | Price per 1M tokens (in / out) | USD per month |
| --- | --- | --- |
| Small hosted model | $0.10 / $0.40 | $60 |
| Mid-size hosted model | $0.50 / $2.00 | $300 |
| Frontier hosted model | $2.00 / $10.00 | $1,380 |
| Self-hosted small GPU, always on | $0.60 per hour | $432 + staff time |

300,000 requests × (800 input + 300 output tokens). Prices are illustrative: always check the providers' current pricing pages.

Method

## Slide 17: Streaming and retries in code

Two patterns every application needs. Streaming: set stream to true and print each chunk's new text as it arrives. Retries: when the provider returns HTTP 429, too many requests, wait and retry, doubling the wait each time and adding random jitter so that many clients do not retry at the same moment. Always set a maximum number of output tokens, which caps both cost and latency, and a timeout. The official SDKs have built-in retries for some errors, but you should understand and configure them.

```python
from openai import OpenAI
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
# Read API_KEY from a secrets store.
stream = client.chat.completions.create(
    model=MODEL, messages=messages,
    stream=True, max_tokens=300)
for chunk in stream:
    text = chunk.choices[0].delta.content or ""
    print(text, end="")
# Retry 429 with capped backoff and jitter.
```

- Cap answer length and set a timeout for the request.

- **Back-off** means waiting longer between retries. **Jitter** adds a small random delay so clients do not all retry together.

- Retry only a limited number of times; report failures clearly.

## Slide 18: Deployment options and efficient serving

Section three compares the three deployment options, then covers the three main efficiency techniques named in the descriptor: quantisation, batching and caching, with real measurements. The key message: cost and speed are design choices that can be measured, and every efficiency gain must be checked against quality. About twenty-five minutes; take the break afterwards.

- Cloud, self-hosted or local, and how to make it cheaper and faster

## Slide 19: Cloud API vs. self-hosted vs. local

There is no universally best option. A cloud API provides hosted models and shifts model-server operation to the provider. Check its quality, charges, region and data terms. Self-hosting gives your team control over compatible open models, but you pay for hardware, electricity and staff. High steady use can improve its economics. Local processing can keep data on a configured device; check network features, licences and resource costs. Some applications combine these routes, after testing quality and access requirements.

|  | Cloud API | Self-hosted (e.g. vLLM on GPUs) | Local (e.g. Ollama) |
| --- | --- | --- | --- |
| Model choice | Provider's supported models | Compatible open models you operate | Models fitting the device |
| Cost | Usage charges; check minimums and terms | GPU and staff costs; utilisation matters | Hardware, electricity and setup time |
| Data | Sent to the provider; check region and terms | Kept within your configured infrastructure | Kept on the configured device |
| Who operates it | Provider plus your application team | Your team manages capacity and updates | Device owner manages installation |
| Main limits | Quotas, service changes, model availability | Capacity, staff skills and maintenance | Memory, speed and model quality |

## Slide 20: Quantisation: fewer bits per weight

Quantisation stores weights with fewer bits. Memory is roughly the number of parameters times bits divided by eight: a seven-billion-parameter model needs fourteen gigabytes in sixteen-bit, seven in eight-bit and under four in four-bit, which is the difference between needing a data-centre GPU and running on a laptop. The absmax method scales each group of weights so the largest maps to the largest integer, rounds, and stores the integers plus one scale per group. Smaller groups keep more precision. Formats such as GPTQ, AWQ and GGUF, used by vLLM and Ollama, refine this idea.

Read the weight-memory estimate as: number of parameters multiplied by bits per weight, divided by eight to convert bits into bytes. This estimate excludes activations, cache and runtime overhead. The complete equation appears in Section 3 above.

Read the rounding example as: choose a scale from the largest absolute weight and the available integer levels; round each weight divided by that scale; multiply back to reconstruct an approximate weight. The lab keeps these values in floating-point storage.

- Weight storage is about parameters x bits/8. For 7 B weights: 16-bit approx. 14 GB; 8-bit approx. 7 GB; 4-bit approx. 3.5 GB before overhead.

- In **absmax quantisation**, choose a scale s, divide weights by s, round to integers q, and reconstruct with q x s.

- Smaller weight groups get separate scales, often preserving detail at extra storage cost. GPTQ, AWQ and GGUF use different related designs.

| Symbol | Meaning |
|---|---|
| b | bits used for each stored weight |
| s | conversion scale for a weight group |
| q | rounded integer stored in memory |
| $\hat{w}$ | approximate weight reconstructed for computation |

## Slide 21: Measured: size and quality after quantisation

The lab rounds weights in linear layers, excluding the output language-model head, then reconstructs floating-point values for inference. It measures perplexity on one paragraph. The displayed storage sizes are estimates of hypothetical packed weights; this demonstration does not store four-bit tensors. Compare the measured quality for eight-bit, row-scaled four-bit and grouped four-bit rounding. Separate group scales can preserve detail at additional storage cost. One paragraph is only a quick check; rerun the actual task evaluation before deployment.

- {{QUANT_NOTES}}

- Always re-evaluate on your own task after quantising.

## Slide 22: Measured: batching raises throughput

Generating for one request at a time leaves most of a GPU idle, because decoding is limited by memory bandwidth, not arithmetic. Processing several requests together reuses each weight read for many sequences. The lab measures throughput for batch sizes one to eight; the numbers are in the bullet. Production engines go further with continuous batching, adding and removing requests at every step rather than waiting for a whole batch to finish. The trade-off: larger batches increase throughput but can increase each user's latency.

- {{BATCH_NOTES}}

- **Batching** runs several requests together. **Continuous batching** lets new requests join when space becomes available.

- Higher total throughput may still increase an individual user's waiting time.

## Slide 23: Serving engines and caching

vLLM's key idea, PagedAttention, manages the key-value cache in fixed-size pages, like an operating system's virtual memory, which reduces wasted memory and allows larger batches. Prefix caching reuses the KV cache for shared prompt beginnings, such as a long system prompt, and API providers pass on the saving as discounted cached input tokens. Response caching answers repeated questions instantly. Speculative decoding speeds up generation by letting a small model draft tokens that the large model verifies in parallel. Often the largest saving is simply using a smaller, fine-tuned model.

- **KV cache** stores earlier attention keys and values so the model need not recompute them.

- vLLM's **PagedAttention** manages this memory in small pages; continuous batching shares GPU work (Kwon et al., 2023).

- **Prefix caching** reuses computations for identical input starts. **Response caching** reuses a completed answer.

- **Speculative decoding** lets a small model propose tokens for a larger model to verify.

- Test a smaller adapted or distilled model: it may reduce cost while meeting your task's quality target.

## Slide 24: What fits where? (weights only)

The figures estimate weight storage, not total runtime memory or a guaranteed model fit. A 0.5 B model at 16-bit needs about 1 GB for weights; 7–8 B at 4-bit about 4 GB before scales; 70 B at 16-bit about 140 GB. A T4 has 16 GB total, shared with attention caches, activations and framework overhead. Context length and batch size change the required runtime memory. Measure the actual configuration before recommending hardware.

- **About 1 GB:** 0.5 B weights at 16-bit; extra memory is still needed.

- **About 4 GB:** 7–8 B weights at 4-bit, before runtime overhead.

- **16 GB:** T4 GPU capacity; what runs also depends on context and batch size.

- **About 140 GB:** 70 B weights at 16-bit, before caches and runtime memory.

- These are weight-storage estimates, not guarantees a model will run. Add attention caches, activations and framework overhead.

## Slide 25: Quiz

**Question.** An accurate self-hosted model struggles with 50 concurrent users. What should you test?

- **A.** A higher temperature so replies finish sooner
- **B.** Continuous batching and quantisation; re-check quality
- **C.** A longer prompt with more detailed instructions
- **D.** Disable streaming so each reply is sent whole

Answer: B. With many concurrent users, a naive server processes requests one at a time, so queues grow. A serving engine with continuous batching raises throughput dramatically, and quantisation reduces memory so more requests fit in a batch. Quality must be rechecked after quantising. A changes randomness, not speed. C makes prefill slower. D does not change total time and makes perceived latency worse.

## Slide 26: Building applications and interfaces

Welcome back. Section four shows how the pieces fit together into an application: an interface for people, an API for other programs, guardrails and logging around the model, and a container for deployment. The code on these slides is what students write in the lab. About fourteen minutes.

- Gradio, FastAPI and Docker

## Slide 27: Proposed integrated application architecture

This is a proposed integrated design. A user interface or other client calls a back-end API, with input checks, model generation, output checks and logs inside that request path. In the lab the routes are separate: Gradio calls respond and generate directly, while the SDK and LangChain call the FastAPI endpoint. That endpoint does not inherit respond's guardrails or logs. The exported app.py is a smaller packaging scaffold without all notebook checks and logging. Students should identify the features that must be connected before deploying the integrated design.

## Slide 28: An OpenAI-compatible endpoint with FastAPI

FastAPI turns a Python function into a web API. Pydantic models validate the request body automatically, rejecting malformed requests, and FastAPI generates interactive documentation. By returning the OpenAI response format, including the usage block with token counts, our server works with the OpenAI SDK and LangChain unchanged, as students verify in the lab. In production you would add authentication, request size limits, timeouts and streaming, and run the server behind a process manager.

```python
@api.post("/v1/chat/completions")
def chat(req: ChatRequest):
    text, n_in, n_out = generate(
        req.messages, req.max_tokens)
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": text}}],
        "usage": {"prompt_tokens": n_in,
                  "completion_tokens": n_out,
                  "total_tokens": n_in + n_out}}
```

- **FastAPI** exposes a Python function as a web endpoint; Pydantic checks the incoming fields.

- This endpoint uses a basic compatible chat format; check which features a client actually requires.

- The lab's endpoint and checked Gradio handler are separate routes; the diagrams explain both.

## Slide 29: A chat interface with Gradio

Gradio builds a web interface from a Python function; ChatInterface provides a complete chat UI in one line. The respond function wraps the model with an input check, logging and an output check. In Colab, share equals true prints a temporary public link, useful for demonstrations but anyone with the link can use it. For longer-lived demos, Hugging Face Spaces hosts Gradio apps, with secrets for API keys. Streamlit is a popular alternative for data-oriented apps.


Compute Spaces have account-plan requirements; limited free ZeroGPU eligibility is different from unconditional free CPU hosting. Current terms: https://huggingface.co/docs/hub/spaces-overview

```python
import gradio as gr

def respond(message, history):
    allowed, cleaned, reason = check_input(message)   # guardrail
    if not allowed:
        return f"Sorry, I can't help with that ({reason})."
    answer, n_in, n_out = generate(build_messages(cleaned, history))
    log_request(n_in, n_out, reason)                   # metadata only
    return check_output(answer)

demo = gr.ChatInterface(respond, title="Course assistant")
demo.launch(share=True)     # temporary public link from Colab
```

- **Gradio** builds a browser interface around a Python function.

- Hugging Face Spaces supports Gradio hosting; account-plan requirements, hardware, quotas and costs vary.

- A public demo link allows others to use the app. Limit access and request volume as needed.

## Slide 30: Packaging and releasing: Docker, versions, CI

A Docker image packages code, dependencies and a start command. Record tested exact versions for repeatable deployment; the lab export supplies version ranges as a starting scaffold. A container reduces environment differences but still depends on compatible hardware, platform settings, network access and resources. Pass secrets at run time. Version the model, prompts, decoding settings and retrieval index as well as the code. Re-run evaluations after changes, release gradually and keep a rollback path.

- A **Docker image** bundles app code, named dependency versions and a start command.

- Pass credentials at run time; keep them out of the image.

- Record model, prompt, generation settings, data and search-index versions.

- **CI** means automated checks when code changes. Run the evaluation set before release.

- Release gradually and keep the previous version so you can **roll back** if errors increase.

## Slide 31: Monitoring, guardrails and comparison

The final section covers what to monitor, a real monitoring summary from the lab, layered guardrails, the LLMOps tooling for tracing and evaluation, a comparison of deployment choices, and a responsible deployment checklist. It answers the question every client asks after launch: how do we know it is still working? About twenty-one minutes.

- How do you know it is working, and safe?

## Slide 32: What to monitor

Monitoring for generative AI extends normal web monitoring. Performance uses percentiles because averages hide slow requests: p95 means ninety-five percent of requests are faster. Cost must be tracked per user and per day, with alerts. Reliability includes rate-limit errors and provider outages. Quality needs signals such as user feedback, sampled reviews and regression tests, because model and data drift can silently degrade answers. Safety metrics track guardrail activity. Versions link every answer to the model and prompt that produced it, which is essential for debugging and accountability.

- **Performance:** Track first-token wait, total wait, timeouts and requests per second. p95 is the wait 95% of requests do not exceed.

- **Cost:** Count tokens by request, user and day. Compare spending with a budget.

- **Reliability:** Count failed requests, limit errors (429) and service outages.

- **Quality:** Review a sample of answers and repeated test questions; watch for changes in input patterns.

- **Safety:** Track blocked inputs, redactions, flagged outputs and attack attempts.

- **Versions:** Record which model, prompt and document index produced each answer.

## Slide 33: Measured: a monitoring summary from the lab app's logs

The lab sends simulated traffic through the app, including a prompt-injection attempt and messages containing an e-mail address and a phone number. Each bar is one request: blue answered, orange answered after redacting personal data, red blocked. The dashed line marks the ninety-fifth percentile latency. The summary in the bullet is computed from the log file. Point out the design choice: logs store metadata, not message text, following GDPR data minimisation; if you need text for quality review, redact it and limit retention and access.

- {{MONITOR_NOTES}}

- Logs record metadata (latency, tokens, blocks, version), not raw personal data.

## Slide 34: Guardrails in layers

Guardrails work in layers because each layer can be bypassed. Input checks enforce length and rate limits, redact personal data and filter known injection patterns. The model setup limits damage: least-privilege tools mean an injected instruction cannot do much. Output checks run moderation classifiers, validate structured outputs and check for leaked secrets or policy violations. Humans handle sensitive or uncertain cases. Monitoring and red-teaming find new attacks. The lab's regex guardrails are deliberately simple; students discuss how to bypass them in question two.

- **Input:** length and rate limits, PII redaction, injection and topic filters.

- **Model / tools:** system prompt, least-privilege tools, safe decoding settings.

- **Output:** moderation classifier, schema validation, leak and policy checks.

- **Oversight:** escalation for sensitive or uncertain cases.

- **Monitor:** track blocks, red-team with test attacks, update the rules.

- Simple pattern rules catch some cases. Combine permission limits, format checks and tested safety classifiers; measure misses and false alarms.

## Slide 35: LLMOps tooling: tracing and evaluation

Tracing is essential once an application has several steps, as RAG and agent systems do next week: a trace shows which retrieval, prompt, tool call or model call caused a bad answer or a slow response. OpenTelemetry is the open standard; Langfuse is an open-source LLM observability platform; LangSmith integrates with LangChain; MLflow tracks experiments and traces; and cloud platforms such as Microsoft Foundry (formerly Azure AI Foundry) and Google's Gemini Enterprise Agent Platform (formerly Vertex AI) include monitoring and evaluation dashboards. Guardrail libraries and safety classifiers provide more robust checks than hand-written rules. Mention these as options, not requirements, for projects.

- **Tracing** records each request step, its timing and token use so you can locate failures.

- Example tools: OpenTelemetry, Langfuse, LangSmith, MLflow and cloud dashboards.

- **LLMOps** means operating language-model applications: versions, tests, deployment and monitoring.

- Use sampled human/model review and repeated test questions. Safety libraries support checks but do not guarantee safety.

## Slide 36: Comparison: choosing a deployment for a student project

The descriptor asks for a comparison, so end with a practical decision table for projects. For a quick prototype, compare hosted model quality, permitted data, quotas and cost, then use a Gradio interface where it fits. For private data without a budget, run a local model. To demo your own fine-tuned model, use Hugging Face Spaces or Colab. For many users with steady load, a serving engine on a GPU. For a reproducible hand-in, a Docker image with pinned versions. Remind students to justify their choice in the project report with measured latency, cost and quality.

| Need | Good choice | Why |
| --- | --- | --- |
| Quick prototype | Hosted API + Gradio | Little server setup; check costs and permitted data |
| Data must stay on device | Suitable local model | Check memory, speed, quality and configuration |
| Show an adapted model | Spaces or Colab + Gradio | Check hardware availability, duration and hosting cost |
| Many concurrent users | A server such as vLLM | Measure throughput and latency under realistic load |
| Repeatable hand-in | Docker + named dependency versions | Another person can reproduce the environment |

## Slide 37: Responsible deployment checklist

This checklist connects the week to responsible AI. Transparency is now a legal duty in the EU: users must be told they are interacting with an AI system. Data protection requires knowing where prompts go and how long they are kept. Security follows the OWASP Top 10 for LLM applications, which lists prompt injection, sensitive information disclosure and unbounded consumption among the main risks. Oversight and evaluation continue after launch. Finally, cost and energy: smaller, well-chosen models reduce both.

- **Transparency:** clearly tell users when they interact with AI; Article 50 includes duties and exceptions.

- **Data protection:** collect only needed information; set access, storage and deletion rules (GDPR).

- **Security:** protect credentials, check caller identity and cap requests; consult OWASP's LLM risks.

- **Oversight:** test before release, monitor use, and provide a clear path to a person.

- **Resources:** set budgets and alerts; choose a model size that meets the measured task need.

## Slide 38: Lab 11: deploy, measure and monitor a generative AI app

The lab covers both descriptor tutorials: exploring tools with local and cloud models, and building and deploying a small application with a web interface, measuring latency and cost and adding logging and guardrails. Cloud API keys are optional; everything else runs without accounts. Remind students to stop the public Gradio link at the end of the session and never to paste keys into the notebook.

- Stream from a local model; measure TTFT and output tokens per second.

- Expose a FastAPI endpoint; call it with the openai SDK and LangChain.

- Estimate monthly cost; implement capped retries with increasing delays.

- Simulate lower-bit weight storage; measure text loss, batching throughput and cache timings.

- Build a checked Gradio handler and metadata log; summarise it and write a Docker packaging example.

- **Runtime:** Colab T4 GPU or CPU (slower). About 3 min on a 4 GB laptop GPU after downloads.

- **Hand in:** notebook with latency, cost, quantisation, batching and monitoring tables, the app link or screenshot, and three answers.

## Slide 39: Summary

The six cards on the slide condense these eight summary points. Students should be able to draw the architecture of a small application, estimate its cost, explain where latency comes from, choose and justify a deployment option, and list what to monitor. Next week, the final technical week, we connect models to knowledge and tools: retrieval-augmented generation and agents, which use everything from today.

- An application has a model, a server, request-handling code and an interface.

- A compatible API can reuse basic client code; verify feature support.

- Protect keys and account for input/output tokens, quotas and retries.

- Measure first-token wait and generation speed separately.

- Hosting choices change cost, data location and maintenance work.

- Fewer weight bits, shared batches and caches may reduce cost; re-check quality.

- Gradio builds the UI, FastAPI serves requests, and Docker packages the environment.

- Record versions, monitor failures and apply several checked protections.

## Slide 40: Resources

Microsoft's Generative AI for Beginners has lessons on building and securing applications. The Hugging Face Inference Providers and Gemini compatibility pages show the OpenAI-compatible pattern with free tiers. The vLLM documentation and Ollama cover self-hosted and local serving. Gradio and FastAPI tutorials cover the application layer. The OWASP Top 10 for LLM applications is the standard security reference. The textbook reading is Tunstall and colleagues, chapter 8, on making transformers efficient in production. See the source-verification report for the latest checks and access limitations.

- **Course:** [Microsoft: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners). Lessons on building and securing apps.
- **Reference:** [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index). OpenAI-compatible router to open models.
- **Reference:** [Gemini API: OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai). Use Gemini with the OpenAI SDK.
- **Reference:** [vLLM documentation](https://docs.vllm.ai/en/latest/). High-throughput serving.
- **Tool:** [Ollama](https://ollama.com/). Run open models locally.
- **Reference:** [Gradio quickstart](https://gradio.app/guides/quickstart). Interfaces in Python.
- **Reference:** [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/). Build APIs with Python.
- **Reference:** [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/). Security risks and mitigations.



# Full lecture code examples

The projected slides show short excerpts. Read these complete examples alongside the lab definitions and imports. The local lab binds port 0, lets the operating system choose a free localhost port, and reads that same server socket before configuring clients.

## Streaming and retries in code

```python
import random
import time

from openai import OpenAI, RateLimitError
client = OpenAI(base_url=BASE_URL, api_key=get_secret("API_KEY"))

stream = client.chat.completions.create(
    model=MODEL, messages=messages, stream=True, max_tokens=300)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

def call_with_retry(fn, max_retries=5, base=0.5):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RateLimitError:                  # HTTP 429
            if attempt == max_retries:
                raise
            time.sleep(base * 2**attempt * (1 + random.random() / 2))
```

## An OpenAI-compatible endpoint with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 128

api = FastAPI()

@api.post("/v1/chat/completions")
def chat(req: ChatRequest):
    text, n_in, n_out = generate(req.messages, req.max_tokens)
    return {"object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                      "total_tokens": n_in + n_out}}
```
