# Lecture plan at a glance

This final technical week connects models to knowledge and tools (MIMLO 3, 4, 5). Students should leave able to explain why RAG is needed, build and evaluate a RAG pipeline (chunking, embeddings, vector stores, hybrid retrieval, re-ranking, grounded generation with citations), describe advanced RAG, explain the components and loop of an agent, distinguish workflows from agents, explain MCP, and design guardrails against prompt injection. Results on the "real" slides come from running this week's lab in advance (`curriculum/assets/week_12/`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | Ask a chatbot about this module's late policy. |
| 5–20 min | 5–9 | 1 · Why RAG? | LLM limitations; RAG vs fine-tuning vs long context; pipeline; quiz. |
| 20–35 min | 10–14 | 2 · Indexing | Embeddings; chunking; vector stores; ANN and filters. |
| 35–65 min | 15–23 | 3 · Retrieval and evaluation | Dense vs BM25; RRF and metrics; re-ranking; real retrieval and RAG results; advanced RAG; RAG evaluation; quiz. |
| 65–72 min | – | Break | |
| 72–102 min | 24–33 | 4 · Agentic AI | Components; ReAct loop; function calling; real trajectory; workflows vs agents; multi-agent; MCP; frameworks; quiz. |
| 102–115 min | 34–37 | 5 · Risks and guardrails | Indirect prompt injection (real test); layered guardrails; oversight. |
| 115–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Before section five, show the forum post from the lab corpus on screen and ask: "Would the agent obey this?" Take a vote, then show what the lab agent actually did and which line of code stopped the e-mail. Students remember that guardrails live in code, not in the prompt.

<!-- pagebreak -->

# Lecture notes

## 1. Why retrieval-augmented generation?

LLMs have four limitations for knowledge-intensive tasks: **static knowledge** (training cut-off), **hallucination** (fluent unsupported answers), **no access to private data**, and **no citations**. **Retrieval-augmented generation (RAG)** (Lewis et al., 2020) retrieves relevant passages at question time and asks the model to answer from them with citations. It reduces but does not eliminate hallucination.

| | RAG | Fine-tuning (week 8) | Long context |
|---|---|---|---|
| Adds knowledge | Yes, updated by re-indexing | Poorly; hard to update | Yes, for what fits |
| Citations | Yes | No | Possible, harder to check |
| Cost per query | Low (top-k passages) | Low once trained | High (every token every time) |
| Access control | Per document at retrieval | Mixed into weights | Per request |
| Best for | Large, changing knowledge bases | Behaviour, format, style | A few long documents |

![The RAG pipeline](fig:rag_pipeline)

**Indexing (offline):** load documents → chunk → embed → store vectors with metadata, plus a keyword index. **Answering (online):** embed the question → retrieve (dense + BM25) → re-rank → prompt with numbered sources → LLM answers with citations or "I don't know".

**Lab measurement (no retrieval):** the same model without retrieval answers almost none of the handbook questions correctly; it produces plausible guesses (hallucination).

## 2. Embeddings, chunking and vector databases

**Embeddings.** A retrieval embedding model is a **bi-encoder** trained contrastively (week 9) so that questions land near passages that answer them; retrieval is nearest-neighbour search by cosine similarity. Open models (BGE, E5, GTE) and API models (OpenAI, Gemini, Cohere) are common; the **MTEB** benchmark compares them. Changing the embedding model requires re-embedding everything.

**Chunking.** Split documents into pieces of a few hundred to about a thousand tokens with 10–20% overlap, preferably at headings and paragraphs. Too large dilutes embeddings and wastes context; too small loses context. Keep **metadata** (source, section, date, access level) for citations and filters.

![Chunking with overlap](fig:chunking)

| Option | Type | Typical use |
|---|---|---|
| FAISS | Library | Fast in-process search (lab) |
| Chroma | Lightweight vector database | Local apps with metadata filters (lab) |
| pgvector | PostgreSQL extension | Vectors next to business data |
| Qdrant, Weaviate, Milvus, Pinecone | Dedicated vector databases | Scale, hybrid search, multi-tenant |
| Azure AI Search, Vertex AI Vector Search | Managed cloud search | Enterprise RAG with security integration |

**Approximate nearest neighbour (ANN)** indexes (HNSW graphs, IVF clustering, product quantisation) trade a little recall for large speed-ups. **Metadata filters** restrict results; enforce access control at retrieval time, because anything retrieved can appear in an answer.

## 3. Retrieval, advanced RAG and evaluation

**Dense vs keyword.** Dense retrieval matches meaning (paraphrases); **BM25** matches words weighted by rarity (codes, names, rare terms). **Hybrid** search combines them, usually with **reciprocal rank fusion**:

$$\mathrm{RRF}(d) = \sum_{r} \frac{1}{k + \mathrm{rank}_r(d)}, \qquad k = 60$$

**Metrics.** Recall@k is the fraction of questions with a relevant chunk in the top k; mean reciprocal rank averages one over the rank of the first relevant chunk:

$$\mathrm{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\mathrm{rank}_q}$$

A question with no relevant result in the returned list contributes zero to MRR. In this lab relevance is checked using the source document ID: a chunk from the known target document counts as a hit.

**Re-ranking.** A fast first stage retrieves 20–100 candidates (recall); a **cross-encoder** reads question and passage together and keeps the best 3–5 (precision). Fewer, better passages mean shorter prompts, lower cost and fewer distractions.

**Lab measurement (retrieval):** {{RETRIEVAL_NOTES}}

![Retrieval quality by method (lab)](fig:lab_retrieval)

**Grounded generation.** Give the model numbered sources; instruct it to answer only from them, cite them, treat them as data (never as instructions), and say "I don't know" when the answer is missing. Check citations and numbers in code. **Placement matters:** with these rules only in the system message, the lab's 1.5 B model cited the right document for 1 question in 12; putting the same rules after the sources, next to the question, fixed it (and reduced needless "I don't know" answers).

**Lab measurement (RAG vs no retrieval):** {{RAG_NOTES}}

![RAG vs no retrieval (lab)](fig:lab_rag)

**Advanced RAG.**

* **Query rewriting / multi-query:** rewrite conversational or vague questions into search queries.
* **HyDE** (Gao et al., 2022): embed a hypothetical answer and search with it.
* **Contextual retrieval** (Anthropic, 2024): prepend chunk-specific context before embedding and BM25 indexing; reported roughly half as many failed retrievals, more with re-ranking.
* **GraphRAG** (Edge et al., 2024): knowledge graph plus community summaries for global questions ("main themes across all reports").
* **Parent–child / hierarchical retrieval:** retrieve small chunks, pass larger parent sections.
* **Agentic RAG:** an agent decides when and what to search and iterates.

**Evaluating RAG.** Evaluate retrieval and generation separately, so you know what to fix.

| Aspect | Question | How to measure |
|---|---|---|
| Retrieval quality | Did the evidence reach the prompt? | Recall@k, MRR, context precision/recall |
| Faithfulness (groundedness) | Is every claim supported by the context? | Claim-level check with NLI or an LLM judge (RAGAS; Microsoft Foundry RAG evaluators) |
| Answer relevance | Does the answer address the question? | LLM judge or human rating |
| Correctness | Is it right? | Reference answers |
| Citations and abstention | Correct sources? "I don't know" when appropriate? | Automatic checks; unanswerable questions |

> **Key idea:** If the right passage was not retrieved, no generator can answer faithfully: fix retrieval first.

## 4. Agentic AI

An **agent** is an LLM that uses **tools** in a **loop** to achieve a goal, with **memory** (conversation, scratchpad, long-term notes) and **planning** (decomposing and revising steps), under **oversight** (permissions, approvals, logs).

![The agent loop](fig:agent_loop)

**ReAct** (Yao et al., 2022) interleaves reasoning and acting: decide → call a tool → observe the result → continue, until the model answers or a step limit is reached.

**Function calling.** Each tool is described by a name, a description and a JSON schema of its arguments. The model proposes a structured call; **your code** parses it, checks it against an allowlist, validates the arguments, asks for approval if it has side effects, executes it and returns the result as a tool message. The model never executes anything itself.

**Lab measurement (agent):** {{AGENT_NOTES}}

![A real agent trajectory from the lab](fig:agent_trace)

**Workflows vs agents** (Anthropic, "Building effective agents"): workflows orchestrate LLM calls through fixed code paths: **prompt chaining**, **routing**, **parallelisation**, **orchestrator–workers**, **evaluator–optimiser**. Agents let the model direct its own steps. Prefer the simplest pattern that works; use agents for open-ended tasks whose steps cannot be predicted, with limits.

![Workflow patterns](fig:workflows)

**Multi-agent systems** split work among specialised agents (researcher, writer, reviewer) coordinated by an orchestrator or messages. Benefits: specialisation, parallelism, separate permissions. Costs: more tokens, harder debugging, cascading errors. Agent-to-agent protocols (e.g. A2A) standardise communication between agents.

**Model Context Protocol (MCP).** An open standard introduced by Anthropic in November 2024 and donated in December 2025 to the Agentic AI Foundation under the Linux Foundation. MCP **servers** expose tools, resources and prompts; MCP **clients** in assistants, IDEs and agent frameworks discover and use them, so each integration is written once. Install only trusted servers and apply least privilege.

![MCP: one protocol between applications and tools](fig:mcp)

| Framework | Strength |
|---|---|
| LangChain / LangGraph | Components; stateful agent graphs with checkpoints and human-in-the-loop |
| LlamaIndex | Data connectors, indexing, retrieval pipelines |
| OpenAI Agents SDK | Agents, tools, handoffs, tracing |
| Google Agent Development Kit (ADK) | Multi-agent systems with Gemini and other models |
| Microsoft Agent Framework / Semantic Kernel | Enterprise agents in Python and .NET |
| Hugging Face smolagents | Minimal agents with open models |

## 5. Risks and guardrails

**Prompt injection** is the top risk in the OWASP Top 10 for LLM applications. **Direct** injection comes from the user; **indirect** injection (Greshake et al., 2023) hides instructions in content the system reads: web pages, e-mails, documents, retrieved chunks. With tools, injected text can trigger harmful actions.

**Lab measurement (indirect injection):** {{INJECTION_NOTES}}

**Guardrails for RAG and agents:**

1. **Least privilege:** read-only tools by default; no access to data the task does not need.
2. **Allowlists and validation:** known tools only; check argument types, ranges and recipients.
3. **Human approval** for side effects: sending, paying, deleting, publishing.
4. **Limits and sandboxing:** step, time and cost limits; sandboxed code execution.
5. **Logging and monitoring:** audit trail of every tool call; alerts on unusual actions.

> **Key idea:** Assume injected instructions will sometimes be followed; design so that following them cannot cause serious harm.

**Human oversight and responsible use:** show sources and provide a route to a human; respect permissions and personal data in the index (GDPR); keep the index current and versioned; evaluate continuously, including red-team injection tests; disclose AI use (EU AI Act Art. 50); treat uses such as admissions or grading decisions as high-risk.

# Common misconceptions

| Misconception | Correction |
|---|---|
| "RAG eliminates hallucination." | It reduces it; models can still misread, mix or ignore sources. Measure faithfulness. |
| "Bigger chunks give the model more context, so they are better." | Large chunks dilute embeddings and waste context; tune size and overlap on your data. |
| "Embeddings beat keyword search." | BM25 wins on exact codes and names; hybrid search is usually best. |
| "If the answer is wrong, use a bigger LLM." | First check retrieval: if the evidence was not retrieved, no model can answer faithfully. |
| "An agent is just a chatbot with a longer prompt." | An agent runs a loop and takes actions through tools, which changes the risk profile. |
| "Always use an agent framework / multi-agent system." | Most tasks are best served by simple workflows; agents add cost and unpredictability. |
| "A good system prompt stops prompt injection." | Models cannot reliably separate instructions from data; guardrails must be in code and permissions. |
| "MCP makes tools safe." | MCP standardises connections; each server is code with access to data, so trust and permissions still matter. |

# Responsible AI lens: knowledge, actions and accountability

* **Accuracy and verifiability:** always show sources; abstain when evidence is missing; test with unanswerable questions.
* **Data protection and access control:** indexes can expose personal or confidential data to anyone who can query them; enforce permissions at retrieval; remove personal data that is not needed (GDPR).
* **Security:** indirect prompt injection, data exfiltration through tools, untrusted MCP servers; least privilege and human approval.
* **Accountability:** audit logs of retrieval and tool calls; clear ownership of the knowledge base and its updates.
* **Transparency and regulation:** disclose AI interaction (EU AI Act Art. 50); decisions about people in education or employment can be high-risk uses with strict duties.
* **Over-reliance:** users may trust cited answers without checking; design interfaces that encourage verification.

# Lab guide and answers

**Runtime:** Colab T4 GPU (≈ 15 min) with Qwen2.5-1.5B-Instruct, BGE-small embeddings and an MS MARCO MiniLM cross-encoder; on CPU the notebook uses Qwen2.5-0.5B-Instruct. No API keys are needed. The handbook is fictional, so the model cannot have memorised it. **Hand-in:** retrieval and RAG tables, agent trajectories and audit log, three written answers.

**Simulation boundaries:** `approve` uses a destination-domain rule rather than asking a person, and `send_email` only returns a string; no email is sent. `AUDIT` records calls requiring simulated approval, while the trajectory records executed tool steps. The exported MCP server is an incomplete scaffold with a `retrieve_passages` placeholder; connect a working retrieval implementation and load its index before running it.

## TODOs

* **TODO 1** `chunk`: accumulate words until the size limit, emit the chunk, step back about `overlap` characters (at a word boundary) for the next start; guarantee progress.
* **TODO 2** `rrf`: for each ranking, add `1 / (k + rank)` to each index's score (rank starting at 1); return indices sorted by score, highest first.
* **TODO 3** `execute`: reject tools not in `TOOLS`; for side-effect tools call `approve(call)` and return `"BLOCKED: human approval denied"` if refused; run the tool inside `try/except` and return errors as text.

## Measured results (instructor run)

{{RESULTS_NOTES}}

## Model answers to the written questions

1. **Retrieval methods and RAG vs no RAG.** Without retrieval the model guesses (hallucination) and almost never answers correctly; RAG raises accuracy sharply. BM25 is strong on exact terms (module code, "GPU hours"), dense retrieval on paraphrases ("hand in after the deadline"); hybrid fusion and re-ranking usually put the right chunk first. Diagnose each failure: if the gold document was not in the top k, it is a retrieval failure (chunking, hybrid search, query rewriting, larger k); if it was retrieved but misused, it is a generation failure (stronger model, clearer prompt, fewer and better chunks). Rigorous faithfulness evaluation decomposes answers into claims checked against the context (RAGAS faithfulness, groundedness evaluators), plus answer relevance, context precision/recall, a larger labelled set with unanswerable and multi-hop questions, and human spot checks.
2. **Agent trajectories and injection.** For the grade task a capable model searches the handbook for the weights, calls the calculator with 0.6 × 72 + 0.4 × 64 and answers 68.8; small models may skip tools or stop early. For the forum task the retrieved post contains an instruction to e-mail grades externally; a model may or may not follow it, because it cannot reliably separate data from instructions, so the system prompt is not a guarantee. The protections demonstrated in code are the allowlist, the simulated e-mail tool's limited access, and a simulated approval policy that denies external addresses. The approval audit records such an attempt if it occurs; a real action requires a real review mechanism.
3. **Production design.** A fixed RAG workflow (hybrid retrieval, re-ranking, citations, abstention) fits factual student questions better than an autonomous agent: cheaper, faster, predictable and testable; add single tool calls only where needed. Use agents for open-ended multi-step tasks, with step, tool and permission limits. Evaluation: labelled real questions (paraphrases, multi-hop, unanswerable) with retrieval and answer metrics run after every change, plus sampled human review and feedback. Risk controls: untrusted-content handling and human approval for actions; access control and personal-data exclusion in the index; versioned, current index with visible sources; logging, monitoring and AI disclosure.

## Troubleshooting

* *Agent never calls a tool:* smaller models (0.5 B on CPU) often answer directly; use the GPU and the 1.5 B model, or a hosted model through the OpenAI-compatible pattern from week 11.
* *`chromadb` warnings about telemetry:* harmless.
* *Out of memory on a small GPU:* move the embedder and re-ranker to CPU (`device="cpu"`).
* *Different numbers from the slides:* generation is deterministic (greedy), but model and library versions change outputs; the patterns should match.

# Practice questions with model answers

## Multiple choice

1. RAG primarily addresses: **(a)** slow decoding; **(b)** static knowledge and lack of private data; **(c)** tokenisation errors; **(d)** GPU memory. *Answer: (b).*
2. BM25 is typically better than dense retrieval for: **(a)** paraphrased questions; **(b)** exact product codes; **(c)** cross-lingual search; **(d)** images. *Answer: (b).*
3. A cross-encoder re-ranker: **(a)** embeds question and passage separately; **(b)** reads question and passage together to score relevance; **(c)** generates answers; **(d)** chunks documents. *Answer: (b).*
4. In function calling, who executes the tool? **(a)** the LLM; **(b)** the application code; **(c)** the vector database; **(d)** the tokenizer. *Answer: (b).*

## Short answer

1. **Compute RRF scores (k = 60) for chunk X ranked 1st by BM25 and 4th by dense, and chunk Y ranked 2nd by both.** (3 marks) *Model answer:* X: 1/61 + 1/64 = 0.01639 + 0.01563 = 0.03202; Y: 2/62 = 0.03226; Y ranks slightly higher.
2. **Recall@3 and MRR:** first relevant chunk at ranks 1, 3 and 5 for three questions. (3 marks) *Model answer:* recall@3 = 2/3; MRR = (1 + 1/3 + 1/5)/3 = 0.511.
3. **Explain indirect prompt injection and two guardrails that do not depend on the model.** (4 marks) *Model answer:* instructions hidden in retrieved or read content that the model may follow; guardrails: least-privilege tools, allowlists and argument validation, human approval for side effects, sandboxing and limits, audit logs.
4. **When would you choose a workflow over an agent?** (3 marks) *Model answer:* when steps are known in advance (e.g. retrieve–answer–check), for predictability, testability, cost and latency; agents only when the path cannot be fixed.

## Exam-style question

**"Design a retrieval-augmented assistant for a hospital's internal clinical guidelines. Describe the indexing and retrieval pipeline, how you would evaluate it, and the risks and safeguards, including whether an agent is appropriate."** (20 marks)

*Marking guide:* pipeline (chunking by section, embeddings, hybrid retrieval, re-ranking, metadata and versioning) (6); grounded generation with citations and abstention (2); evaluation of retrieval, faithfulness and correctness with clinicians (5); risks: outdated guidance, hallucination, access control and patient data, injection, over-reliance (4); justified choice of workflow vs agent, human oversight and regulatory considerations (3).

# Glossary

| Term | Meaning |
|---|---|
| Retrieval-augmented generation (RAG) | Answering from retrieved passages supplied in the prompt |
| Chunk | Piece of a document indexed and retrieved as a unit |
| Embedding model (bi-encoder) | Maps text to vectors for similarity search |
| Vector database | Stores vectors with metadata and supports similarity search |
| ANN (HNSW, IVF) | Approximate nearest-neighbour search structures |
| BM25 | Keyword ranking function weighting rare matching terms |
| Hybrid search | Combining dense and keyword retrieval |
| Reciprocal rank fusion (RRF) | Combining rankings by summing 1/(k + rank) |
| Cross-encoder re-ranker | Model scoring a question–passage pair jointly |
| Recall@k / MRR | Retrieval metrics: evidence in top k / reciprocal rank of first hit |
| Faithfulness (groundedness) | Answer claims supported by the retrieved context |
| HyDE | Retrieval with a hypothetical generated answer |
| GraphRAG | RAG over a knowledge graph and community summaries |
| Agent | LLM using tools in a loop to reach a goal |
| Function (tool) calling | Model outputs structured calls to described functions |
| ReAct | Interleaving reasoning, actions and observations |
| Workflow | LLM calls orchestrated by fixed code paths |
| Model Context Protocol (MCP) | Open standard connecting AI applications to tools and data |
| Indirect prompt injection | Malicious instructions hidden in content the system reads |
| Human-in-the-loop approval | A person must approve an action before it runs |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Ozdemir, S. (2023) *Quick Start Guide to Large Language Models*: chapter 2, "Semantic Search with LLMs".

**Papers**

* Lewis et al. (2020) [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401); Yao et al. (2022) [ReAct](https://arxiv.org/abs/2210.03629); Es et al. (2023) [RAGAS](https://arxiv.org/abs/2309.15217); Edge et al. (2024) [GraphRAG](https://arxiv.org/abs/2404.16130); Gao et al. (2022) [HyDE](https://arxiv.org/abs/2212.10496); Greshake et al. (2023) [Indirect prompt injection](https://arxiv.org/abs/2302.12173)

**Guides and courses**

* [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) · [Anthropic: Contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
* [OpenAI: Function calling](https://developers.openai.com/api/docs/guides/function-calling) · [Gemini API: Function calling](https://ai.google.dev/gemini-api/docs/function-calling)
* [Microsoft Foundry: RAG evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators)
* [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) · [Hugging Face MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)
* [Microsoft: AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
* [Model Context Protocol](https://modelcontextprotocol.io/) · [LangChain documentation](https://docs.langchain.com/) · [LlamaIndex documentation](https://developers.llamaindex.ai/python/framework/)
* [Lilian Weng: LLM powered autonomous agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

# Link to the group project

Most projects will use RAG, an agent or both. Minimum expectations: a labelled evaluation set from your own documents (including unanswerable questions), retrieval metrics (recall@k, MRR) and answer metrics (correctness, faithfulness, citation accuracy, abstention), a justified choice between workflow and agent, least-privilege tools with human approval for any action, and a short threat model covering prompt injection and data access.
