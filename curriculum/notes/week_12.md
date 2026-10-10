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

Start with the vocabulary and overall flow. For each section below, read the simple explanation first, trace the worked example aloud, and ask students to name the input and output. Introduce the equation only after the operation makes sense. The detailed text retains the full syllabus, while the lab and checks show what each method can and cannot establish.

## 1. Why retrieval-augmented generation?

**Start here.** RAG means retrieval-augmented generation: search sources, then give useful passages to the model before it answers. Preparation builds the indexes; answering searches them for one question. The model weights remain fixed. A citation marker is a pointer to evidence, not proof that the evidence supports the claim.

**Small worked example.** The model does not know a private module handbook. Retrieve its late-penalty passage, label it [1], and ask for an answer supported by that text. If the passage is missing, the correct behaviour is to decline rather than invent a rule.

A language model may lack current facts or private documents, produce unsupported claims and omit usable evidence links. **Retrieval-augmented generation (RAG)** (Lewis et al., 2020) searches relevant sources when a question arrives and includes them with the prompt. An application can then attach citations. Retrieval can reduce unsupported answers, but the selected sources and each answer claim still need checking.

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

**Start here.** Split each document into searchable chunks and keep its source ID and permissions. Embeddings are vectors that enable meaning-based search. Keyword search matches terms instead. Both are prepared before a question arrives. Changing the embedding model requires rebuilding stored vectors.

**Small worked example.** A handbook section split into two chunks keeps the same source ID. A chunk with a deadline but no section heading may be unclear; overlap or a parent section can restore needed context. Test the size instead of guessing a universal best number.

**Embeddings.** A **bi-encoder** represents the question and each passage separately, allowing passage vectors to be stored in advance. Contrastive training encourages answering passages to receive high similarity to their questions (week 9). Examples include BGE, E5 and GTE or hosted embedding models. Compare relevant tasks, languages and permitted uses; MTEB is one benchmark collection. A changed embedding model needs newly encoded passage vectors.

**Chunking.** Split at headings and paragraphs where possible. A few hundred to about a thousand tokens with some overlap is a starting experiment, not a universal optimum. Large chunks may include distractions; very small chunks may lose meaning. Store source, section, date and permissions as **metadata** for filtering and citations. Test retrieval and answer quality at several sizes.

![Chunking with overlap](fig:chunking)

| Option | Type | Typical use |
|---|---|---|
| FAISS | Library | Fast in-process search (lab) |
| Chroma | Lightweight vector database | Local apps with metadata filters (lab) |
| pgvector | PostgreSQL extension | Vectors next to business data |
| Qdrant, Weaviate, Milvus, Pinecone | Dedicated vector databases | Scale, hybrid search, multi-tenant |
| Azure AI Search, Google Vector Search (formerly Vertex AI Vector Search) | Managed cloud search | Enterprise RAG with security integration |

**Approximate nearest neighbour (ANN)** search avoids comparing every vector exactly. HNSW uses linked neighbours, IVF uses clusters, and product quantisation compresses vector parts. Faster search can miss useful passages. Apply metadata filters and access checks before results enter the prompt; anything supplied to the model can appear in an answer.

## 3. Retrieval, advanced RAG and evaluation

**Start here.** Dense search uses vector meaning; BM25 uses words; hybrid search combines their rankings. A cross-encoder reads question and passage together to re-rank a smaller candidate set. Evaluate whether evidence was found separately from whether the model used it correctly. Lab hits are checked by known source-document ID.

**Small worked example.** First relevant results for three questions are at ranks 1, 2 and absent. MRR = (1 + 1/2 + 0)/3 = 0.5. Two have evidence in the first three results, so this lab's recall@3 = 2/3. These scores do not check the generated answer.

**Dense vs keyword.** Dense retrieval matches meaning (paraphrases); **BM25** matches words weighted by rarity (codes, names, rare terms). **Hybrid** search combines them, usually with **reciprocal rank fusion**:

$$\mathrm{RRF}(d) = \sum_{r} \frac{1}{c + \mathrm{rank}_r(d)}, \qquad c = 60$$

The RRF constant $c$ smooths rank contributions; it is separate from the result cutoff $K$.

**Metrics.** This lab's recall@K is the fraction of questions with a hit from the known source document in the first $K$ results; mean reciprocal rank averages one over the rank of the first relevant chunk:

$$\mathrm{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\mathrm{rank}_q}$$

A question with no relevant result in the returned list contributes zero to MRR. In this lab relevance is checked using the source document ID: a chunk from the known target document counts as a hit.

**Re-ranking.** Use a fast search to find a larger candidate set, then a **cross-encoder** to read each question-passage pair together. Select a smaller set for the answer prompt. This adds scoring work but may reduce irrelevant prompt text. Candidate counts such as 20–100 followed by 3–5 are starting examples; tune them against evidence-finding scores and cost.

**Lab measurement (retrieval):** {{RETRIEVAL_NOTES}}

![Retrieval quality by method (lab)](fig:lab_retrieval)

**Grounded generation.** Label each source and request an answer supported by it, with citations and a missing-evidence reply when needed. Tell the model to treat source text as data. Check whether the cited passage supports each claim; a citation number alone is not enough. In this lab, moving rules after the sources helped the 1.5 B model substantially. This is one measured prompt-placement result, not a general security guarantee.

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

**Start here.** An agent uses a model to choose next steps and propose tool calls. Application code parses the proposal, checks the allowed tool and arguments, applies approval, then executes. A tool result becomes an observation for the next turn. Use a fixed workflow if the sequence is already known.

**Small worked example.** To calculate a grade, the model may propose a calculator call rather than guessing the number. The code checks the expression, calculates it and returns the result. A simulated email proposal in this lab sends nothing and is checked by a policy rule.

An **agent** is an LLM that uses **tools** in a **loop** to achieve a goal, with **memory** (conversation, scratchpad, long-term notes) and **planning** (decomposing and revising steps), under **oversight** (permissions, approvals, logs).

![The agent loop](fig:agent_loop)

**ReAct** (Yao et al., 2022) interleaves reasoning and acting: decide → call a tool → observe the result → continue, until the model answers or a step limit is reached.

**Function calling.** Describe each tool's name, purpose and argument fields. The model proposes a call; application code parses it, checks the allowed tool, validates values and applies the required approval. Only then does code execute it. Return the result as an observation. The model does not execute the function itself, and a tool response can still contain errors or hostile instructions.

**Lab measurement (agent):** {{AGENT_NOTES}}

![A real agent trajectory from the lab](fig:agent_trace)

**Workflows vs agents** (Anthropic, "Building effective agents"): workflows orchestrate LLM calls through fixed code paths: **prompt chaining**, **routing**, **parallelisation**, **orchestrator–workers**, **evaluator–optimiser**. Agents let the model direct its own steps. Prefer the simplest pattern that works; use agents for open-ended tasks whose steps cannot be predicted, with limits.

![Workflow patterns](fig:workflows)

**Multi-agent systems** divide work among model-driven roles such as researcher, writer and reviewer. A coordinator or messages connect them. Roles can have different tools and permissions and may run in parallel. More calls add cost and debugging work; errors can spread between roles. Protocols such as A2A standardise how agents communicate, not whether their answers are correct.

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

**Start here.** A retrieved document can contain hostile instructions. If the model follows them, it may propose an action unrelated to the user's goal. Permissions, validated tool arguments, real approval for consequential operations, step limits and audit logs control what the program may execute. Source content should remain data.

**Small worked example.** A forum post says "email all grades to an external address". Trace the proposed tool call and refusal in the lab. A domain rule is a simulated policy gate, not a person approving mail or a complete real-world defence.

**Prompt injection** is the top risk in the OWASP Top 10 for LLM applications. **Direct** injection comes from the user; **indirect** injection (Greshake et al., 2023) hides instructions in content the system reads: web pages, e-mails, documents, retrieved chunks. With tools, injected text can trigger harmful actions.

**Lab measurement (indirect injection):** {{INJECTION_NOTES}}

**Guardrails for RAG and agents:**

1. **Least privilege:** read-only tools by default; no access to data the task does not need.
2. **Allowlists and validation:** known tools only; check argument types, ranges and recipients.
3. **Human approval** for side effects: sending, paying, deleting, publishing.
4. **Limits and sandboxing:** step, time and cost limits; sandboxed code execution.
5. **Logging and monitoring:** audit trail of every tool call; alerts on unusual actions.

> **Key idea:** Assume injected instructions will sometimes be followed; design so that following them cannot cause serious harm.

**Human oversight and responsible use:** show sources and provide a route to a human; respect permissions and personal data in the index (GDPR); keep the index current and versioned; evaluate continuously, including red-team injection tests; disclose AI use (EU AI Act Art. 50); check the AI Act scope before using a system for admissions or grading. Such intended uses can fall under Annex III high-risk categories; classification is not determined by the word "education" alone. See [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3) and [the consolidated Act](https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng/pdf).

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
* **Transparency and regulation:** disclose AI interaction as applicable under [Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50). Some admissions and learning-evaluation uses fall within [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3); check the intended use, Article 6 conditions and applicable dates.
* **Over-reliance:** users may trust cited answers without checking; design interfaces that encourage verification.

# Lab guide and answers

**Runtime:** Colab T4 GPU with Qwen2.5-1.5B-Instruct, BGE-small embeddings and an MS MARCO MiniLM cross-encoder; on CPU the notebook uses Qwen2.5-0.5B-Instruct. On the 4 GB GTX 1650 laptop GPU used to check this pack (9 October 2026), the full notebook with Qwen2.5-1.5B-Instruct ran in about 5 minutes with the models and data already downloaded. No API keys are needed. The handbook is fictional, so the model cannot have memorised it. **Hand-in:** retrieval and RAG tables, agent trajectories and audit log, three written answers.

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

1. RAG primarily addresses: **(a)** slow token-by-token decoding of answers; **(b)** limited GPU memory during inference; **(c)** tokenisation errors on rare words; **(d)** static knowledge and lack of private data. *Answer: (d).*
2. BM25 is typically better than dense retrieval for: **(a)** exact product codes; **(b)** paraphrased questions; **(c)** cross-lingual search; **(d)** images. *Answer: (a).*
3. A cross-encoder re-ranker: **(a)** embeds question and passage separately, then compares them; **(b)** generates the answer from the top passages; **(c)** reads question and passage together to score relevance; **(d)** splits documents into passages before indexing. *Answer: (c).*
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

# Detailed explanations behind the short slides

The slides show one idea at a time. Use the detail below for follow-up questions, technical terms and further study.

## Slide 1: Title

Welcome to week 12, the last technical week: retrieval-augmented generation and agentic AI. Everything so far comes together here. Embeddings from weeks 6 and 9 power retrieval; prompting and evaluation from week 7 shape grounded answers; deployment from week 11 turns the system into an application. RAG connects a model to knowledge it was never trained on; agents let a model act by calling tools. Both are among the most common designs for student projects and in industry, and both bring new risks, which we measure in the lab.

## Slide 2: Outcomes

Six outcomes matching the revised descriptor's week 12: motivation and LLM limitations; embeddings, vector databases and RAG architectures; advanced RAG and RAG evaluation; agentic AI concepts and components, including workflows versus agents, multi-agent systems and the Model Context Protocol; risks and guardrails; and choosing a design. This week practises MIMLO 5, implementing retrieval-augmented and agentic systems, together with MIMLO 3 and 4.

## Slide 3: Agenda

The plan follows the revised descriptor. Sections one to three build and evaluate a RAG system, with real retrieval and answer-quality results from the lab. Take the break after section three. Section four explains agents with a real agent trajectory from the lab, and section five covers prompt injection, including a real test with a malicious document, and guardrails.

## Slide 4: Warm-up

Two minutes. Students predict, correctly, that the model does not know this module's handbook, and usually that it will produce a plausible generic answer, such as five percent per working day, rather than saying it does not know. That is hallucination, and it can even be right by coincidence, which is worse because nobody can verify it. The fix they suggest is to give the model the handbook. That is retrieval-augmented generation, and the lab builds exactly this assistant for a fictional handbook.

- Ask a public chatbot: 'What is the late-submission penalty for this module?' What do you expect, and why?

- Does the model know your module's documents?

- Will it say 'I don't know', or invent a plausible rule?

- How could it answer correctly, and show where the answer came from?

## Slide 5: Why RAG?

Section one explains the limitations that motivate retrieval, compares RAG with the alternatives, fine-tuning and long context, and walks through the standard RAG pipeline. Refer back to the warm-up: the chatbot could not know the module's rules because they were never in its training data, and they may have changed since. About fifteen minutes.

- What can't an LLM do on its own?

## Slide 6: LLM limitations that retrieval addresses

Four limitations. Static knowledge: a model knows only what was in its training data up to its cut-off. Hallucination: when it does not know, it often produces a fluent guess. No private data: internal documents were never seen. No citations: answers from parametric memory cannot be traced to a source. Retrieval-augmented generation addresses all four by retrieving relevant passages at question time and asking the model to answer from them, with citations. It reduces hallucination but does not eliminate it, which is why evaluation matters.

- **Static knowledge:** Stored model knowledge may be out of date. Supply up-to-date sources.

- **Hallucination:** An answer can sound confident without supporting evidence. Retrieval helps but does not remove this risk.

- **No private data:** Private handbooks and records need an authorised connection; the model does not have access by default.

- **No citations:** A model answer alone may not identify a reliable source. A RAG app can link retrieved evidence.

## Slide 7: RAG, fine-tuning or long context?

Compare the three ways of giving a model knowledge. RAG adds knowledge that can be updated by re-indexing and supports citations and per-document access control, at low cost per query because only the top passages are sent. Fine-tuning, as week 8 showed, is good for behaviour and format but a poor way to add facts that change. Long-context models can read whole documents, which is excellent for a few long files, but sending hundreds of thousands of tokens with every question is slow and expensive, and quality can drop for information in the middle. Real systems combine them.

|  | RAG | Fine-tuning (week 8) | Long context |
| --- | --- | --- | --- |
| How information enters | Search for passages and add them | Learn patterns through weight updates | Put documents directly in the prompt |
| Evidence links | Attach source IDs and verify citations | Not guaranteed by weight training | Possible if sources are labelled and checked |
| Per-question work | Search plus selected passages | Model generation after training | Read every included token |
| Permissions | Filter documents before retrieval | Training data can be mixed into weights | Check all supplied documents |
| Useful when | Many documents or changing facts | Behaviour, style or skills need adaptation | A small document set fits the context |

## Slide 8: The RAG pipeline

There are two phases. Indexing, done offline: load documents, split them into chunks, embed each chunk, and store the vectors in a vector store together with a keyword index. Answering, done online: embed the question, retrieve candidate chunks with dense and keyword search, re-rank them, put the best few into a prompt as numbered sources, and ask the LLM to answer only from them with citations, or to say it does not know. Each box is a design decision with measurable effects, and the rest of the lecture goes through them.

## Slide 9: Quiz

**Question.** HR policies change monthly. Staff need exact source links. What should you try?

- **A.** Fine-tune the model every month
- **B.** Paste every policy into every prompt
- **C.** Rely on the model's general knowledge
- **D.** Re-index for RAG and check the citations

Answer: D. RAG keeps answers current through re-indexing, provides citations to the exact policy, and can respect access permissions. A is expensive, slow to update and gives no citations. C cannot know internal policies. B may work for a small set, but costs grow with the number of documents and every question pays for all of them; it also makes access control per document harder.

## Slide 10: Embeddings, chunking and vector databases

Section two covers the indexing side: text embeddings, how to chunk documents, and where to store and search the vectors. Indexing decisions are made once but affect every answer, and they are the most common cause of poor RAG quality in projects. About fifteen minutes.

- How can numbers represent a question and help find an answering passage?

## Slide 11: Text embeddings for retrieval

Embeddings connect back to week 9's contrastive learning. A retrieval embedding model is a bi-encoder: it encodes questions and passages separately into vectors, trained so that a question lands near passages that answer it. Retrieval is then a nearest-neighbour search by cosine similarity. Open models such as BGE, E5 and GTE, and API models from OpenAI, Google and Cohere, are common choices. The Massive Text Embedding Benchmark, MTEB, compares them across tasks and languages. Changing the embedding model means re-embedding every chunk.

- An **embedding** is a vector: a list of numbers representing text. Similar meanings should receive similar directions.

- A **bi-encoder** processes the question and passage separately, so passage vectors can be stored in advance.

- Compare the question vector with stored vectors using **cosine similarity**; return the closest matches.

- Choose a model using relevant tests, languages and licence. MTEB compares embedding tasks; changing models requires re-encoding.

## Slide 12: Chunking: splitting documents into retrievable pieces

Documents are split into chunks because embeddings of long texts become vague and prompts have limited space. Overlap ensures a sentence cut at a boundary appears in two chunks. There is no universal best size: typical values are a few hundred to a thousand tokens with ten to twenty percent overlap, and structure-aware splitting at headings and paragraphs usually beats fixed windows. Keep metadata with each chunk: source document, section, date and access level, which enable citations and filtering. In the lab, students implement a word-boundary chunker with overlap.

- A **chunk** is one searchable piece of a document. Large pieces may add irrelevant text; tiny pieces may lose meaning.

- Split at headings or paragraphs where possible. Test chunk size and overlap on your questions.

- **Metadata** is information kept beside the text: source, section, date and who may access it.

## Slide 13: Where to store vectors

FAISS, from Meta, is a library: very fast, but you manage persistence yourself. Chroma is a simple vector database with metadata, used in the lab. pgvector adds vector search to PostgreSQL, often the pragmatic choice when data already lives there. Dedicated vector databases such as Qdrant, Weaviate, Milvus and Pinecone add scale, hybrid search and multi-tenancy. Cloud search services from Microsoft and Google integrate with enterprise security and identity. For student projects, FAISS or Chroma are enough.

| Option | Type | Typical use |
| --- | --- | --- |
| FAISS | Library (in-process) | Fast search in notebooks and prototypes (the lab) |
| Chroma | Lightweight vector database | Local apps with metadata filters (the lab) |
| pgvector (PostgreSQL) | Extension to a relational database | Keep vectors next to existing business data |
| Qdrant, Weaviate, Milvus, Pinecone | Dedicated vector databases | Large scale, hybrid search, multi-tenant |
| Azure AI Search, Google Vector Search (formerly Vertex AI Vector Search) | Managed cloud search | Enterprise RAG with security integration |

## Slide 14: Searching millions of vectors quickly

Exact nearest-neighbour search compares the question with every stored vector, which is fine for the lab's handful of chunks and even for a few hundred thousand, but too slow at web scale. Approximate indexes such as HNSW, a navigable graph, IVF, which searches only the nearest clusters, and product quantisation, which compresses vectors, give large speed-ups with a small loss in recall. Metadata filters limit results to allowed documents; enforcing access control at retrieval time is essential, because anything retrieved can appear in the answer.

- Exact search compares the question with every stored vector; cost grows with the collection.

- **Approximate nearest neighbour (ANN)** search returns likely matches faster, but may miss some good ones.

- **HNSW** uses a neighbour graph; **IVF** searches clusters; **product quantisation** compresses vector parts.

- Apply permission and date filters before any passage reaches the prompt.

## Slide 15: Retrieval, advanced RAG and evaluation

Section three is the core of RAG quality: dense, keyword and hybrid retrieval; re-ranking; real results from the lab; grounded generation with citations; advanced techniques; and how to evaluate RAG systems. Stress the diagnostic mindset: retrieval and generation fail for different reasons and need different fixes. About thirty minutes; take the break afterwards.

- How do we retrieve well, answer faithfully and prove it?

## Slide 16: Dense vs. keyword retrieval

Dense and keyword retrieval fail in complementary ways. Dense retrieval matches meaning, so a question about handing work in after the deadline finds the late-submission policy even without shared words. BM25, the classic keyword ranking function, weights matching words by rarity and excels at exact identifiers such as module codes or product numbers, which embeddings often blur. Hybrid search runs both and fuses the results, which is why it is the default in production systems and in the lab.

| Dense (embeddings) | Keyword (BM25) |
|---|---|
| Matches **meaning**: 'hand in after the deadline' ↔ 'late submission' | Matches **words**, weighted by rarity (TF-IDF family) |
| Handles paraphrases and other wordings | Excellent for codes, IDs, names ('GAI-5011') |
| Weaker on exact codes, names, rare terms | Misses synonyms and paraphrases |

**Hybrid search** combines meaning-based and keyword search. Compare it with each alone on your questions.

## Slide 17: Fusing rankings and measuring retrieval

Reciprocal rank fusion combines ranked lists: each retriever contributes one over sixty plus the rank. A chunk ranked first by one retriever and fifth by the other scores well. RRF needs only ranks, which is convenient because BM25 scores and cosine similarities are on different scales. To measure retrieval, recall at k asks whether any relevant chunk made it into the top k, and mean reciprocal rank rewards putting it first. In the lab students implement RRF and compute both metrics.

Read reciprocal rank fusion (RRF) as: for each document, add one contribution from each ranking list. Each contribution is one divided by 60 plus that document's rank. The complete equation appears in Section 3 above.

Read the lab's gold-document hit rate as: questions whose identified source appears in the top K results, divided by all questions. Read mean reciprocal rank (MRR) as: average one divided by the first relevant result's rank. A missing relevant result contributes zero. These checks do not establish exhaustive passage recall or answer support.

- **RRF** adds 1/(c + rank) from each search list. A passage ranked highly in both lists gets a high combined score.

- Lab recall@K asks whether any chunk from the known source document appears in the first K results.

- **MRR** averages 1/rank for the first hit; no hit contributes 0. Keep RRF's constant c separate from cutoff K.

| Symbol | Meaning |
|---|---|
| rank_r(d) | position of passage d in a search list |
| c | RRF smoothing constant; 60 here |
| K | number of returned passages checked |
| rank_q | position of the first hit for question q |

## Slide 18: Re-ranking: a second, more careful look

Retrieval usually has two stages. The first is fast and aims for recall: embeddings or BM25 retrieve many candidates. The second, re-ranking, uses a cross-encoder that reads the question and each candidate passage together through a transformer, so it can judge relevance far more precisely, but it is too slow to run on the whole collection. Keeping only the top few re-ranked passages shortens prompts, reduces cost and gives the LLM fewer distracting passages. The lab uses a small open cross-encoder trained on the MS MARCO dataset.

- First retrieve a larger set quickly, for example 20–100 passages; this aims to avoid missing evidence.

- A **cross-encoder** then reads the question and each passage together to assign a more careful score.

- Keep a smaller set, often 3–5 passages. This is **re-ranking**.

- It adds computation, but may give the answer model fewer distractions. Measure whether it helps.

## Slide 19: Measured: retrieval quality on the handbook questions

These are the lab's measured retrieval results on twelve answerable questions about the handbook. The values are in the bullet. The collection is tiny, so all methods reach the right document within the top three; the differences appear at rank one, which matters because the first passage has the most influence. Hybrid fusion and re-ranking typically lift recall at one and MRR. On real collections with thousands of similar documents, the gaps between methods are much larger, which is why students must measure on their own data.

- {{RETRIEVAL_NOTES}}

- Small collection: real systems show larger gaps, so measure on your own data.

## Slide 20: Measured: RAG vs. no retrieval

The lab compares the same open model answering handbook questions without retrieval and with RAG. Without retrieval it can only guess, so accuracy is near zero, while RAG answers most questions correctly. The bullet reports the measured accuracy, whether numbers in answers are supported by the sources, whether the model cited the right document, and whether it abstained on the unanswerable question about the external examiner. One prompt-design lesson came from building this lab. With the answering rules only in the system message, the small model cited the right document for just one question in twelve and said I don't know more often. Moving the same rules to the end of the user message, after the sources and next to the question, fixed both. Small models attend most to nearby text, so put instructions where they will be read, and still check citations in code.

- {{RAG_NOTES}}

- Number the sources. Ask for an answer supported by them, a citation [n], or "I do not know" when evidence is missing.

- In this small-model run, placing rules after the sources helped; test placement rather than assuming it generalises.

## Slide 21: Advanced RAG techniques

Advanced techniques fix specific failures. Query rewriting turns follow-up questions such as "and for the exam?" into standalone queries. HyDE embeds a hypothetical answer, which often resembles relevant passages more than the question does. Contextual retrieval, published by Anthropic, adds a short explanation of each chunk's context before embedding; they reported roughly half as many failed retrievals, and more with re-ranking. GraphRAG, from Microsoft Research, answers global questions such as "what are the main themes?" using a knowledge graph. Parent-child retrieval balances precision and context. Agentic RAG lets an agent search iteratively. Use them when evaluation shows the specific failure.

- **Query rewriting:** Rewrite a vague question into a clearer search query; several variants can search different wordings.

- **HyDE:** generate a hypothetical answer and search with its vector. It is a search aid, not evidence.

- **Contextual retrieval:** Add a short explanation of where a chunk fits before indexing it. This can help isolated passages make sense.

- **GraphRAG:** Build a graph of entities and relationships, plus group summaries, to answer questions across documents.

- **Hierarchical / parent–child:** Search a small chunk, then supply its larger parent section so context is preserved.

- **Agentic RAG:** Let a controlled agent choose searches, inspect results and search again when needed.

## Slide 22: Evaluating RAG: retrieval and generation separately

Evaluate retrieval and generation separately, so that you know which one to fix. Retrieval quality asks whether the evidence reached the prompt. Faithfulness, also called groundedness, asks whether each claim in the answer is supported by the retrieved context; frameworks such as RAGAS and the RAG evaluators in Microsoft Foundry decompose answers into claims and check each one with a model. Answer relevance asks whether the answer addresses the question. Correctness needs reference answers. Include unanswerable questions to test abstention. The lab implements simple versions of these metrics.

| Aspect | Question it answers | How to measure |
| --- | --- | --- |
| Search results | Did useful evidence reach the prompt? | Recall@K, MRR; inspect returned passages |
| Faithfulness | Does each claim follow from the supplied sources? | Compare claims with passages; use calibrated human/model checks |
| Relevance | Does the answer address this question? | Human or model rubric |
| Correctness | Does it match the known answer? | Reference answers and field checks |
| Citations / abstention | Do sources support it? Does it decline missing evidence? | Check cited text; include unanswerable questions |

## Slide 23: Quiz

**Question.** Needed evidence is ranked 12th; only three passages reach the prompt. Improve first?

- **A.** Retrieval, chunking or re-ranking
- **B.** The size of the generator model
- **C.** The generation temperature
- **D.** A “be accurate” instruction

Answer: A. This is a retrieval failure: the evidence never reached the model, so no generator can answer faithfully. Improve retrieval with hybrid search, a re-ranker, different chunking or query rewriting, then re-measure recall at three. A bigger model may guess better but that is hallucination, not grounding. C and D do not change what the model sees. The general lesson: diagnose whether a failure is in retrieval or generation before fixing it.

## Slide 24: Agentic AI

Welcome back. Section four introduces agentic AI: the components of an agent, the loop that drives it, function calling, a real trajectory from the lab, the difference between workflows and agents, multi-agent systems, the Model Context Protocol and the frameworks. Agents build directly on RAG: search becomes one tool among several. About thirty minutes.

- What changes when the model can act?

## Slide 25: Components of an LLM agent

An agent is an LLM that uses tools in a loop to achieve a goal. The LLM reasons and chooses the next step. Tools are functions it can call, from search and calculators to code execution and external APIs. Memory holds the conversation and intermediate results, and sometimes long-term notes. Planning breaks goals into steps and revises them. The loop continues until the task is done or a limit is reached. Oversight is part of the design, not an add-on: permissions, approvals for risky actions, and logs.

- **LLM:** Proposes the next step or a final answer; the code enforces what may run.

- **Tools:** Named functions, such as a search or calculator, with described arguments.

- **Memory:** Earlier messages and saved notes; stale or hostile information must be handled carefully.

- **Planning:** Break the goal into steps and revise them from tool results.

- **Loop:** Choose a step, run an approved tool, read its result and repeat within limits.

- **Oversight:** Check permissions and approvals; record calls and outcomes.

## Slide 26: The agent loop (ReAct: reason, act, observe)

The ReAct pattern, from Yao and colleagues in 2022, interleaves reasoning and acting. The LLM reads the task and decides whether it needs a tool. If so, it emits a structured tool call with JSON arguments; the program, not the model, executes it and returns the result as an observation; the LLM reasons again with the new information. When no tool is needed, it gives the final answer. The step limit prevents endless loops. Stress that the model only proposes actions: your code decides whether to execute them, which is where guardrails live.

## Slide 27: Function calling: tools are described, the model proposes, your code executes

Function calling is supported by many endpoints and open-model chat templates; check the exact model and endpoint. You describe each tool with a name, a description and a JSON schema of its arguments; here the schema is generated from the Python type hints and docstring. The model replies either with text or with a structured call. Your code parses the call, checks it against an allowlist, requires argument validation and approval for consequential actions before executing; the lab illustrates an allowlist and simulated gate, then runs the permitted tool and appends the result as a tool message. This short loop illustrates tool selection and observations; production frameworks add their own state and execution controls.

```python
for step in range(MAX_STEPS):
    reply = llm(messages, tools=list(TOOLS.values()))
    calls = parse_tool_calls(reply)[:1]  # one call per turn
    if not calls:
        return reply
    call = calls[0]
    result = execute(call)              # allowlist and simulated gate
    messages.append({"role": "tool", "name": call["name"],
                     "content": str(result)})
```

- The model reads each tool's name, purpose and argument schema, then proposes a call.

- Your code checks the name, arguments and permission before executing it.

- Return the result as data. A tool response can contain wrong facts or hostile instructions.

## Slide 28: Measured: a real agent trajectory from the lab

This is the actual trajectory of the lab agent, an open 1.5-billion-parameter model, on the task "what is my final mark?". Walk through each step: which tool it called with which arguments, what came back, and the final answer. The correct answer is 68.8, from sixty percent of 72 plus forty percent of 64. Small models sometimes skip a tool or make arithmetic errors, which is why the lab checks the answer and why the calculator tool exists at all: arithmetic belongs in code.

- {{AGENT_NOTES}}

## Slide 29: Workflows vs. agents

Anthropic's guide "Building effective agents" distinguishes workflows, where LLM calls follow code paths you define, from agents, where the model decides its own steps. It names five workflow patterns: prompt chaining, routing, parallelisation, orchestrator-workers and evaluator-optimiser. Most successful applications are workflows or simple agents. Autonomous agents are worth their cost and unpredictability only for open-ended tasks where the steps cannot be fixed in advance, such as research or coding tasks. Recommend that students start with the simplest pattern that works.

- A **workflow** follows steps written in code: for example, retrieve → answer → check citations.

- An **agent** chooses its next step from the current information.

- Use a fixed workflow when the path is known; give agents step, time, cost and permission limits.

## Slide 30: Multi-agent systems

Multi-agent systems split a task among specialised agents, for example a research agent that searches, a writer and a reviewer, coordinated by an orchestrator or by exchanging messages. The benefits are specialisation, parallelism and different permissions per agent. The costs are real: multi-agent systems use many more tokens, are harder to debug, and errors can cascade. Protocols such as the Agent2Agent protocol, started by Google and now under the Linux Foundation, standardise communication between agents. For projects, one agent with good tools is usually enough.

- A **multi-agent system** divides work among model-driven roles, such as researcher, writer and reviewer.

- A coordinator or messages connect the roles; tools and permissions can differ.

- Parallel work may help, but adds calls, coordination and ways for one error to spread.

- Protocols such as **A2A** describe agent-to-agent communication. More agents do not guarantee better results.

## Slide 31: The Model Context Protocol (MCP)

The Model Context Protocol, introduced by Anthropic in November 2024 and donated in December 2025 to the Agentic AI Foundation under the Linux Foundation, standardises how AI applications connect to tools and data. Servers expose tools, resources and prompt templates; clients in assistants, IDEs and agent frameworks discover and call them. This solves the problem of writing a separate integration for every pair of application and tool. Security still matters: an MCP server is code with access to your data, so install only trusted servers and apply least privilege. The lab writes an incomplete MCP tool scaffold: retrieve_passages is a placeholder that needs a working retrieval implementation and loaded index.

- **MCP (Model Context Protocol)** describes how AI applications discover and use connected tools and data.

- An MCP **server** exposes capabilities; an MCP **client** in an app requests them.

- It was introduced by Anthropic in 2024 and donated to the Linux Foundation's Agentic AI Foundation in December 2025.

- A standard connection does not establish trust: choose permissions and trusted servers.

## Slide 32: Frameworks and tools for RAG and agents

A snapshot of frameworks. LangGraph models agents as graphs with state, checkpoints and human-in-the-loop steps. LlamaIndex specialises in data connectors and retrieval. The major providers have their own agent kits: OpenAI's Agents SDK, Google's Agent Development Kit and Microsoft's Agent Framework, which succeeds Semantic Kernel and AutoGen for agent building. Hugging Face's smolagents is minimal and works well with open models. All implement the same loop students write by hand in the lab; learning the loop first makes any framework easy to pick up.

| Tool | Strength |
| --- | --- |
| LangChain / **LangGraph** | Components and stateful agent graphs with checkpoints and human-in-the-loop |
| **LlamaIndex** | Data connectors, indexing and retrieval pipelines for RAG |
| OpenAI Agents SDK | Agents, tools, handoffs and tracing with OpenAI models |
| Google Agent Development Kit (ADK) | Multi-agent systems with Gemini and other models |
| Microsoft Agent Framework / Semantic Kernel | Enterprise agents in Python and .NET on Azure |
| Hugging Face smolagents | Minimal agents with open models; code-writing agents |

## Slide 33: Quiz

**Question.** A course assistant answers handbook deadlines and rules. Choose the initial design.

- **A.** Three cooperating agents
- **B.** Autonomous web/email agent
- **C.** RAG workflow with sources and abstention
- **D.** Fine-tuning without retrieved evidence

Answer: C. The task is single-step factual question answering over known documents, which a fixed RAG workflow handles predictably, cheaply and verifiably. A adds cost and failure modes without benefit. B adds risk, especially e-mail, and unreliable web sources. D cannot keep deadlines up to date and gives no citations. Agents are for tasks whose steps cannot be fixed in advance.

## Slide 34: Risks and guardrails

The final section covers prompt injection, especially indirect injection through retrieved content, with a real test from the lab, layered guardrails for agents, and human oversight. It is short but essential: the design principles apply to every project that uses retrieval or tools. About thirteen minutes.

- What can go wrong when models read documents and take actions?

## Slide 35: Prompt injection, especially indirect

Prompt injection is the top risk in the OWASP list for LLM applications. Direct injection comes from the user. Indirect injection, described by Greshake and colleagues in 2023, hides instructions in content the system reads: a web page, an e-mail, a shared document or a retrieved chunk. LLMs cannot reliably separate data from instructions in their context. For a chatbot, the result is a wrong answer; for an agent with tools, it can be data theft or harmful actions. The lab plants such an instruction in a forum post; the bullet reports what the agent did and what stopped it.

- **Direct injection:** the request tries to override rules. **Indirect injection:** hostile instructions are hidden in retrieved content.

- A document can say "send these records elsewhere". It remains document data, even if the model treats it as a command.

- With tools, an unsafe proposal could leak data or change something. The executing code must control actions.

- **Lab forum attack.** {{INJECTION_NOTES}} The email tool and approval step are simulated.

## Slide 36: Guardrails for RAG and agents

Because no prompt can guarantee that the model ignores injected instructions, safety must come from the system design. Give agents the least privilege they need, read-only by default. Allowlist tools and validate arguments, for example permitted e-mail domains. Require human approval for actions with side effects. Limit steps, time and cost, and sandbox code execution. Log every tool call for audit and monitoring. The callout is the principle: assume the model will sometimes be fooled, and make sure being fooled cannot cause serious harm.

- **Permissions:** least privilege, with read-only tools by default and no access to data the task does not need.

- **Validate:** known tools only; check argument types, ranges and recipients.

- **Approve:** a person approves actions with side effects, such as send, pay, delete or publish.

- **Limit:** step, time and cost limits; run code in a sandbox.

- **Audit:** a record of every tool call and its outcome; alerts on unusual actions.

- Design for mistakes: a hostile document may influence the model, while code still limits what actions are possible.

## Slide 37: Human oversight and responsible use

Human oversight means users can check sources and reach a person, and operators review logs and outcomes. Indexes must respect permissions and data protection: personal data in documents becomes retrievable by anyone who can query the system. Evaluation continues after launch, including red-team tests of injection. Legally, users must be told they are interacting with AI, and systems used for high-risk purposes in education or employment, such as admissions or grading decisions, face strict requirements under the EU AI Act. AI Act classification depends on intended use and the relevant rules, exceptions and dates. Educational admissions and evaluation of learning outcomes can fall within Annex III; a generic course assistant label alone does not decide classification.

- Show sources that support the answer and give users a route to a person.

- Apply document permissions, minimise personal data and update the index when sources change.

- Test search, supported claims, missing-evidence replies and hostile instructions.

- Disclose AI interaction. Admissions or grading systems may fall under the AI Act's high-risk rules; scope and dates need checking.

## Slide 38: Lab 12: a RAG assistant and a tool-using agent, measured

The lab covers both descriptor tutorials: building and evaluating a RAG pipeline with and without retrieval, and implementing an agent that calls tools, including a calculator and search over the RAG index. The handbook is fictional, so the model cannot have memorised it. The injection test shows why guardrails belong in code. The final question asks students to design a production version, which prepares them for their projects.

- Measure the model's answers to private handbook questions without documents.

- Split the handbook, create vectors and index it with FAISS and Chroma.

- Compare BM25, dense search, RRF hybrid search and re-ranking; calculate recall@K and MRR.

- Answer from numbered sources; check accuracy, supported claims, citations and missing-evidence replies.

- Trace a tool-using agent with search, calculator and simulated email; inspect approved-tool and simulated-approval checks, then an MCP scaffold.

- **Runtime:** Colab T4 GPU; CPU uses a smaller model. About 5 min on a 4 GB laptop GPU after downloads.

- **Hand in:** notebook with retrieval and RAG tables, agent trajectories and audit log, and three answers.

## Slide 39: Summary

The six cards on the slide condense these eight summary points. Students should be able to draw the RAG pipeline, justify each design choice with a metric, explain the agent loop and function calling, distinguish workflows from agents, and design guardrails that do not rely on the model obeying its instructions. This completes the technical content of the module; the remaining sessions focus on the projects, which will combine RAG or agents with the deployment practices from week 11.

- A model may lack your facts; RAG supplies searched sources at question time.

- Prepare documents as chunks with vectors, source IDs and permissions.

- Compare keyword and meaning-based search; combine and re-rank when tests justify it.

- Answer from evidence, cite it, and decline when the answer is absent.

- Test retrieval separately from answer relevance, correctness and supported claims.

- An agent proposes steps and tool calls; your code controls execution.

- MCP standardises tool connections; fixed workflows often suit predictable tasks.

- Limit access, validate actions, require appropriate approval and keep an audit trail.

## Slide 40: Resources

The RAG and ReAct papers are the foundations. Anthropic's two engineering posts are practical guides to agent design and retrieval quality. OpenAI's function-calling guide explains tool schemas; Google's Gemini function-calling guide is equivalent. The Hugging Face Agents course and Microsoft's AI Agents for Beginners are free, hands-on courses. The Model Context Protocol site has the specification and SDKs. Ozdemir's Quick Start Guide to Large Language Models, chapters on semantic search and retrieval, is the textbook reading. See the source-verification report for the latest checks and access limitations.

- **Paper:** [Lewis et al. (2020) Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401). The original RAG paper.
- **Paper:** [Yao et al. (2022) ReAct](https://arxiv.org/abs/2210.03629). Reasoning and acting with tools.
- **Reference:** [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents). Workflows vs agents.
- **Reference:** [Anthropic: Contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval). Improving RAG retrieval.
- **Reference:** [OpenAI: Function calling](https://developers.openai.com/api/docs/guides/function-calling). Tool use with the API.
- **Course:** [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction). Free course on agents.
- **Course:** [Microsoft: AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners). Lessons with code.
- **Reference:** [Model Context Protocol](https://modelcontextprotocol.io/). Specification and SDKs.


# Complete code illustrations

These longer illustrations require the definitions and imports supplied by the lab. They explain the pattern; use the notebook for the executable version.

This generic illustration can process several proposed calls in a turn. The classroom notebook deliberately handles one call per turn, then lets the model read that observation before proposing another.

## Function calling: tools are described, the model proposes, your code executes

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(15 + 3) * 2'."""
    ...

TOOLS = {"search_handbook": search_handbook, "calculator": calculator}

for step in range(MAX_STEPS):
    reply = llm(messages, tools=list(TOOLS.values()))  # schemas from docstrings
    calls = parse_tool_calls(reply)       # e.g. {"name": ..., "arguments": {...}}
    if not calls:
        return reply                      # final answer
    for call in calls:
        result = execute(call)            # allowlist, validation, approval
        messages.append({"role": "tool", "name": call["name"], "content": result})
```
