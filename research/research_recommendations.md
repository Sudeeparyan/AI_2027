# Research recommendations for the revised Generative AI module

Checked on 28 September 2026. This is a curriculum recommendation, not a change to the approved learning outcomes, assessment, hours or other descriptor sections. The machine-readable evidence ledger is [verified_sources_2026.json](verified_sources_2026.json). OpenAI sources are researched separately.

## Overall decision

Teach a broad, practical overview with a clear explanation of how the models work. Introduce responsible use immediately, then return to it in every lab. Teach learners to compare results and explain errors, not simply follow vendor interfaces. This recommendation follows the accessible progression in [Microsoft's introduction](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/) and [Google's beginner path](https://www.skills.google/paths/118), while adding the mathematical and implementation depth required by the unchanged MSc outcomes.

The fixed descriptor outcomes determine which practical tasks are compulsory. In particular, an outcome requiring design, implementation and fine-tuning cannot be satisfied by a slide saying what fine-tuning is, or by an optional demonstration. Use a genuinely trained, small model with a held-out evaluation; explain how that small experiment differs from tuning a production LLM.

## Keep, strengthen and limit

| Decision | Content | Reason and teaching treatment |
|---|---|---|
| Keep | Probability, latent spaces, VAEs, GANs, diffusion and transformers | These give learners a defensible understanding of model families. Use visual comparisons, simple equations and small worked examples before optional derivations. |
| Strengthen | Tokens, embeddings, context limits, model versus application | These concepts explain everyday tool behaviour and prepare learners for prompting and retrieval. |
| Strengthen | Prompting, useful context and structured responses | Use a fixed task, several prompts and a rubric. A valid output format does not establish factual correctness. |
| Strengthen | Text, image, audio and video | Compare both generation and interpretation. Test alignment between modalities and preserve evidence, consent and attribution. |
| Strengthen | Retrieval and grounded answers | Explain retrieval, generation and citation checking separately. RAG can still fail. |
| Strengthen | Baselines, held-out cases and human evaluation | Students should show whether a change improves results and where it fails. |
| Include briefly | Function calling, agents, MCP and multiple agents | Trace a bounded tool call. Explain MCP as a connection standard; it is not a model, vector database or guarantee of trustworthy tools. |
| Limit | Production serving stacks, framework surveys, complex orchestration | These consume overview time without directly improving basic conceptual understanding. Provide extension links instead. |
| Limit | Enterprise strategy, SHAP/LIME and detailed compliance implementation | Adjacent modules already address business adoption and trustworthy AI assurance. Retain GenAI-specific practical responsibility here. |

The local neighbouring descriptors reviewed were AI Technologies, Explainable and Emerging AI Technologies, and Generative AI for Business. Their extracted 7.3 tables are incomplete in places, so the overlap judgement uses their learning outcomes and assessment descriptions as well as the available topic rows. It is a scope judgement, not evidence that every neighbouring module teaches every item listed.

## Concrete corrections to teach

- Compare encoder-only, decoder-only and encoder-decoder architectures accurately. The [BERT paper](https://arxiv.org/abs/1810.04805) describes bidirectional encoder representations; BERT should not be presented as a like-for-like text chat generator.
- Explain prompt changes, retrieval and weight updates separately. These can be combined, but they change different parts of a system. [Microsoft's comparison](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/concepts/customizing-llms) supports this decision exercise; its classic-portal UI should not become current setup instructions.
- Ask for concise reasons, evidence and checks in assessed answers. Do not assess a model's displayed reasoning as a faithful record of its internal computation.
- Teach structured output as a format contract. The [Google guide](https://ai.google.dev/gemini-api/docs/structured-output) explicitly requires value validation.
- Teach that the application executes a tool, after checking its name, arguments and permissions. The [function-calling workflow](https://ai.google.dev/gemini-api/docs/function-calling) separates model selection from execution.
- Keep retrieval quality distinct from answer quality. [Microsoft's RAG evaluation guidance](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators) separates retrieval, groundedness, relevance and completeness.
- Teach that context selection still matters with large windows. [Google's long-context guidance](https://ai.google.dev/gemini-api/docs/long-context) records retrieval limitations and cost/latency trade-offs.

## Suggested classroom evidence

Every practical exercise should produce a small evidence table: task and success criterion; input and configuration; observed output; check against a reference or rubric; one failure; one justified improvement. Model names and access dates belong in the experiment record. Keep API keys outside notebooks. Supply an offline alternative that is clearly identified as a simulation or a review of recorded evidence.

For real fine-tuning, split examples before training, record baseline scores, train a small model or adapter, and compare on the same unseen cases. Explain overfitting and data leakage. [Google's adaptation lesson](https://developers.google.com/machine-learning/crash-course/llm/tuning) and the [LoRA abstract](https://arxiv.org/abs/2106.09685) support the concepts; neither proves a classroom implementation works. That must be established by running it.

## Video references and verification limits

| Reference | Verified evidence | Suggested use |
|---|---|---|
| [Microsoft Developer full beginner series](https://www.youtube.com/watch?v=k7HaeJs-N-o) | Search metadata identifies the verified publisher and chapter list; direct fetch failed. | Choose one short chapter after instructor preview. |
| [Google Cloud introduction to LLMs](https://www.youtube.com/watch?v=RBzXsQHjptQ) | Search metadata identifies the verified publisher and description; direct fetch failed. | Opening or revision viewing after preview. |
| [Basics of prompt engineering](https://www.youtube.com/watch?v=e7w6QV1NX1c) | Linked by title from the opened Microsoft Learn customization page. | Optional supplementary explanation. |
| [To fine-tune or not to fine-tune?](https://www.youtube.com/watch?v=0Jo-z-MFxJs) | Linked by title from the opened Microsoft Learn customization page. | Optional decision discussion. |

No video was watched and no transcript was reviewed in this research pass. Written sources support the instructional claims. Do not claim that every video statement or demonstration has been checked. Provide written notes as the primary accessible resource.

## Critical reading and refresh

Even official introductory pages simplify. For example, an original encoder-decoder transformer picture does not describe every modern LLM; broad claims that every practical task requires fine-tuning or that larger models always perform better should not be repeated without qualification. Vendor capability claims and example benchmark results are not universal guarantees. Foundations remain stable, while model names, APIs, free access, limits and interfaces change. Recheck the live demonstration instructions shortly before teaching in 2027; the present evidence date is September 2026.

