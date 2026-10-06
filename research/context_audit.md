# Context and scope audit

Reviewed 28 September 2026. This audit reads the actual Word XML, including nested tables, rather than assuming the earlier plain-text extraction is complete. It does not approve programme changes.

## Authoritative boundary

The user's controlling template is `context/Descriptor - Generative AI-old.docx`. Only the weekly content table within Section 7.3 may change. Section 7.3's heading and table format remain. Preserve Sections 7.1, 7.2 and 7.4–7.9, the assessment percentages, learning outcomes, overview, reading list, formatting, tracked changes, comments and package resources. Do not use `context/01_Generative_AI_2027.docx` as the template.

At the top level of `word/document.xml`'s body, zero-based child 11 is the Section 7.3 heading table; child 12 is the 13-row weekly table. Its columns are **Lecture Topic | Week | Detail | Tutorials (Examples)**. It is the only permitted replacement. Compare the other body children, and every ZIP part other than `word/document.xml`, against the source after editing.

The original contains tracked changes (11 insertion elements and 2 deletion elements) and reviewer comments. The text extract `research/original_descriptor_extracted.txt` includes revision text as present in the XML; it is an audit aid, not an instruction to accept or reject revisions. The comments about independent hours, ratios, assessment and newer books are outside the authorised edit. In particular, the original narrative says project 60% and terminal examination 40%, while a comment questions the Programme Schedule. Preserve the descriptor's wording and flag the existing discrepancy separately; do not invent an assessment decision.

## Immutable learning outcomes

1. Critically explain core concepts and mathematical principles, and show in-depth knowledge of VAEs, GANs, diffusion models and transformer-based LLMs (MIPLO 1, 4).
2. Analyse and apply multimodal generative AI, including alignment and cross-modal reasoning across text, image, audio and video (MIPLO 1, 2).
3. Design, implement and fine-tune generative AI models using contemporary frameworks and parameter-efficient methods (MIPLO 2).
4. Critically evaluate outputs using quantitative and human-centred assessment, including fairness, bias and hallucinations (MIPLO 3, 5).
5. Deploy generative AI models in real-world scenarios using contemporary frameworks and implement RAG or agentic systems (MIPLO 1, 2).

The teaching can be clear and supported without removing the performance verbs. A replay comparison alone is not evidence that a learner implements or fine-tunes a generative model. A manually assigned vector is not a learned embedding experiment. An extractive text lookup is a useful retrieval baseline but is not a complete retrieval-augmented generation implementation. A private/local demonstration is an appropriate deployment exercise when its limitations are explicit; commercial production rollout is unnecessary.

## Why the previous revision did not fit

`context/01_Generative_AI_2027.docx` is a substantially rewritten descriptor. It changes outcomes, prerequisites, staffing, delivery and assessment elaboration, rather than changing only 7.3. Its LO3 is an interpretation task and its live tuning is optional. This conflicts with the original requirement to design, implement and fine-tune.

The existing weekly sources and `docs/SYLLABUS_ALIGNMENT.md` explicitly disclose toy/replay limits: Week 2 has hand-built VAE/GAN proxies, Week 3 uses the clean target in a denoising illustration, Week 6 has keyword/synonym search and supplied vectors, Week 9 has supplied correctness labels, Week 10 has manual source-linked transcription, and Weeks 7/11/12 use a rule-based extractive policy assistant. The honest labels should be retained where such activities remain, but they cannot be the entire practical pathway. Two weeks of retrieval and several policy-assistant control weeks crowd out the broad generative overview, multimodal coverage and real supported adaptation.

The prior project's easy vocabulary, explicit examples, guided notebooks, repeated checks and local fallbacks are useful teaching practices to retain. The corrected pack should add genuine small models and clear tool exercises, rather than abandoning accessible explanations.

## Neighbouring modules and duplication

`sources/originals/08-04-AI-Technologies.docx` covers AI/data foundations, no-code ML, deep learning, vision, NLP, conceptual transformers in Week 8, generation in Week 9, prompting in Week 10, decision support and deployment. Its role is general technological/business literacy. This module should revisit those ideas briefly, then add generative mechanisms, evidence and hands-on modification.

`sources/originals/09-09-Explainable-and-Emerging-AI-Technologies.docx` already covers regulation, compliance, lineage, advanced explanations, drift, red-teaming, simulation, sustainability, human controls, edge/federated/neuromorphic AI, foundation-model governance and system assurance. This module still teaches responsible use throughout, but does not need to duplicate a full compliance or assurance course.

`sources/originals/10-06-Generative-AI-For-Business-new-.docx` covers business use cases, strategy, operations, analytics, enterprise integration and agents. This MSc Generative AI module should use relatable business/student examples while focusing on model understanding and practical generative tools, rather than adopting that business module's whole syllabus.

**Extraction error:** the existing extracted files for 08 and 09 omit their nested Section 7.3 weekly tables. The originals contain full 12-week schedules. Empty extracted sections must not be interpreted as absent teaching. Re-ingestion should recursively preserve tables or extract all paragraph descendants in reading order.

`sources/extracted/11-Generative-AI-for-Developers-and-Business_-A-Research-Based-Curriculum-Blueprint-for-2027.txt` describes 30-week, 150-contact-hour pathways, developer infrastructure and extensive operations. It is inspiration, not the 12-week module specification. Its future-facing model, pricing and regulatory claims must be independently checked before reuse. Do not transfer the 30-week workload or assessment scheme into this descriptor.

## Recommended 12-week sequence

All weeks retain the original 2-hour lecture and 2-hour tutorial/lab pattern. A three-part lab should provide a worked start, a learner modification and an evidence-based explanation. The ordinary route uses small CPU-compatible tasks where feasible; live hosted tools are clearly marked and have a documented demonstration fallback. A fallback does not silently replace required implementation evidence.

| Week | Topic | Core evidence | Outcomes |
|---|---|---|---|
| 1 | Generative AI foundations and responsible tool use | Generate and evaluate a small output; distinguish probability, model and application | LO1, LO4 |
| 2 | VAEs and GANs | Run and modify small implementations; explain latent sampling, losses and failure modes | LO1, LO3, LO4 |
| 3 | Diffusion and generated media | Guided denoising experiment; compare output controls and media limitations | LO1, LO2, LO4 |
| 4 | Transformers and LLMs | Tokenisation, scaled attention and causal-mask calculations; small language generation | LO1, LO3 |
| 5 | Prompting, context and structured outputs | Compare prompts on fixed tasks and validate a structured response | LO4, LO5 |
| 6 | Multimodal AI | Apply text/image/audio/video examples; distinguish alignment, perception and reasoning | LO2, LO4 |
| 7 | Embeddings and RAG | Build a small retrieved-context generator; evaluate retrieval and answer support separately | LO4, LO5 |
| 8 | Model adaptation and parameter-efficient fine-tuning | Perform a small real fine-tuning update; compare base and adapted outputs on unseen cases | LO3, LO4 |
| 9 | Tools, workflows and agents | Build a limited tool-using workflow; compare deterministic steps and agent choice | LO4, LO5 |
| 10 | Evaluation and responsible generative AI | Quantitative, human and failure analysis across examples; document limitations | LO4 |
| 11 | Local/cloud deployment and frameworks | Run a small application with documented setup, privacy/cost limits and failure handling | LO5, LO4 |
| 12 | Integration and critical demonstration | Demonstrate a coherent small application, compare alternatives and explain individual work | LO1–5 |

MCP, reasoning-model behaviour, context management and multimodal changes are appropriate concise current topics. Multi-agent orchestration, large distributed training and specialist GPU operations are extensions. Keep substantive mathematics through small worked examples (probability, reconstruction/regularisation, adversarial objective, diffusion noise, attention), with lengthy proofs optional. Preserve actual LO3 practice through short supported implementations and fine-tuning. Explain BERT as an encoder/masked-language model, not a conversational generative peer to a decoder LLM. Do not describe chain-of-thought text as a trustworthy view of a model's internal reasoning.

## Review gates

1. Confirm only Section 7.3 changes in the descriptor, with its four-column format retained.
2. Check every original outcome has taught and practised evidence; no outcome is silently weakened.
3. Check every week has specific PPT explanations, teacher reference material, notebook instructions and practice answers.
4. Execute notebook core paths from clean kernels. Label mocks, synthetic data and optional online routes.
5. Check citations against original publications and official learning pages; do not use the blueprint as authority for current product claims.
6. Inspect rendered documents/slides, including dense tables, mathematical notation and speaker notes.
7. Report what automated checks establish and which human teaching/academic decisions remain; do not claim perfection or programme approval.
