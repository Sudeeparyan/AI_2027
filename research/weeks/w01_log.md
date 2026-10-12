# Week 01 lecture upgrade log

## Iteration 3 · W01 L · 2026-10-11

- Read/design: all 40 core slides, five beginner diagrams, recap/vocabulary, Word notes and figures reviewed; past-defect table read.
- Authored: 40 core and seven support notes use In short/Explain/Example/Ask/Answer/Watch out; topic-specific worked numbers or shapes and analogy limits retained.
- Visuals: informative examples/tables replace decorative bullet icons on core slides 6, 25 and 34. Crowded discussion/cards/compare/table/quiz wording shortened; 40 core plus six inserted beginner slides retained.
- Visible learning checks: core quizzes 12/29, predict-before-reveal temperature question 24, and reference-verification discussion 37; existing order/count retained. Compact outcome items budget the unchanged MIMLO sidebar; full official objectives remain unchanged.
- Course map: agenda and map notes define VAE (variational autoencoder), GAN (generative adversarial network), LLM (large language model) and RAG (retrieval-augmented generation) through their operations.
- Figures: compacted timeline/token/temperature/classifier/course-map labels; family rows show actual generation operations, symbolic flow shapes, role legend and dashed repetition. Symbolic diagrams are not measured image-quality results.
- Diagrams: code labels show only an owning local operation; repeated sample_name/chat/N/P labels hidden where they imply the whole function belongs to multiple substeps. Row-sum check and NLL recap role use metric rather than loss.
- Word: added stepwise count normalisation and Bayes examples, five answered self-checks and beginner FAQ. CPU/default route and USE_GEMINI/local fallback wording aligned; unsupported download-size and old runtime prose removed. Existing measured-result placeholders preserved.
- Sources proposed for shared sources.yaml: Karpathy makemore https://www.youtube.com/watch?v=PaCmpygFfXo ; Google MLCC https://developers.google.com/machine-learning/crash-course/llm ; Transformers generation parameters https://huggingface.co/docs/transformers/main/en/main_classes/text_generation . Historical scale sources checked 2026-10-10: GPT-3 paper https://arxiv.org/abs/2005.14165 ; Meta Llama 3 https://ai.meta.com/blog/meta-llama-3/ ; Google Gemini 1.5 Pro 2M preview https://developers.googleblog.com/en/google-io-2024-recap-making-ai-accessible-and-helpful-for-every-developer/ ; OpenAI weekly user report https://openai.com/index/openai-and-broadcom-announce-strategic-collaboration/ . Article 50 official text https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50 ; guidelines https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems (publication 20 July, page update 31 July 2026, corrected in notes) ; transition FAQ https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act . No current-record claim made.
- Glossary proposals: confirm canonical Bigram, Conditional probability, Logit, Softmax, Normalisation, Nat, Perplexity, Prior, Likelihood, Density, Grounding and Prompt injection entries; this task did not edit glossary.json.
- Validation pending root coordination: build, five text audits, coverage, visual render and Word/PPT/notebook comparison. No live Gemini, Colab timing or widget rendering claimed.
- Handoff: root rebuilt final source on 11 Oct; all46actualslides have ≤40nativebodywords. PASS: prose, inline maths, slide text, flat bullets, coverage100%, beginner verification12/12, figure layout and projected figure text. Guarded render/visual review follows; authored note assertions141–177words.

| Week | File | Slide/page | Problem | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 01 | PPT/JSON | 8,21,24 | Diagram legends/questions took actual body above40words | Medium | Shortened visible questions; full explanations stay in notes | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | beginner JSON | training/inference | Empty block details failed schema | Medium | Added Names/Pairs/Current details | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | YAML/lab | callout22/widget prose | Approximation spacing and repeated-word audit | Low | Spaces around ≈; explicit temperature instruction | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | figure source | timeline/families | Text reached block edges or feedback route | Medium | Wrapped year/family labels; moved data labels; shortened feedback label | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | beginner lab diagram | 43 | Solid arrows falsely linked independent classifier/text exercises | High | Replaced with name-scoring flow and separate supplied-name branch | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | Word engine | 1 | File table promised Python lab exports | Medium | notes.js describes notebooks and diagrams | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | Word source | 12 | Training cost generalised frontier-scale compute to all models | Low | Defined count/neural training separately; qualified frontier scale | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | Word source | 32 | T4/GPU-first lab instruction conflicted with CPU default | Medium | CPU-first Colab/VSCode; GPU availability optional | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | Word source | 33 | Final summary bullet orphaned on new page | Low | Keep whole Summary section together with source pagebreak | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | native flow slides | 18,41 | Arrows lacked flow labels and role legend/export parity | Medium | Authored truthful data arrows/roles; E3f sharedscene adapter | Closed: final rebuilt output and checks/review confirmed (12 Oct) |
| 01 | card layout | 10,13,40 | Default lightbulb badges added no teaching information | Low | Remove default badge; preserve informative comparison cards | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

- Final pilot handoff (11 Oct): root viewed all46slides,9delivered diagrams,11notebookimages; independent review covered all33Wordpages in finalPDF F6C78AB376303F5691A3F33B9FBBF8EDC2D3675D9B0E22C5F65EC9B4CFE2CEEB. Layout report is []; zero open findings. Eight source/build audits and actualbody inventory passed, coverage100%. API choice aligned across slide44,Wordp22 andnotebook; no live provider call. Before/after review: build/review_week01/index.md.

| 01 | Word/slide/notebook | p22/slide44/API comparison | New provider option was missing from references and hosted checkpoint label always said Gemini | Medium | Aligned both opt-ins/key guidance/charges/localfallback; selected checkpoint by provider | Closed: final rebuilt output and checks/review confirmed (12 Oct) |

## Final coordinated closure — 12 October 2026
- Final current coverage: 46 slides, 33 Word pages, 9 delivered diagram PNGs and 11 notebook images; actual views plus byte-identical previously viewed images are documented in the final review manifests. Layout report [], zero open findings.
- Latest successful cached local CPU full core: 44.6 s (0.7 min), recorded in `research/lab_runs_2026-10.json`; final smoke PASS (65.7 s). Current executed core code matches the successful full-run evidence.
- All 26 current release gates passed, including coverage 100%, strict notes, body ≤ 40, fonts ≥ 14, semantic helpers and shared diagram/attachment parity. See `research/ITERATION3_2026-10-12.json`.
- No timed Colab, clean-install/download, live-provider, browser-widget or optional GPU/stretch claim. Earlier pending handoffs above describe intermediate states.
