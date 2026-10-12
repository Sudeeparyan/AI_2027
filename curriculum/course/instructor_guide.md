# How to use this teaching pack

This pack delivers the 12-week **Generative AI** module of the MSc in Artificial Intelligence as described in the module descriptor, with the proposed Section 7.3 revision (see the separate rationale document). Every week follows the same pattern: a 2-hour lecture and a 2-hour lab.

| Each week's folder contains | Use it for |
|---|---|
| `Week_XX_Lecture_Slides.pptx` | 46 slides with full speaker notes (View ▸ Notes), including the course map, starting vocabulary, five technical walkthroughs and a final recap infographic. |
| `Week_XX_Teaching_Notes.docx` | Your reference: lecture plan with timings, explanations, maths and worked examples, misconceptions, responsible-AI lens, lab answers, practice and exam-style questions with model answers, glossary, readings. |
| `Week_XX_Lab.ipynb` | Student notebook for the 2-hour lab (Google Colab or local Jupyter). Complete each TODO before running cells that depend on it. |
| `Week_XX_Lab_Solutions.ipynb` | Instructor version with complete code and model answers. Do not distribute before the lab. |
| Developer Python exports in `build/week_XX/` | Source-comparison files for maintainers. Students use the `.ipynb` notebooks in Colab or VS Code. |
| `Diagrams/` | Seven beginner diagrams: five technical walkthroughs, the course map and the recap, plus additional lecture flows where authored, as PNG and editable SVG. The notebooks embed the relevant diagrams for portable use in Colab. |

The course folder (`00_Course`) contains this guide, the common Course Glossary, the two copies of the module descriptor (clean and tracked changes) and the Section 7.3 rationale.

Every speaker note follows the same six labels: **In short**, **Explain**, **Example**, **Ask**, **Answer** and **Watch out**. Its 110–180 words include the labels. The Explain section has three to six plain sentences of at most twenty words. Use the Example to trace concrete values or tensor shapes, and reveal the Answer after students predict.

## A typical week

For students who are new to the domain, begin with the starting vocabulary and the overall flow. Ask them to name the input, predict what one block produces, and follow the labelled data to the next block. Explain each detailed diagram beside its related lecture topic before introducing the equation. The six supporting slides fit within the existing lecture segments, and the weekly notes show the updated slide references. A solid arrow carries data to another block. A dashed arrow shows a step that repeats.

Before the lab code, students trace one example through the embedded diagrams and the step-to-code map. Ask what changes during training and what stays fixed when the model produces an output. Use the worked examples and plain-language glossary in the Beginner walkthrough section of the Word notes when a student cannot explain what a line of code is doing. Describe each metric's limits before interpreting a score.

Teach each new idea in this order: why the task matters, a familiar analogy, a small worked example, the diagram, then the technical term or equation. Read an equation aloud and explain every symbol before asking students to calculate. In the lab, ask students to read the short explanation, predict an output, run the cell, change one setting, and check what changed. Give students time to describe the result in their own words. Keep the full theory available in the notes for revision. If students need more time to trace a diagram, keep the vocabulary and worked example; shorten a later quiz or move an advanced extension to the lab.

1. **Before the lecture (about 1 hour).** Read the teaching notes, especially the lecture plan, misconceptions and the teaching tip. Open the slides in Presenter View to see the speaker notes. Run the lab solution notebook once on Colab so that models are cached and you know the runtime.
2. **Lecture (2 hours).** The slides have four or five sections, one to four quick-check quizzes and a warm-up discussion. Slides marked "real" or "measured" show saved lab observations; identify the recorded settings and help students trace each measurement to its code. Repeating an experiment on another system may produce different values.
3. **Lab (2 hours).** Students open the student notebook in Colab with its default CPU runtime, or in VS Code/Jupyter. The installation cell is followed by shared setup and a readiness check. Students trace the diagrams, predict results, complete each TODO using its hint and run the corresponding `check_…()` helper. Each part includes a core task, a change-one-setting task, an optional stretch and common errors. Week 8 performs real bounded LoRA training on CPU. GPU experiments, longer training and hosted calls are optional stretches.
4. **After the lab.** Release the solution notebook. Use the practice questions in the notes for revision sessions.

# Course map

{{WEEK_MAP}}

# Module learning outcomes (MIMLOs) across the weeks

{{MIMLO_MATRIX}}

Every MIMLO is practised in at least three weeks. MIMLO 5 (deployment, RAG and agents) is concentrated in weeks 11–12 but prepared in weeks 6–7 (APIs, prompting, evaluation), which suits the timing of the group project.

# Assessment alignment

The descriptor's assessment is unchanged: a **group project (60%, MIMLOs 1–5)** and a **two-hour proctored examination (40%, MIMLOs 1–3)**.

* **Group project (groups of two to three).** Each week's notes end with a "Link to the group project" section that states what the week contributes (evaluation method, fine-tuning decision, deployment evidence, RAG evaluation set). A suggested milestone plan: groups and problem statement in week 3; proposal with data, baseline and evaluation plan in week 5; working prototype with a first evaluation in week 9; final demonstration and report after week 12. Weeks 7, 11 and 12 give students reusable code for evaluation harnesses, deployment and RAG.
* **Examination.** Each week's notes contain four or five multiple-choice questions, four short-answer questions with marks and model answers, and an exam-style question with a marking guide. These are designed to be reused or adapted for the examination and for formative quizzes on Moodle.
* **Academic integrity and AI use.** Week 1 sets the rules for using AI tools: students may use them where permitted but must declare the tool, the purpose and the prompts in an appendix; undeclared AI-generated work is academic misconduct. Align the wording with the institute's current policy.

# Lab environment and accounts

* **Default platform:** free Google Colab with its CPU runtime, or local VS Code/Jupyter. Shared setup selects CPU by default, fixes seeds and prints the device and package versions. Keep the required bounded core separate from optional GPU, media-generation and longer-training stretches.
* **Accounts:** a Google account for Colab; a Hugging Face account only when the selected download or optional service needs one. Required local tasks need no model-provider API key. To try a hosted comparison, enable exactly one shared switch: `USE_GEMINI = True` or `USE_OPENAI = True`. Both default to `False`; credentials alone do not enable requests. `ask_model()` returns optional text or `None`, and each exercise retains a local answer or rubric.
* **Keys and provider choice:** Gemini uses `google-genai` and `GEMINI_API_KEY`; OpenAI uses its SDK and `OPENAI_API_KEY`. Store keys in Colab Secrets or environment variables. Local users may explicitly select a private file with `LOCAL_ENV_FILE`; setup reads it only after provider opt-in. Use invented, non-personal teaching prompts. Check current account, region, quota and data terms before a call. Gemini free-tier access is account-dependent; OpenAI API charges are separate from a ChatGPT subscription.
* **Local use:** install the packages in the notebook installation cell into a compatible Python environment; the recorded local validation environment uses Python 3.12. Widgets change a setting or cached display without repeating model calls in callbacks. Cells tagged `skip-test` are interactive or optional and are excluded from automated timing. Week 11 uses in-process FastAPI TestClient in its core; Gradio and remote hosting remain optional.

The [official Colab FAQ](https://research.google.com/colaboratory/faq.html) explains that hardware and usage limits vary. The table selects the latest successful full CPU core for each week from `iteration_3_full_runs` in `research/lab_runs_2026-10.json`. These are cached local laptop measurements, with package installation, downloads, interactive widgets and optional provider/stretch routes excluded. Hardware, threads, dates, package versions and notebook hashes remain in that ledger. A first download, clean installation or different computer can take longer; no timed Colab run is recorded.

{{LAB_SUMMARY}}

{{LAB_TABLE}}

## Archived earlier full experiments

The following summary belongs to the larger experiments measured on 9 October 2026. Its hardware guidance and durations describe that earlier configuration. Use it when interpreting the labelled stored plots; plan the required classroom core with the current CPU table above. New CPU exports are stored separately in `results_cpu_core.json`.

{{ARCHIVED_LAB_TABLE}}

# Models, data and licences used in the labs

The local models are open-weight and download from the Hugging Face Hub when a notebook first runs. Check the exact release's model or dataset card and full licence for the intended use; a short licence label cannot describe every condition. The listed MMS-TTS release uses **CC BY-NC 4.0**, so its non-commercial condition matters when planning projects. Optional hosted APIs have separate service terms.

{{MODEL_TABLE}}

The publisher cards for [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) specify Apache 2.0, checked on 11 October 2026. Keep checkpoint identity and the full licence with a project; archived or optional models may use different terms.

# How the materials were produced and checked

* **Sources.** Content draws on the module descriptor, original papers, official documentation and courses (OpenAI, Google, Microsoft Learn, Anthropic, Hugging Face), and the textbooks in descriptor Section 7.9. The link check was rerun in October 2026 (YouTube links by their video titles). Some publishers block automated checks; use the link report to identify pages that need a manual check.
* **Real results.** Figures and numbers marked "real" or "measured" use saved lab measurements from the local laptop (NVIDIA GTX 1650 GPU with 4 GB, or its CPU). Keep the recorded model, data and settings beside any comparison. Student results may differ with hardware, library versions and random sampling; use differences as observations to investigate. Illustrative arithmetic is labelled separately. The revision report distinguishes fresh executions from retained historical measurements.
* **Automated checks.** Smoke mode tests small settings; full CPU mode measures the required core. The checks cover the authored slide count, strict speaker-note template, Section 7.3 coverage, editable diagrams, notebook attachments, Python syntax, semantic TODO helpers and student/solution separation. Diagram checks cover labelled routes, operation roles and shared geometry. After a rebuild, inspect every rendered slide, Word page and diagram beside the text-bound reports. Final release gates and rendered review must be recorded against the current sources.
* **Model names.** The descriptor and slides name model **families** (GPT, Gemini, Claude, Llama, Qwen, Mistral, DeepSeek) rather than versions, which change every few months. Labs pin specific open models so that results are reproducible.

# Keeping the pack up to date

The pack is generated from editable sources, so updates do not require editing PowerPoint or Word files by hand.

| To change | Edit | Then run |
|---|---|---|
| Section 7.3 text | `curriculum/section_7_3.yaml` | `tools/build_descriptor.py` and rebuild the weeks |
| Slides and speaker notes | `curriculum/weeks/week_XX.yaml` | `tools/build_week.py --weeks XX` |
| Teaching notes | `curriculum/notes/week_XX.md` | `tools/build_week.py --weeks XX` |
| Beginner walkthroughs | `curriculum/beginner/week_XX.json` | `tools/build_week.py --weeks XX` |
| Lab notebooks | `curriculum/labs/week_XX_lab.py` | `tools/build_week.py --weeks XX` then `tools/test_labs.py --weeks XX` |
| Diagrams | `curriculum/figures/week_XX.py` | `tools/build_week.py --weeks XX` |
| Measured results | rebuild the week, run `tools/test_labs.py --weeks XX --full --device cpu` with `GENAI_RESULTS_PATH` pointing to a scratch `results_cpu_core.json`, compare with the stored evidence and record the run | promote the separate CPU asset, `tools/fill_results.py --weeks XX`, then rebuild; harvest figures only from the matching successful full notebook |
| Links | `curriculum/sources.yaml` and the files above | `tools/check_links.py` |

**Yearly refresh checklist:** check exact model cards and shared API model names, re-run all required CPU cores and measure a clean Colab installation, update the families tables in weeks 6 and 10, re-check prices used as examples in week 11, review the dated legal references in weeks 1, 3, 4 and 10, and run the link checker. Optional provider calls need their own evidence.

# Known limitations

* Small open models (0.1–2 billion parameters) are used so that labs run on free hardware. They make visible mistakes: this is deliberate, because students learn evaluation by finding them, but set expectations in the first lab.
* Some demonstrations sample their outputs (week 6's thinking on/off comparison, week 7's self-consistency), so answers vary from run to run. Repeat sampling with several seeds and report counts before drawing conclusions; the quick CPU routes use fewer repeats.
* Hosted API model names, quotas and free-tier eligibility change. Both provider switches start disabled, smoke tests skip provider requests, and an unavailable optional request returns to the local fallback. A local CPU run does not verify live provider access.
* Weeks 4 and 10 use SD-Turbo under the Stability AI Community License, which permits limited commercial use subject to registration, revenue thresholds and other conditions, and week 10 uses MMS-TTS (CC BY-NC); week 4 uses Stable Diffusion 1.5 under the CreativeML OpenRAIL-M licence, which restricts harmful uses.

# Use the final infographic to improve the next explanation

Reserve the last five minutes of each class for its recap infographic. Ask students to explain one card in their own words, trace the illustrative example and answer the exit question before showing the answer. Distinguish illustrative arithmetic from measured lab results. If a student names a term but cannot trace its input and output, return to that diagram and use a smaller example.

Keep a short teaching log: the question that caused difficulty, the misconception heard, the revised example and whether students could explain it afterwards. Begin the next class with that revised example and the course-map connection. This gives the teacher a repeatable improvement loop without adding lecture slides. The speaker notes supply analogies, class questions and common mistakes; the Word notes supply the full equations and code omitted from the projected slides.

Use `Course_Glossary.docx` as the common vocabulary reference. Its definitions come from `curriculum/glossary.json`, which also supplies the weekly starting vocabulary and quick glossaries. Context-specific examples in each week explain how an encoder, decoder or embedding is used in that particular model.

Before distributing an update, read `research/REVISION_2026-10.md` and the current task handoffs for the actual verification scope. The run ledger contains all twelve measured CPU cores. The revision records list the release checks, rendered image review and verification limits. Local execution does not establish clean Colab installation, GPU-stretch performance or successful live provider calls.

The SD-Turbo licence was checked against its [official licence file](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md) on 9 October 2026. Read the full terms for the intended use.
