# How to use this teaching pack

This pack delivers the 12-week **Generative AI** module of the MSc in Artificial Intelligence as described in the module descriptor, with the proposed Section 7.3 revision (see the separate rationale document). Every week follows the same pattern: a 2-hour lecture and a 2-hour lab.

| Each week's folder contains | Use it for |
|---|---|
| `Week_XX_Lecture_Slides.pptx` | 46 slides with full speaker notes (View ▸ Notes), including the course map, starting vocabulary, five technical walkthroughs and a final recap infographic. |
| `Week_XX_Teaching_Notes.docx` | Your reference: lecture plan with timings, explanations, maths and worked examples, misconceptions, responsible-AI lens, lab answers, practice and exam-style questions with model answers, glossary, readings. |
| `Week_XX_Lab.ipynb` | Student notebook for the 2-hour lab (Google Colab or local Jupyter). Complete each TODO before running cells that depend on it. |
| `Week_XX_Lab_Solutions.ipynb` | Instructor version with complete code and model answers. Do not distribute before the lab. |
| `Week_XX_Lab.py` and `Week_XX_Lab_Solutions.py` | The same commented code in percent-format Python cells for VS Code or a local Python environment. |
| `Diagrams/` | Seven shared images: five technical diagrams, the course map and the recap, as PNG and editable SVG. The notebooks embed every diagram for portable use in Colab. |

The course folder (`00_Course`) contains this guide, the common Course Glossary, the two copies of the module descriptor (clean and tracked changes) and the Section 7.3 rationale.

## A typical week

For students who are new to the domain, begin with the starting vocabulary and the overall flow. Ask them to name the input, predict what one block produces, and follow the labelled data to the next block. Explain each detailed diagram beside its related lecture topic before introducing the equation. The six supporting slides fit within the existing lecture segments, and the weekly notes show the updated slide references. A solid arrow carries data to another block. A dashed arrow shows a step that repeats.

Before the lab code, students trace one example through the embedded diagrams and the step-to-code map. Ask what changes during training and what stays fixed when the model produces an output. Use the worked examples and plain-language glossary in the Beginner walkthrough section of the Word notes when a student cannot explain what a line of code is doing. Describe each metric's limits before interpreting a score.

Teach each new idea in this order: why the task matters, a familiar analogy, a small worked example, the diagram, then the technical term or equation. Read an equation aloud and explain every symbol before asking students to calculate. In the lab, ask students to read the short explanation, predict an output, run the cell, change one setting, and check what changed. Give students time to describe the result in their own words. Keep the full theory available in the notes for revision. If students need more time to trace a diagram, keep the vocabulary and worked example; shorten a later quiz or move an advanced extension to the lab.

1. **Before the lecture (about 1 hour).** Read the teaching notes, especially the lecture plan, misconceptions and the teaching tip. Open the slides in Presenter View to see the speaker notes. Run the lab solution notebook once on Colab so that models are cached and you know the runtime.
2. **Lecture (2 hours).** The slides have four or five sections, one to four quick-check quizzes and a warm-up discussion. Slides marked "real" or "measured" show saved lab observations; identify the recorded settings and help students trace each measurement to its code. Repeating an experiment on another system may produce different values.
3. **Lab (2 hours).** Students open the student notebook in Colab (Runtime ▸ Change runtime type ▸ T4 GPU), trace the initial diagrams, and complete the TODOs and written questions. The first executable cell installs packages. CPU settings use smaller models or fewer iterations where available. Prepare the full Week 8 CPU experiment before class or discuss its recorded results.
4. **After the lab.** Release the solution notebook. Use the practice questions in the notes for revision sessions.

# Course map

{{WEEK_MAP}}

<!-- pagebreak -->

# Module learning outcomes (MIMLOs) across the weeks

{{MIMLO_MATRIX}}

Every MIMLO is practised in at least three weeks. MIMLO 5 (deployment, RAG and agents) is concentrated in weeks 11–12 but prepared in weeks 6–7 (APIs, prompting, evaluation), which suits the timing of the group project.

# Assessment alignment

The descriptor's assessment is unchanged: a **group project (60%, MIMLOs 1–5)** and a **two-hour proctored examination (40%, MIMLOs 1–3)**.

* **Group project (groups of two to three).** Each week's notes end with a "Link to the group project" section that states what the week contributes (evaluation method, fine-tuning decision, deployment evidence, RAG evaluation set). A suggested milestone plan: groups and problem statement in week 3; proposal with data, baseline and evaluation plan in week 5; working prototype with a first evaluation in week 9; final demonstration and report after week 12. Weeks 7, 11 and 12 give students reusable code for evaluation harnesses, deployment and RAG.
* **Examination.** Each week's notes contain four or five multiple-choice questions, four short-answer questions with marks and model answers, and an exam-style question with a marking guide. These are designed to be reused or adapted for the examination and for formative quizzes on Moodle.
* **Academic integrity and AI use.** Week 1 sets the rules for using AI tools: students may use them where permitted but must declare the tool, the purpose and the prompts in an appendix; undeclared AI-generated work is academic misconduct. Align the wording with the institute's current policy.

# Lab environment and accounts

* **Default platform:** Google Colab. Select a GPU if one is available. CPU settings use smaller models or fewer steps where possible, but heavy image generation and full fine-tuning can be impractical on CPU. Check each week's runtime note and prepare recorded outputs when classroom hardware is limited. Colab GPU availability is not guaranteed.

The [official Colab FAQ](https://research.google.com/colaboratory/faq.html) explains that hardware and usage limits vary. The times below were measured on 9 October 2026 by running each complete solution notebook on the laptop used to check this pack: its CPU, or its NVIDIA GTX 1650 GPU (4 GB) where the table says so. Models and data were already downloaded, so first-run downloads add time. They are not Colab T4 timings; a T4 is usually faster than the GTX 1650, and classroom hardware can change them substantially. Week 8 is the slowest on a CPU (training took 53 to 76 minutes in two runs); prepare its saved results when a GPU is unavailable.
* **Accounts:** a Google account for Colab; a Hugging Face account when a model or hosted service requires it. API keys are optional: `GEMINI_API_KEY` and `HF_TOKEN` are used in weeks 1, 6 and 11 for comparisons with hosted models. Check the service's current access and quota. Keys are read only from Colab **Secrets** (🔑 icon), never typed into notebooks.
* **Local use:** any Python 3.11+ environment with the packages in each notebook's first cell. Week 11 also shows Ollama for local models.

{{LAB_TABLE}}

# Models, data and licences used in the labs

The local models are open-weight and download from the Hugging Face Hub when a notebook first runs. Check the exact release's model or dataset card and full licence for the intended use; a short licence label cannot describe every condition. The listed MMS-TTS release uses **CC BY-NC 4.0**, so its non-commercial condition matters when planning projects. Optional hosted APIs have separate service terms.

{{MODEL_TABLE}}

# How the materials were produced and checked

* **Sources.** Content draws on the module descriptor, original papers, official documentation and courses (OpenAI, Google, Microsoft Learn, Anthropic, Hugging Face), and the textbooks in descriptor Section 7.9. The link check was rerun in October 2026 (YouTube links by their video titles). Some publishers block automated checks; use the link report to identify pages that need a manual check.
* **Real results.** Figures and numbers marked "real" or "measured" use saved lab measurements from the local laptop (NVIDIA GTX 1650 GPU with 4 GB, or its CPU). Keep the recorded model, data and settings beside any comparison. Student results may differ with hardware, library versions and random sampling; use differences as observations to investigate. Illustrative arithmetic is labelled separately. The revision report distinguishes fresh executions from retained historical measurements.
* **Automated checks.** The lab runner executes solution notebooks in a quick "smoke" mode. Every week is checked for its authored slide count, speaker notes, required teaching-note sections and coverage of Section 7.3. The beginner-pack check verifies editable slide diagrams, notebook attachments, Python syntax and student/solution separation. The diagram check verifies that arrows touch their blocks. Render the slides and notes after a rebuild and inspect the page images and measured text bounds.
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
| Measured results | rebuild the week (the runner executes the built notebook), then re-run it with `tools/test_labs.py --full` and `GENAI_RESULTS_PATH` set, or run `curriculum/assets/make_week_XX.py` | `tools/harvest_figures.py --week XX` (weeks 9–12), `tools/fill_results.py --weeks XX`, then rebuild |
| Links | `curriculum/sources.yaml` and the files above | `tools/check_links.py` |

**Yearly refresh checklist:** check model availability and names in the labs (especially API model names in weeks 1, 6 and 11), re-run all labs on Colab, update the "families" tables in weeks 6 and 10, re-check prices used as examples in week 11, review legal references (EU AI Act guidance, copyright cases) in weeks 1, 3, 4 and 10, and run the link checker.

# Known limitations

* Small open models (0.1–2 billion parameters) are used so that labs run on free hardware. They make visible mistakes: this is deliberate, because students learn evaluation by finding them, but set expectations in the first lab.
* Some demonstrations sample their outputs (week 6's thinking on/off comparison, week 7's self-consistency), so answers vary from run to run; the notebooks run several seeds and report counts, and students should do the same before drawing conclusions.
* Hosted API model names and free-tier limits change often; the notebooks skip the optional API cells when no key is present and print how to list available models.
* Weeks 4 and 10 use SD-Turbo under the Stability AI Community License, which permits limited commercial use subject to registration, revenue thresholds and other conditions, and week 10 uses MMS-TTS (CC BY-NC); week 4 uses Stable Diffusion 1.5 under the CreativeML OpenRAIL-M licence, which restricts harmful uses.

# Use the final infographic to improve the next explanation

Reserve the last five minutes of each class for its recap infographic. Ask students to explain one card in their own words, trace the illustrative example and answer the exit question before showing the answer. Distinguish illustrative arithmetic from measured lab results. If a student names a term but cannot trace its input and output, return to that diagram and use a smaller example.

Keep a short teaching log: the question that caused difficulty, the misconception heard, the revised example and whether students could explain it afterwards. Begin the next class with that revised example and the course-map connection. This gives the teacher a repeatable improvement loop without adding lecture slides. The speaker notes supply analogies, class questions and common mistakes; the Word notes supply the full equations and code omitted from the projected slides.

Use `Course_Glossary.docx` as the common vocabulary reference. Its definitions come from `curriculum/glossary.json`, which also supplies the weekly starting vocabulary and quick glossaries. Context-specific examples in each week explain how an encoder, decoder or embedding is used in that particular model.

Before distributing an update, read `research/REVISION_2026-10.md` for the actual verification scope. Local smoke execution is not evidence of a Colab T4 run or a successful optional hosted API call.

The SD-Turbo licence was checked against its [official licence file](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md) on 9 October 2026. Read the full terms for the intended use.
