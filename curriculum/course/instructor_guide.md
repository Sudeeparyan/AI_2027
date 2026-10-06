# How to use this teaching pack

This pack delivers the 12-week **Generative AI** module of the MSc in Artificial Intelligence as described in the module descriptor, with the proposed Section 7.3 revision (see the separate rationale document). Every week follows the same pattern: a 2-hour lecture and a 2-hour lab.

| Each week's folder contains | Use it for |
|---|---|
| `Week_XX_Lecture_Slides.pptx` | 44 slides with full speaker notes (View ▸ Notes), including starting vocabulary and three technical walkthroughs. |
| `Week_XX_Teaching_Notes.docx` | Your reference: lecture plan with timings, explanations, maths and worked examples, misconceptions, responsible-AI lens, lab answers, practice and exam-style questions with model answers, glossary, readings. |
| `Week_XX_Lab.ipynb` | Student notebook for the 2-hour lab (Google Colab or local Jupyter). TODO cells contain runnable starter code. |
| `Week_XX_Lab_Solutions.ipynb` | Instructor version with complete code and model answers. Do not distribute before the lab. |
| `Week_XX_Lab.py` and `Week_XX_Lab_Solutions.py` | The same commented code in percent-format Python cells for VS Code or a local Python environment. |
| `Diagrams/` | Three technical diagrams as PNG and editable SVG. The notebooks embed their diagrams for portable use in Colab. |

The course folder (`00_Course`) contains this guide, the two copies of the module descriptor (clean and tracked changes) and the Section 7.3 rationale.

## A typical week

For students who are new to the domain, begin with the starting vocabulary and the overall flow. Ask them to name the input, predict what one block produces, and follow the labelled data to the next block. Explain the mechanism diagram beside its related lecture topic before introducing its equation. The four additional slides fit within the existing lecture segments, and the weekly notes show the updated slide references.

Before the lab code, students read the two embedded diagrams and the step-to-code map. Use the worked examples and plain-language glossary in the Beginner walkthrough section of the Word notes when a student cannot explain what a line of code is doing. Distinguish model training from inference, and describe each metric's limits before interpreting a score.

1. **Before the lecture (about 1 hour).** Read the teaching notes, especially the lecture plan, misconceptions and the teaching tip. Open the slides in Presenter View to see the speaker notes. Run the lab solution notebook once on Colab so that models are cached and you know the runtime.
2. **Lecture (2 hours).** The slides have five sections, two to four quick-check quizzes and a warm-up discussion. Slides marked "real" or "measured" show outputs produced by running the lab in advance; say so to students, because it shows that the numbers are reproducible.
3. **Lab (2 hours).** Students open the student notebook in Colab (Runtime ▸ Change runtime type ▸ T4 GPU), trace the initial diagrams, and complete the TODOs and written questions. The first executable cell installs packages. CPU settings use smaller models or fewer iterations where available. Week 8 fine-tuning is impractical without a GPU.
4. **After the lab.** Release the solution notebook. Use the practice questions in the notes for revision sessions.

# Course map

{{WEEK_MAP}}

# Module learning outcomes (MIMLOs) across the weeks

{{MIMLO_MATRIX}}

Every MIMLO is practised in at least three weeks. MIMLO 5 (deployment, RAG and agents) is concentrated in weeks 11–12 but prepared in weeks 6–7 (APIs, prompting, evaluation), which suits the timing of the group project.

# Assessment alignment

The descriptor's assessment is unchanged: a **group project (60%, MIMLOs 1–5)** and a **two-hour proctored examination (40%, MIMLOs 1–3)**.

* **Group project (groups of two to three).** Each week's notes end with a "Link to the group project" section that states what the week contributes (evaluation method, fine-tuning decision, deployment evidence, RAG evaluation set). A suggested milestone plan: groups and problem statement in week 3; proposal with data, baseline and evaluation plan in week 5; working prototype with a first evaluation in week 9; final demonstration and report after week 12. Weeks 7, 11 and 12 give students reusable code for evaluation harnesses, deployment and RAG.
* **Examination.** Each week's notes contain four or five multiple-choice questions, four short-answer questions with marks and model answers, and one 20-mark exam-style question with a marking guide. These are designed to be reused or adapted for the examination and for formative quizzes on Moodle.
* **Academic integrity and AI use.** Week 1 sets the rules for using AI tools: students may use them where permitted but must declare the tool, the purpose and the prompts in an appendix; undeclared AI-generated work is academic misconduct. Align the wording with the institute's current policy.

# Lab environment and accounts

* **Default platform:** Google Colab with the free **T4 GPU**. Every lab also runs on a CPU (automatically smaller models or fewer steps), more slowly; only the week 8 fine-tuning run is impractical without a GPU.
* **Accounts:** a Google account for Colab and a free Hugging Face account. API keys are optional: `GEMINI_API_KEY` (Google AI Studio free tier) and `HF_TOKEN` (Hugging Face Inference Providers) are used in weeks 1, 6 and 11 for comparisons with hosted models. Keys are read only from Colab **Secrets** (🔑 icon), never typed into notebooks.
* **Local use:** any Python 3.11+ environment with the packages in each notebook's first cell. Week 11 also shows Ollama for local models.

{{LAB_TABLE}}

# Models, data and licences used in the labs

All models are open-weight and downloaded from the Hugging Face Hub when a notebook first runs. Licences are as stated on the model or dataset cards in September 2026; check the cards before any commercial use. Several image and speech models are licensed for **non-commercial** use only, which is fine for teaching and must be mentioned for projects.

{{MODEL_TABLE}}

# How the materials were produced and checked

* **Sources.** All content is written from primary sources: the module descriptor, original papers, official documentation and courses (OpenAI, Google, Microsoft Learn, Anthropic, Hugging Face) and the textbooks in descriptor Section 7.9. Every link in the slides and notes was checked automatically in September 2026 (YouTube links by their video titles).
* **Real results.** Figures and numbers marked "real" or "measured" come from running the solution notebooks and asset scripts in advance on a laptop (NVIDIA GTX 1650 GPU with 4 GB, or its CPU). Students' numbers on Colab will differ slightly (hardware, library versions); the patterns should match, and the notes say which patterns to expect.
* **Automated checks.** The lab runner executes solution notebooks in a quick "smoke" mode. Every week is checked for 44 slides with speaker notes, required teaching-note sections and coverage of Section 7.3. The beginner-pack check verifies editable slide diagrams, notebook attachments, Python syntax and student/solution separation. Slides and notes were rendered and inspected visually.
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
* Weeks 4 and 10 use SD-Turbo, a research release whose commercial use needs Stability AI's licence, and week 10 uses MMS-TTS (CC BY-NC); week 4 uses Stable Diffusion 1.5 under the CreativeML OpenRAIL-M licence, which restricts harmful uses.
* New students may need more time to trace the diagrams. Keep the vocabulary and worked examples; if a section runs long, shorten a later quiz or move an advanced extension to the lab.
