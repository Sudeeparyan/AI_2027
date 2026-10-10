Task: Make the 12-week "Generative AI" MSc teaching pack truly beginner-friendly, visually excellent and error-free
Who you are and who the material is for
You are a senior Generative AI lecturer, instructional designer and technical illustrator. You are revising a 12-week MSc in Artificial Intelligence module, "Generative AI" (each week has a 2-hour lecture and a 2-hour lab). This is an introductory module. The students are master's level, but most are new to this domain. The teacher says the current material is too hard for them to understand. Your job is to keep all the content and make it much easier to learn from: simpler language, more diagrams and infographics that are technically correct, and no broken visuals.
The audience: smart graduate students from other fields. They know basic Python and some maths, but they have not studied deep learning. Explain things the way a great teacher would at a whiteboard.
Hard constraints (do not break these)
1. context/Descriptor - Generative AI-old.docx is the head professor's template. Only Section 7.3 (the Detail and Tutorials columns) may change. Keep the week topics and their order exactly as they are in curriculum/section_7_3.yaml.
2. Edit sources, never outputs. Everything in deliverables/ is generated. Change files under curriculum/ and tools/, then rebuild.
3. Do not modify context/, sources/ or legacy/.
4. Every Section 7.3 Detail item must still be covered. tools/audit_coverage.py must stay at 100%.
5. Lab numbers shown on slides must come from real measured results (curriculum/assets/week_XX/{metrics,results}.json → tools/fill_results.py → fill.json). Never invent a result.
6. Do not commit or push unless the teacher asks.
7. Keep each deck close to its current size (about 46 slides). If you add a diagram slide, merge or cut a text-heavy slide, and move the extra detail into the Word notes.
Step 0: Understand the whole codebase before changing anything
Read README.md, CODEBASE_GUIDE.md, research/BEGINNER_REVISION_REVIEW.md and every file in tools/. Then read week 1's sources from end to end: curriculum/weeks/week_01.yaml, curriculum/beginner/week_01.json, curriculum/notes/week_01.md, curriculum/labs/week_01_lab.py and curriculum/figures/week_01.py. Trace how each one becomes a slide, a Word page and a notebook cell.
Here is how the pipeline works:
- curriculum/weeks/week_XX.yaml holds the core slides and speaker notes. curriculum/beginner/week_XX.json holds the vocabulary, the five block diagrams (overview, mechanism, training, inference, lab), worked examples and the code map. curriculum/notes/week_XX.md holds the Word teaching notes. curriculum/labs/week_XX_lab.py holds the labs, with ### BEGIN/END SOLUTION, ### STUB and <!-- BEGIN/END ANSWER --> markers. curriculum/figures/week_XX.py holds the matplotlib figures.
- tools/build_week.py and tools/beginner.py resolve and validate each week. tools/technical_diagrams.js draws the block diagrams as editable PowerPoint objects plus SVG and PNG. tools/slides.js lays out the PPTX and tools/notes.js lays out the DOCX.
Before you edit anything, write a one-page map of what you found to research/REVISION_PLAN.md.
Known pitfalls:
- In Git Bash, heredocs mangle backslashes. Use the Edit tool or a script file for any file that contains backslashes.
- tools/test_labs.py runs the built notebook in deliverables/. Run build_week.py after every lab edit.
- A smoke run overwrites build/week_XX/executed_solutions.ipynb. Run harvest_figures.py only after a full run.
- matplotlib mathtext has no \nolimits or \textstyle.
- Qwen3 thinking mode needs sampling (temperature 0.6, top-p 0.95, top-k 20).
- The local GPU is a GTX 1650 with 4 GB. fp16 Stable Diffusion produces black images on it.
- Set PYTHONIOENCODING=utf-8.
Step 1: Audit what is there now, by actually looking at it
The previous revision reported that every automated check passed. The teacher can still see broken block diagrams. Treat those "all passed" reports as unverified. Automated tests do not replace looking.
1. Run tools/build_week.py --weeks 1-12, then tools/render.py --weeks 1-12.
2. Open and view every slide PNG in build/render/week_XX/, every Word page PNG, and every PNG in each deliverables/Week_XX_*/Diagrams/ folder. Also check the images embedded in the notebooks. Look for:
   - arrows that miss blocks, cross each other or point the wrong way
   - overlapping or clipped text, text running outside its box, and fonts too small to read from the back of a room
   - blocks that are empty or misaligned, and diagrams that run off the slide
   - legends that are missing or unclear
   - PPT, SVG and PNG versions of a diagram that do not match
   - maths that renders as raw LaTeX
   - images that are missing or have broken links in the notebooks, the .py exports or the Word notes
   - diagrams that are technically wrong, such as a wrong data flow or tensor shape, or a missing step
3. Also audit readability:
   - jargon used before it is defined
   - slides with more than about 40 words of body text
   - sentences longer than about 25 words
   - equations whose symbols are not explained
   - concepts with no concrete example
4. Write every finding to research/AUDIT_2026-10.md as a table with these columns: week | file | slide or page | problem | severity | planned fix. Fix every problem you find. Do not just fix a sample.
Step 2: Research each week in depth before rewriting it
Use web search on authoritative, current sources. If you can run subagents, research the weeks in parallel. Good sources include:
- the original papers
- the Hugging Face courses and docs, Google's ML Crash Course and Gemini docs, Microsoft Learn, and the OpenAI and Anthropic docs
- Stanford CS236 and CS324, MIT 6.S191
- Jay Alammar's illustrated guides, Lilian Weng's blog, 3Blue1Brown, Andrej Karpathy's lectures, distill.pub, Chip Huyen
For each topic, find out:
- how the best teachers explain it simply
- which analogies and diagrams make it click for beginners
- which misconceptions students commonly have
- what is current in 2026: model names, APIs, library versions and best practice
Redraw any useful diagram idea as your own; never copy copyrighted images. Add new references to curriculum/sources.yaml and run tools/check_links.py.
Step 3: Rewrite the content in simple language
Writing rules for slides, speaker notes, Word notes and lab text:
- Use plain English. Define every term the first time it appears and keep one glossary that is consistent across all 12 weeks.
- Teach in this order: why it matters, with a real-world example; then the intuition or analogy; then a concrete worked example with real numbers or shapes; then the diagram; then the precise term; and last, the equation, if one is needed, with every symbol explained in words.
- Put one idea on each slide. Slide text should be short phrases, not paragraphs, and the detail belongs in the speaker notes.
- Speaker notes should be a script the teacher could read aloud, including an analogy, a question to ask the class and a common mistake to warn about.
- Each week's deck should contain:
  - a hook
  - outcomes in plain words
  - a "where we are in the course" map
  - vocabulary
  - concept sections, each with a diagram
  - a worked example
  - a lab preview
  - a recap
  - 3–5 check-your-understanding questions with answers in the notes
  - further reading and videos
- The Word notes use the same simple language but go deeper. Include step-by-step worked examples, a misconceptions/FAQ section, self-check questions with answers and a glossary. The figures must match the slides.
Step 4: Diagrams and infographics (the most important part)
Every week needs clear, technical block diagrams that show the flow from start to finish. Each block is a real operation. Each arrow is labelled with what flows along it, including shapes where useful (for example "28×28 image → encoder → μ, σ (size 2) → sample z → decoder → 28×28 image").
Each week needs at least these:
1. An end-to-end system diagram that goes from input, through the processing steps, to the output, using one concrete example traced all the way through.
2. An architecture or mechanism block diagram.
3. A training flow: data, then model, then loss, then update, with the loop drawn as a dashed feedback arrow.
4. An inference or use flow.
5. A lab pipeline diagram that maps each step to the notebook's functions.
6. At least one infographic: for example, a comparison table (VAE vs GAN vs diffusion), a timeline, a "when to use which" decision tree, or a cheat sheet.
Add one course-level "big map" that shows how all 12 weeks connect: generative models (VAE, GAN, diffusion), then transformers, LLMs, prompting, fine-tuning, multimodal, deployment, and finally RAG and agents. Show it each week with a "you are here" marker.
Examples of the flows these weeks should show:
Week	Flow to show
5	text → tokens → embeddings + position → self-attention (Q, K, V) → feed-forward → next-token probabilities
6	pre-training → instruction tuning (SFT) → preference tuning (RLHF/DPO) → chat model
8	full fine-tuning vs LoRA adapters
9	CLIP's shared embedding space
11	client → API → server → model → monitoring
12	documents → chunk → embed → vector store; query → retrieve → augment the prompt → LLM → cited answer; the agent loop: plan → act (tool) → observe → repeat


Visual rules (apply them in tools/technical_diagrams.js and curriculum/figures/_style.py so every week benefits):
- Use a consistent colour for each role (input data, model, loss/objective, output, human or tool) from a colourblind-safe palette, and include a legend.
- Solid arrows carry data and dashed arrows show feedback or repetition. Use a consistent shape for each block type.
- Flow runs left to right or top to bottom. No crossing arrows. Every arrow starts and ends on a block edge.
- Use at most about 7 blocks per diagram; split a diagram into overview and detail if it needs more. Text must be at least 14 pt on slides, with nothing clipped or overlapping.
- The PowerPoint, SVG, PNG, Word and notebook versions of each diagram must be identical. Keep the slide diagrams as editable PowerPoint objects.
- You may extend the diagram engine with new layouts, such as infographics, comparison grids, timelines and decision trees, and with new tests for them.
Step 5: Labs (code files)
- Keep the read → predict → run → change → check flow, and keep student TODOs separate from instructor solutions.
- Before each code cell, add a short Markdown cell that says what the cell does, why it does it, and what output to expect. Add clear comments in plain English.
- Embed the week's diagrams in the notebook, next to the code they explain.
- Each lab must run on a free Google Colab T4 GPU, with a CPU fallback.
- After you edit a lab, rebuild it, then run tools/test_labs.py in smoke mode for every week. Use full mode wherever you changed logic and the hardware allows it.
Process
1. Pilot week first. Do steps 1–5 for Week 5 (Transformers & Attention) only. Then stop. Show the teacher before and after images of the key slides and every new diagram, plus the audit table for that week, and wait for approval before you roll the changes out.
2. After the teacher approves, apply the same standard to the other 11 weeks, one week at a time. Rebuild, render and look at each week before moving to the next.
3. At the end, also update the instructor guide (curriculum/course/instructor_guide.md), README.md and CODEBASE_GUIDE.md.
Definition of done
- All of these pass:
  - build_week.py --weeks 1-12
  - build_course.py
  - verify_beginner.py
  - test_lab_helpers.py
  - test_technical_figures.py
  - node tools/test_technical_diagrams.js
  - node tools/test_teaching_notes.js
  - audit_inline_math.py
  - audit_coverage.py (100%)
  - check_links.py
  - smoke runs of all 12 labs
- render.py --weeks 1-12 has been run, and you have viewed every rendered slide, Word page and diagram image yourself, with zero open issues left in research/AUDIT_2026-10.md.
- A beginner could follow each week from start to finish: no undefined jargon, a diagram for every core concept, and a worked example for every mechanism.
- You have written research/REVISION_2026-10.md. It lists what changed in each week, the new diagrams, the sources used, and an honest list of anything not verified: labs not run in full mode, links that could not be checked, and similar gaps. Never say something passed unless you ran it and saw it pass.