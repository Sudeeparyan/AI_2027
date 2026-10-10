# Beginner teaching-pack revision

Reviewed and rebuilt on 7 October 2026.

All 12 weeks now introduce ideas through a concrete input and output, a technical flow, a worked example, and then the terminology or equation. The revised files are in [deliverables](../deliverables/). Start with the [instructor guide](../deliverables/00_Course/Instructor_Guide.docx) and [Week 1 slides](../deliverables/Week_01_Intro_Generative_and_Responsible_AI/Week_01_Lecture_Slides.pptx).

## What changed

- Simplified slide explanations, speaker notes, weekly summaries and Word explanations. Added starting vocabulary, section examples, symbol explanations and checks that ask students to trace an operation.
- Expanded the shared technical walkthroughs from three to five per week: **60 diagrams, including 24 new diagrams**. Each flow names operations, the data passed between them, and relevant lab functions or variables. Training or preparation and later use have separate explanations.
- Added matching diagrams, worked examples and code maps to Word notes and notebooks. Notebook images are embedded, so they travel with the file into Colab. Python exports link to the adjacent `Diagrams/` folder.
- Added explanatory comments and a read, predict, run, change and check sequence throughout all 12 lab sources. Student TODOs and instructor answers remain separate.
- Repaired shared arrow routing and feedback paths, U-Net and DiT connections, transformer residual connections, multimodal token connections, adaptation decision branches, agent observations and diagram borders. PowerPoint walkthrough blocks remain editable; every flow also has a PNG and SVG export.
- Corrected misleading explanations of GAN gradients, VAE sampling, CLIP loss and retrieval metrics, fine-tuning methods, theoretical weight storage, application request paths, and simulated agent approval. Updated relevant legal and hosting references against official sources.
- Fixed Word image metadata, paragraph and callout pagination, equation/table rendering, slide text sizing and stale slide references. Updated [CODEBASE_GUIDE.md](../CODEBASE_GUIDE.md) to explain how the sources produce the teaching files.

The 40 core slide titles and their code, equation and figure references are preserved for every week. Each rebuilt deck has **46 slides**: the core 40, starting vocabulary and five walkthroughs. The syllabus topics, week order and assessment structure are preserved. Lab executable Python syntax trees match the original sources after removing notebook install magics; the lab edits add teaching guidance.

## Checks completed

| Check | Result |
|---|---|
| Weekly build validation | 12/12 passed; 552 slides; no build errors |
| Beginner-pack verification | 12/12 passed, seven checks per week; six helper self-tests passed |
| Required Section 7.3 Detail items | 214/214 found in slides or notes by the coverage audit |
| Shared diagram routes | All 60 authored diagrams and 15 feedback cases passed |
| Conceptual figure and lab-helper regressions | 16 tests passed |
| Teaching-note regressions and inline maths | Passed; zero inline expressions flagged |
| Office rendering and visual review | All 12 decks, all 12 weekly Word files and the instructor guide opened and rendered; reviewed slide and page images |
| Measured layout | Zero slide text-bound issues across 12 decks; zero out-of-margin text/image issues across 262 Word-rendered pages |
| Python exports and student/solution separation | All 24 exports regenerated and syntax checked; embedded diagram content and answer separation verified |
| Representative solution-notebook smoke runs | Weeks 1, 4, 5 and 12 passed in 48.6, 73.1, 118.8 and 75.5 seconds respectively |
| Reference links | 139 checked: 135 accessible/title-verified, four automated checks blocked; no hard failures |
| Git whitespace check | Passed |

Machine reports and preview images are in the ignored `build/` folder: `beginner_verification.json`, `link_check.json`, `notes_layout_review.json`, weekly `qa.json`, `lab_test.json`, and `render/week_XX/`.

## Validation limits

The fresh execution checks cover four representative labs in smoke mode. Full training settings, every week's complete execution, and optional hosted API or interactive cells were not rerun during this revision. Existing measured figures retain their recorded instructor-run provenance. The coverage audit confirms that required items appear; it does not measure students' learning.

Four pages block automated requests: the Foster and Tunstall O'Reilly books, the Rothman Packt book, and the ACM *Stochastic Parrots* paper. Their links remain in the source ledger and are marked for manual checking in the link report.

For future edits, change the curriculum sources and rebuild. The [README](../README.md) and [codebase guide](../CODEBASE_GUIDE.md) list the relevant files and commands.
