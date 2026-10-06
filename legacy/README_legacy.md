# Generative AI 2027: reusable teaching project

## Reusable application

Read [the operating guide](docs/APPLICATION_GUIDE.md) for provider/account switching, resumable AI drafting, source ingestion, review and the 2027 development roadmap. `studio.py` manages content authoring; `course.py` builds classroom files; `package_project.py` exports the project. All 11 supplied source documents are retained.

## Start here

Open `outputs/current/index.html` for the slide gallery. Each week has a classroom
PowerPoint, an editable PowerPoint, a Word guide and a notebook. The classroom
PowerPoint embeds the diagram PNGs stored in `assets/diagrams/`. The editable deck
preserves the original PowerPoint shapes. Speaker notes are in both versions.

## Folder map

| Folder or file | What you edit or use |
|---|---|
| `content/week_01/lesson.json` | Weekly goals, case, explanations used in the handout |
| `content/week_01/slides.json` | Exact slide titles, explanations and speaker scripts; stable IDs s01–s40 |
| `content/week_01/visuals.json` | Labels and paths inside the week's block diagrams |
| `content/week_01/lab.ipynb` | Notebook source; edit here before rebuilding |
| `content/references.json` | Shared reference titles and URLs |
| `src/engaging_slides.py` | Forty-slide layouts, pictograms and editable diagrams |
| `src/slides.py` | Earlier renderer and reusable week-opening illustrations |
| `content/teaching_design/` | Authored analogy stories, lesson compiler and generated design JSON |
| `src/handouts.py` | Word document layout |
| `assets/illustrations/` | Generated course illustration, its original prompt and provenance |
| `assets/diagrams/week_01/` | Exported PNG and SVG images; content hashes retain old versions |
| `assets/catalog.json` | Links from slide IDs to the images actually used |
| `outputs/current/` | Latest files for teaching and a visual gallery |
| `outputs/runs/` | Dated builds, reports, editable decks, previews and source snapshots |
| `reviews/` | Per-slide review CSVs and revision notes |
| `course.py` | Build, check, repair, export, reuse and watch commands |

The same structure exists for all 12 weeks. The latest builds are convenient
copies; previous runs remain in the history. Do not edit files inside outputs/
because rebuilding replaces the current copy.

## Install once

Use Python 3.10 or newer. Install LibreOffice and ensure `soffice` is on PATH.
On Windows it is normally inside the LibreOffice `program` directory. It is
required to render slides and export the diagram assets.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this project folder. The script resolves all project paths
relative to its own location, so it also works when launched elsewhere.

## The improvement loop

1. Open the gallery and review a week as a beginner would.
2. Record the issue and a specific change in its review CSV.
3. Edit the source JSON, notebook, or drawing function.
4. Run the improvement command. It checks slide count, note length, shape bounds,
   long-word wrapping and notebook execution. It makes conservative font or
   position fixes, with at most the requested number of passes.
5. Open the new gallery and check diagram meaning, examples and reading order.
6. Mark reviewed rows in the CSV and repeat when you have another improvement.

```bash
# Improve and rebuild just week 2
python course.py improve --weeks 2 --max-passes 3

# Rebuild affected weeks across the course; unchanged weeks are reused
python course.py improve --weeks all --max-passes 3

# Keep watching while you edit: every saved source change triggers a rebuild
python course.py watch --weeks 2 --max-passes 3
# Stop the watcher with Ctrl+C
```

The watch command runs on your computer while it is open. Nothing has been
scheduled to run unattended. It does not call a language model or invent new
teaching content. Automatic checks identify mechanical issues; a teacher checks
accuracy, educational depth and whether a visual helps a beginner.

## Example: improve one explanation

Open `content/week_02/slides.json`. Find `s07`, the VAE mechanism slide, and revise its
`explanation` or `speaker_notes`. To change the diagram labels, edit the first
list under `term` in `content/week_02/visuals.json`. Run the week 2 command above.
The new deck reuses the newly exported diagram PNG and keeps a native editable
version next to it. A dated snapshot records exactly what generated that run.

Changing `lesson.json` updates the handout. Once `slides.json` exists, its text
is the authoritative slide copy. Keep these two files aligned when changing a
teaching claim. Images do not regenerate merely because you rebuild: use a new
versioned image file and update the relevant reference for a new illustration.

## Images and reuse

Precise diagram PNG/SVG files come from the editable PowerPoint render, so their
labels agree with the deck. They are indexed by stable slide ID and content hash.
The week 1 conceptual illustration is in `assets/illustrations/`. Other weeks use editable, week-specific PowerPoint illustrations. The analogy slides also use native pictograms chosen from their example nouns. Its prompt is included for future revisions.
Other visuals are exact block diagrams and worked illustrations, not AI-generated
technical claims. No paid image service or API key is required to rebuild.

## Review limits

The teaching content remains an intermediate module with beginner explanations.
Some labs are simplified stand-ins for trained models, as stated in the notebooks.
The automated loop cannot establish that a course is academically approved or
that a student will understand every slide. Read reports for unresolved issues;
they are never silently marked as fixed. Use the institution-approved assessment brief for summative assessment; the
included questions are practice exercises.

## Syllabus-aligned revision additions

Each week includes study.json with an original worked example, three exam-practice questions and model answers, outcome mapping and selected official resources. The Word guide incorporates this content. research/READING_AND_VIDEO_MAP.md links the sources and videos. content/shared/policy_assistant.py supplies the tagged code cell in weeks 7, 11 and 12 during each build. Edit that shared module to update those cells consistently. instructor_only/ contains the ten separate final cases and runner; do not include that folder in the student distribution.

## Final release check

Run `python verify_release.py` after building. It checks all 12 weekly sets, speaker notes, embedded diagram hashes, exam sections, notebook structure and PDF page bounds. Results are saved in `reviews/FINAL_STRUCTURAL_CHECK.json`. See `reviews/FINAL_QA_REPORT.md` for the latest findings and limits.

## Re-authoring the teaching sequence

`content/teaching_design/author_lessons.py` is the authored set of 48 analogy stories with precise mappings, limits, worked micro-examples, questions and answers. Edit it to change the systematic lesson design, then run:

```bash
python content/teaching_design/author_lessons.py
python content/teaching_design/build_lessons.py
python course.py improve --weeks all --max-passes 3
python verify_release.py
```

The compiler intentionally replaces all weekly `slides.json` files. For a single slide revision, edit its stable `sXX` record directly and rebuild the relevant week. The renderer and speaker notes are included in the project, so switching models or accounts does not require recreating the decks.
