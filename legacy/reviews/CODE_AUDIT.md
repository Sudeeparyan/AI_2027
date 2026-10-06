# Repository audit and repair evidence

Audit date: 28 September 2026. Scope: source code, canonical content flow, existing tests, distribution and reproducibility. No live provider credentials or paid API calls were used.

## Architecture and source of the curriculum mismatch

This repository is a teaching-material generator, not a single student-facing application. `studio.py` extracts supplied documents and optionally drafts/reviews JSON or notebooks through `src/providers.py`. Candidates are staged, hash checked and backed up before application. `course.py` reads weekly lesson, slide, study, visual and notebook sources; `src/engaging_slides.py`, `src/slides.py` and `src/handouts.py` create artifacts. Its historical build used LibreOffice to render PDF, exported diagrams, executed concatenated notebook cells and then copied successful weekly releases into `outputs/current`. `verify_release.py` checked artifact structure; it did not establish academic suitability. `package_project.py` creates the complete instructor archive.

`content/teaching_design/author_lessons.py` and `build_lessons.py` historically contained another source of truth: a hardcoded 12-week story and a compiler producing exactly 40 repeated-pattern slides. Running those scripts could restore the old curriculum after manual changes. `src/slides.py` also selected introductory diagrams by week number, rather than by lesson topic. These mechanisms must track the revised source or be retired; otherwise a successful build can be educationally wrong.

The former week sequence devoted several weeks to document retrieval, policy assistants, release checks and operational details. Core notebooks were explicitly labelled simulations, but those simulations did not demonstrate genuine training. This is a material alignment issue where the unchanged descriptor requires use of contemporary frameworks to train generative models. The authoritative source for this revision is the OLD descriptor in `context/`; earlier references to the NEW descriptor and research blueprint are not authority to alter sections outside 7.3.

## Concrete code defects repaired

- Provider responses with an empty `choices` array raised an uncaught `IndexError`; these now produce a controlled malformed-response error.
- Provider `parameters` could silently replace the configured model and all messages, invalidating both authoring instructions and recorded metadata. Reserved fields and unsupported streaming are rejected.
- Endpoints with fragments, missing hostnames or embedded credentials are rejected. Routes and retry counts are checked before calls.
- Slide validation compared unordered IDs even though rendering uses dictionary iteration order. Reordered slides now fail validation, as do empty diagrams and malformed table/mapping rows that previously reached rendering.
- Malformed notebook cell objects, cell sources, duplicate IDs and shared-code references moved into markdown now produce validation errors.
- A second model repair pass received an error without the failed candidate; it now receives the failed output and the request for a complete correction.
- Source ingestion now creates its extraction folder and accepts uppercase document extensions; a missing source directory fails clearly.
- Packaging now has an import-safe callable entry point, creates missing destination directories, excludes isolated environments such as `.venv-labs`, cache directories and nested ZIPs, and checks a temporary ZIP before replacing an existing archive. Machine-specific provider configuration is replaced with the clean example.

The package remains the teacher's complete project: it includes answer guides and instructor-only material. It is not a student distribution.

## Verification performed

The original studio/provider tests passed. Initial complete test discovery failed because the default system Python lacked `fitz` (PyMuPDF), `python-docx` and `python-pptx`; that failure is an environment issue and is not evidence that the release is valid. The existing runtime and the revised build environment are checked separately by the main rebuild.

After the targeted fixes, `python -m unittest discover -s tests -p test_studio.py -v` passed **18 tests**. These include local HTTP transport, exhausted retries/fallback, missing route, malformed response, reserved payload fields, slide order, empty diagrams, malformed notebooks, repair context, candidate/source hashes, backups, locking and archive exclusions. Tests use temporary directories and a localhost mock service. No cloud endpoint was called.

## Build/release concerns forwarded for the curriculum rebuild

- Historical fingerprinting watched shared code only for weeks 7, 11 and 12, and reused unchanged releases without checking whether output files still existed.
- A course-wide build publishes successful weeks individually; a later failure can leave a mixture of old and new weeks. Verify all twelve current weeks against the final canonical source.
- Concatenated Python cells prove a script can run, but do not validate full Jupyter metadata, kernel state, cell outputs or learner exercise solutions. The revised notebooks should be executed cell by cell from fresh state.
- Slide bounds and text-length checks do not detect all overflow, awkward truncation or misleading diagrams. Native rendering and visual review are necessary.
- Historical handouts hardcoded two hours of concepts plus two hours of practice and a reference to the NEW descriptor. Check both against the unchanged OLD template.
- Historical QA reports disagree about 28 versus 40 slides. New release evidence must describe actual delivered artifacts, with old reports clearly superseded.
- The historical verifier hardcoded 40 slides and assumed one raster diagram per slide. Those checks need to follow the final renderer's actual contract, rather than force the old design onto the revision.
- `instructor_only/run_final.py` was a small policy-assistant example, not proof that the revised broad course assessment is valid. It must not substitute for the descriptor's unchanged assessment scheme.

## Verification limits

Automatic tests cannot certify teaching quality, correctness of every cited claim or formal academic approval. A completed artifact build must be accompanied by descriptor-preservation checks, current source references, real training evidence where required, all-notebook execution, rendered visual review and an honest record of remaining limits.
