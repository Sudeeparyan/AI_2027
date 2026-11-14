# Independent implementation audit — 13 October 2026

**Verdict: no, everything is not implemented perfectly.** The pack is substantially implemented and its current local verification evidence is sound, but four confirmed defects remain. Timed free Colab, live providers, most widget interactions and beginner/classroom usability remain unverified. Do not convert successful local gates or the earlier zero-findings closeout into a claim of complete perfection.

This audit answers the teacher's status question. It does not repair teaching sources. No deliverables, measured assets, credentials or historical completion manifests were changed; no build, render, full lab, model download, hosted request, commit or push was performed. Only audit diagnostics, this report and the task-board handoff were added. Existing uncommitted work was preserved.

## Confirmed open findings

| ID | Week / location | Problem | Severity | Required correction | Status |
|---|---|---|---|---|---|
| Q3-01 | 08; [lab source](../curriculum/labs/week_08_lab.py), lines 595, 613, 623, 777; distributed student cells 33, 35, 47 | TODO 3's failed comparison check only prints feedback. Its unfinished stub supplies validation frames and an empty final comparison; dependent error analysis continues on validation rows, and later reporting raises unrelated `KeyError('accuracy')`. | Medium | Enforce the failed check before any dependent analysis; give a TODO 3/rerun message. Add a paired built-student/built-solution regression. | Open |
| Q3-02 | 08; same lab, lines 676–681, 774; distributed student cells 41, 47 | TODO 4's unfinished zeros are printed as adapter/base sizes. `size_ok=False` is ignored, so zero sizes reach `run_results` if TODO 3 is completed. | Medium | Validate sizes through a named checker and stop before printing/exporting unsupported measurements. Add a paired regression. | Open |
| Q3-03 | 09; [slide source](../curriculum/weeks/week_09.yaml), lines 559–560; distributed slide 22 speaker notes | Notes say shuffled loss is above the uniform baseline, despite the actual values **correct 1.14 < shuffled 1.64 < uniform 1.79**. The Word explanation correctly says it remains below uniform, so formats disagree. | Medium | Explain the measured ordering without changing the measured numbers. Rebuild and verify the distributed speaker notes. | Open |
| Q3-04 | 09; same slide source, lines 563–566; distributed slide 22 speaker notes | The answer attributes a loss that is not enormous to softmax probabilities never reaching zero. Positive exact probabilities do not bound the loss; finite-precision softmax can also underflow to zero. | Medium | Explain this batch's actual scores and partial matches. Distinguish exact probabilities, numerical computation and stable cross-entropy. | Open |

Week 8 findings were reproduced using the **exact distributed cells**, invented tiny inputs and intercepted external boundaries. The unfinished student cells continued incorrectly; corresponding solution cells passed. Root reran the diagnostic and observed the same results. These fixtures are audit evidence, not course/model measurements; nothing was promoted into assets.

- [Week 8 diagnostic](../build/implementation_audit_20261013/reproduce_w08_student_guards.py) and [paired results](../build/implementation_audit_20261013/w08_student_guard_reproduction.json).
- [Week 9 numerical probe](../build/implementation_audit_20261013/probability_probe.json): float32 scores `[0, -1000]` produce softmax `[1, 0]` and stable cross-entropy `1000`. The false sentences were also confirmed in `ppt/notesSlides/notesSlide22.xml` inside the distributed PPT.
- Test scope explains the misses: ordinary smoke/full runs execute solutions; `test_student_guards.py` explicitly covers four Week 3 and five Week 9 TODOs, not every student task.

## Fresh checks and retained evidence

**29/29 fresh local command groups passed.** These included beginner/output structure; strict speaker notes; setup/provider/runtime/evidence regressions; paired student checks within their stated scope; diagram/figure/visual-support tests; all five text audits; figure lint/font checks; pilot-artifact checks; and current full-run executable-code parity. Coverage was **214/214 (100%)**. Exact commands, timestamps, exit codes and logs are in [fresh_checks.json](../build/implementation_audit_20261013/fresh_checks.json).

The new checks do not invalidate the defects above: the gates do not cover these semantic explanations or missed Week 8 student paths.

Read-only evidence validation established:

| Evidence checked in this audit | Result |
|---|---|
| Current manifest hashes | All 1,337 reviewed-image, 1,196 initial source/document and 259 deliverable hashes matched. The subsequent audit task-board update is an expected administrative change. |
| Office render inputs | All 24 weekly and five course retained inputs matched distributed files exactly. |
| Full/smoke executable parity | All twelve retained full and smoke notebooks matched current non-skipped executable core code; executed-file hashes matched their run records. These runs were not repeated in this audit. |
| Measured-result linkage | Latest CPU times matched the ledger; all twelve fill files regenerated identically in memory; ten promoted result assets matched their raw measured exports. Weeks 2/4 retain unchanged assets and timing records. |
| Recorded visual scope | Hashes/inventories cover 552 slides, 412 weekly Word pages, 103 diagram PNGs, 130 notebook images and 54 course pages. All twelve slide layout reports are `[]`. These are retained views, not a fresh exhaustive visual inspection. |
| Binding descriptor | Clean/tracked descriptor preservation checks returned no problems; XML outside Section 7.3 remains preserved, and in-memory clean regeneration matched the distributed descriptor. |
| Student notebook structure | All twelve have Markdown before every code cell and tagged slider cells; no `.py` lab files are distributed. Presence alone does not establish explanation quality. |

The latest retained local CPU core maximum is Week 8: **728.7 seconds (12.15 minutes)**. This is measured cached local execution, excluding installation/downloads, interactive cells, hosted APIs and optional stretches. It is not a new timing from this audit or evidence of free Colab timing.

Root additionally viewed six current contact sheets: Week 2 sheet 2; Week 8 sheets 2 and 6; Week 9 sheet 4; Week 10 sheet 5; Week 12 sheet 4 (36 slides total). No new clipping or overlap was identified in this sample. Week 2's categorical title about random codes decoding to nonsense is a low-priority teaching judgment: some displayed samples are recognisable, while its own notes appropriately qualify the claim. It is not counted among the four confirmed defects.

## Requirements still awaiting verification

| Requirement | Current limit |
|---|---|
| Free Google Colab CPU execution within budget | No connected/timed runtime; clean local installation and cached local timings do not prove Colab behaviour. |
| Live Gemini/OpenAI | Stubbed clients and SDK construction verified; live endpoints, account eligibility and actual quota/billing behaviour not exercised. |
| Interactive widgets | Earlier exact-cell Week 1 VS Code interaction is recorded. Other week interactions and Colab frontends remain unverified. |
| Beginner can follow every week alone | No beginner/classroom trial. Solution execution and structural tests cannot establish this requirement. |
| Reading access | Retained check: 206/209 reachable, three blocked/challenged pages. Links were not fetched again in this audit. |
| Optional GPU/media/stretch work and public deployment | Outside verified local core scope; retain these as explicit limits rather than core defects. |
| Exhaustive current visual reading | Current hashes preserve earlier actual views. This audit inspected a sample and source semantics, not every slide/page anew. |
| Every student TODO under incomplete/incorrect input | Dedicated paired checks cover selected cases. The two Week 8 reproductions demonstrate residual gaps. |

No completion percentage is assigned: passing automation, source implementation, actual execution, visual review and learner usability are different kinds of evidence.

Independent reviews: [labs](../build/implementation_audit_20261013/labs_review.md), [teaching](../build/implementation_audit_20261013/teaching_review.md), [evidence](../build/implementation_audit_20261013/evidence_review.md). Earlier completion/retry reports remain historical records. The **four open findings in this audit supersede any interpretation of their zero-findings counts as a current guarantee**.
