# Final end-to-end verification (requested 13 October, completed 14 October 2026)

Goal: every deck, speaker note, Word note, diagram and lab must be correct and easy for a beginner to follow. This pass fixed the four open Q3 findings. It then reread the text of all 12 decks (titles, bodies and speaker notes) and freshly viewed all 552 slides. Every finding below is fixed in `curriculum/` or `tools/`, rebuilt, re-rendered and viewed again. Helper scripts, logs and render records are in [build/final_verification_20261013/](../build/final_verification_20261013/).

## Findings and fixes

| ID | Week, place | Mistake | Fix |
|---|---|---|---|
| Q3-01 | 08 lab, TODO 3 | A failed comparison check only printed feedback; error analysis then ran on validation rows. | `assert check_comparison(...)` stops the cell with "Complete TODO 3, then rerun". Paired student/solution regression test added. |
| Q3-02 | 08 lab, TODO 4 | Unfinished zero sizes were printed and could reach the results export. | A named `check_sizes()` plus an assert runs before printing. Regression test added. |
| Q3-03 | 09 slide 22 notes | The notes said the shuffled loss was above ln 6; the run gives 1.14 < 1.64 < 1.79. | Notes rewritten: the shuffle moves 3 of 6 captions (seed 42 permutation checked), so the loss rises but stays below ln 6. |
| Q3-04 | 09 slide 22 notes | The answer said a loss is bounded because softmax never reaches zero. | The new answer explains partial matches and close scores, and the Watch out tells students to use `F.cross_entropy`. |
| F1 | 03 coverage diagram trace | Wrong sample count. | "512 generated digits (2,000 in the stretch)". |
| F2 | 10 slide notes | Called the measured local model "proposed". | "The CPU core uses SmolVLM-256M". |
| F3 | 12 three quiz slides | Ask/Answer did not restate the question or give the letter. | Ask restates the question; Answer gives letter + reason (D, A, C). |
| F4 | 01 slide 35 families figure | Crowded labels; row label touched the first box. | Re-spaced rows and columns. |
| F5 | 02 slide 10 title | "random codes decode to nonsense" overstated the images shown. | "random codes often decode poorly". |
| F6 | 03 figures | Legend "Flow" line and the conditional-GAN branch colours did not match the arrows. | Both drawn in the arrow grey. |
| F8 | 05 slide 31, 12 slide 32 | Code-slide captions were never rendered by the engine. | Captions moved to short side bullets. |
| F9 | 08 slide 18 RLHF | Reward model and PPO update had misleading role colours. | Reward model = model, PPO update = loss/update. |
| F10 | engine, 05 slide 43 and 08 slide 34 | Legend listed "Loss" and "Check" separately, but they are drawn identically. | Identically drawn roles merge into "Loss / Check". `test_technical_diagrams.js` updated. |
| F11 | 09 slide 33 | The architecture-families diagram did not meet the diagram standard. | Redrawn: left→right flow, each arrow labelled (vectors, tokens, features, visual tokens), role legend. |
| F12 | 10 slides 8, 20 | Arrow labels collided; the legend was dropped in the editable copy. | Straight labelled arrows; box-swatch legend. |
| F13 | 11 slide 31 | Block labels for the two checks were unclear. | Renamed "Input checks" and "Output check". |
| F14 | 12 slide 10 | The RAG pipeline did not show which steps run once and which run per question. | Redrawn as two labelled rows, "Indexing (once per source update)" and "Answering (per question)", joined by one "index" route. |
| F15 | 08 slides 19, 27, 34; 12 slide 20 | Over the 40-word body limit (`count_body_words` gate). | Trimmed. The LoRA symbol table now says α / r is the scale (α alone is not). The training diagram now says "held-out loss". |
| F16 | 09 slide 22 | The second bullet repeated "ln N" from the first. | "Shuffled: below ln N." |
| F17 | 09 Word notes, slide 19 section | Claimed the horse image scored the cat caption above its own caption. | False. The recorded heatmap and a recomputation that reproduces the recorded losses (1.138 and 1.640) show all six images pick their own caption, in both directions. Notes corrected; slide strip now says "Every true pair wins → lowest loss". |
| F18 | 09 Word notes, confusions | Listed airplane → ship among the largest confusions but left out frog → bird, which ties it (6 each in the stored matrix). | Both listed as tied after deer → horse, dog → cat and frog → cat. |
| F19 | 10 Word notes, image editing | Said strength 0.6 left the painting "almost unchanged". | The stored image keeps the lighthouse but redraws the shore; wording corrected. |

Also re-verified with no change needed: every worked calculation in the 12 decks (for example likelihood 6.6×10⁻⁵, KL 0.32/2.32, LoRA 2,162,688 = 0.44%, Wilson 65.2–89.5%, RRF 0.03178, RM loss −ln σ(2) ≈ 0.13, img2img effective steps 1/2/3), and every lab-slide runtime against `research/lab_runs_2026-10.json`.

## Checks run in this session (14 October)

| Check | Result |
|---|---|
| Rebuild of all 12 weeks + course pack | OK, 46 slides each; Weeks 1, 8, 9, 10, 12 and the course pack rebuilt again after late fixes |
| Full "Build and check" gate list (26 gates) | 26/26 PASS on the final run (links: 204 of 209 reachable, 5 sites block automated checks, 0 failures). Two earlier runs each lost `check_links` to transient github.com 503s. |
| `test_student_guards.py` (pinned env) | 11 tests OK, including the two new Week 8 tests |
| Smoke runs, built labs | 12/12 PASS; Weeks 1, 8, 9, 12 re-smoked after their rebuilds, PASS. Week 9 and 10 notebooks are byte-identical after the final notes rebuild. |
| Week 8 full CPU core (pinned env, offline, 4 threads) | PASS in 695.5 s (11.6 min) on 14 Oct; results export identical to the 12 Oct run (accuracy table, loss curves, adapter 4.4 MB); promoted to the ledger. Slides now say 11.6 min (training 5.0 min). |
| `compare_lab_code` | 12/12 weeks: same code as their full run |
| Renders (PowerPoint/Word) | All 12 decks and Word notes re-rendered from the final build; `slides.layout.json` reports 0 layout issues in every week. Every fixed slide and the changed Word pages were viewed after the final render. |

## Not verified or still open

- **F7 (limitation):** in Week 3's `losses.png`, the legend is crossed by the curves. It is an archived GPU result with no stored loss history, so it cannot be redrawn without a GPU rerun.
- **Links:** 5 sites block automated checks and need a manual click (unchanged). github.com file pages sometimes return HTTP 503 to the checker; the affected files exist (raw and API URLs return 200).
- **Not run:** Colab timing, live Gemini/OpenAI calls, widget interaction outside Week 1, GPU/stretch routes, classroom testing.
- **Word pages:** only the pages with changed text were viewed (for example Week 8 p18 and Week 9 p26). Notebook image sheets were regenerated but not all viewed.
- No commit or push was made.

Lesson for future reviews: claims in Word notes about what happened in a specific run (which pair won, which classes were confused, how an edit looked) must be checked against the stored export or recorded image, not the slide numbers alone. F17–F19 were all of this kind.
