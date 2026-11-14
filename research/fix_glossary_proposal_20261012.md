# A08/A10 repair handoff — 12 October 2026

- Canonical glossary: 142 → 253 entries; 111 missing advanced concepts merged. Existing definitions retained; 146 original Word rows across nine weeks mapped, with three reference-only sections also inventoried.
- Machine-readable migration: `fix_glossary_proposal_20261012.json` contains exact canonical `term_keys` for every original row and each full week. Combined concepts have separate keys; synonyms reuse existing entries. The canonical file now also contains `week_glossary`, mapping `01`–`12` to ordered keys. All 253 terms appear in at least one weekly list; every key agrees with the term's `weeks` membership.
- Generator proposal: replace the body under each `# Glossary` heading with a source placeholder, then have `tools/beginner.py` generate its table from `week_glossary` and canonical definitions. Keep Quick glossary and slide vocabulary compact. Fail on unknown keys and unexpanded placeholders; test that changing one canonical definition changes both Word glossary locations. Root owns this engine/source migration.
- Definition corrections: RLHF does not require PPO universally; PPO is an RL algorithm used in some RLHF pipelines; DPO trains directly from preferences without a separate reward-model stage/online RL loop. GRPO and RLVR are separate concepts. QLoRA trains adapters through a frozen quantised base. Compute-optimal training and small-model sizes have no universal numeric threshold. CLIP-score scaling remains implementation-dependent in the existing canonical definition. C2PA verifies recorded provenance/integrity rather than the truth of depicted events.

## Primary sources checked on 12 October 2026

| Concept | Source |
|---|---|
| RLHF | [InstructGPT paper](https://arxiv.org/abs/2203.02155) |
| PPO | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |
| DPO | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) |
| QLoRA | [QLoRA paper](https://arxiv.org/abs/2305.14314) |
| GRPO | [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) |
| Flow matching / rectified flow | [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003) |
| Compute-optimal training | [Chinchilla paper](https://arxiv.org/abs/2203.15556) |
| POPE / CHAIR | [Object-hallucination evaluation](https://arxiv.org/abs/2305.10355), [Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156) |
| C2PA / SynthID | [C2PA explainer](https://spec.c2pa.org/specifications/specifications/2.3/explainer/Explainer.html), [Google DeepMind SynthID](https://deepmind.google/models/synthid/) |

## Runtime provenance disposition

| Week / quoted archive measurement | Inspected evidence | Action |
|---|---|---|
| 04 / U-Net training, 8.1 min | `curriculum/assets/week_04/metrics.json`, `build/assets_week04.log`, asset script and central ledger | No genuine matching dated run metadata found. Removed quote from Word and source slide 18; retained epochs, hardware, parameter count and samples. |
| 04 / SD 1.5 and SD-Turbo endpoint image times | Same metrics, `build/assets_week04b.log`, asset script and ledger | No genuine matching dated run metadata found. Removed numeric timing quotes from Word and source slides 26/28/33 before order swaps. Stored grids and settings retained. |
| 05 / character model training, 8.7 min | `curriculum/assets/week_05/metrics.json`, `build/assets_week05.log`, `build/week_05/full_measurements_2026_10.json`, dated `results.json` and ledger | Dated detailed runs have different durations/losses. Removed unsupported archive-runtime row only. Preserved the distinct dated GPU stretch and current CPU-core quotes. |

No date was inferred from file modification time or partial log timestamps. No new ledger record was proposed. Current core/runtime placeholders remain intact. Stored metrics files are unchanged.

## W04 teaching-order extension

- Whole dictionaries swapped for source slides 10/11 and 27/28 using `edit_slides`; all 40 original slides remain. Companion sections/references and plan text aligned; title-based diagram anchors unchanged.
- Existing numerical examples in Word now precede the associated forward/reverse, noise-loss, DDIM, guidance and flow-matching equations. No result values were added.
- Root must rebuild and review changed slides/Word pages after engine migration. No notebook execution, build or render was performed by this task.
- Source-only validation observed PASS: 253 unique canonical terms, 111 additions, 146 original-row mappings, 12 complete weekly lists, all 40 W04 strict speaker-note templates, requested slide order and removal of unsupported archive runtime placeholders. Ownership released to root for canonical marker migration and complete generated-output validation.


## Fresh evidence handoff — 12 October 2026
- Validated all 11 successful clean-pinned CPU full records against executed notebook hashes and absence of error outputs. Promoted nine staged exports (W01/03/05/06/07/09/10/11/12); W02/W04 provide runtime records only.
- Appended 11 actual full-run records to the existing iteration_3_full_runs ledger, preserving all historical entries. Source exports retain genuine exporter timestamps and add exact record/evidence hashes. W08's failed batch record was not promoted; its corrected successful rerun is pending.
- 59 archived experiment files remained byte-identical. Figure-harvest dry runs validated all expected W09/W11/W12 full figures; W10 core lacks optional image-generation/editing plots and the harvester correctly refused to mutate any assets. No archived figures harvested or overwritten.
- Manifest: build/audit_fixes_20261012/result_promotion_manifest.json. Root must fill/build only after W08 ownership is released, then inspect the changed result text and timing charts.

## Final dated CPU result promotion — 12 October 2026
- W08 portable-loss retry succeeded: started 2026-10-12T22:26:04.169647+00:00; 728.7s full local CPU core (12.15min). Its own genuine record/export was validated and promoted after lab ownership release. Original failed full batch row was not replaced.
- Final manifest: 12 successful current CPU records in the ledger, 10 authentic per-week CPU JSON exports (W02/W04 do not export results JSON). All executed-notebook hashes verified and zero error outputs observed. Archived experiment assets remained unchanged.
- Canonical glossary remains 253 terms with 12 weekly maps. Q1-M source/evidence work is complete; coordinator owns final fill/build, notebook smoke gates and rendered review. No timed Colab, live API or browser-widget claim.
