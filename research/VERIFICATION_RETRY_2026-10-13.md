# Verification retry — completed 13 October 2026

The retry found additional student-facing problems that solution-only execution had missed. Nineteen TODO checks now stop unfinished code before it becomes a result, plot, training run or optional demonstration. The error identifies the TODO and tells the student to complete it and rerun. The actual model algorithms and MSc content are retained.

| Finding | Small correction | Local status |
|---|---|---|
| Ignored failed TODO checks in Weeks 3, 5 and 9–11 | Existing checks stop dependent work; Week 9 reporting follows the successful check. | Fixed; regression and full CPU evidence retained |
| Week 4 optional image-training dependency | Install torchvision with its existing verified pin. | Fixed; installation coverage test passed |
| Ten floating direct installation packages | Freeze the already-tested versions in the shared pin table; 24 direct pins now match the local environment. | Offline resolver and pip check passed |
| Week 12 archived score sounded like current correctness | Identify the archive and reference-pattern matching; require answer and cited-passage review. | Rebuilt and visually reviewed |

All 50 final command groups passed, including all twelve CPU smoke runs, coverage at 100%, nine paired built-student/built-solution regression cases, shared setup checks, callbacks and SDK contracts. All five changed labs received successful new full CPU runs. The remaining seven full records retain exact current executable-code parity. The longest current core remains Week 8 at 728.7 seconds (12.15 minutes). Timings use the tested local CPU environment and cached models/data; installation, downloads, widgets and hosted routes are excluded.

| Changed lab | New measured full CPU core |
|---|---:|
| 03 | 267.1 seconds |
| 05 | 283.1 seconds |
| 09 | 154.3 seconds |
| 10 | 169.3 seconds |
| 11 | 38.2 seconds |

Visual scope: current replacements in Weeks 3, 5, 9–12 and course documents were reviewed. Unchanged images retain exact prior-view hashes. All current Office inputs, notebook-image inventories and inspected hashes passed the closeout guard; no open visual findings.

The UI retry also corrected the advertised Python minimum to 3.12, matching the pinned NumPy/SciPy requirements. Each notebook now explains its numeric fallback when a slider cannot display. These are Markdown-only clarifications; executable core parity and the 29 setup tests passed after rebuilding all twelve notebooks.

UI follow-up: the exact current Week 1 widget cells ran in a VS Code scratch notebook, using five invented names. The slider rendered, and changing temperature from 1.0 to 1.7 and then 0.4 updated the chart title and distribution. This verifies Week 1 interaction in the observed local environment; other week frontends remain unverified. The earlier third-party-download prompt no longer appeared on rerun. The agent changed no permission or security setting. See [the observed pilot record](../build/retry_20261013/widget_ui_verified.json).

Limits: timed Colab, live Gemini/OpenAI, other week widget interactions, GPU/stretch/public deployment and classroom effectiveness remain unverified. Week 1 solutions were uploaded and opened in Colab, but its browser window is still minimized and recovery reports "user input was detected in this window; call get_window_state before continuing". A fresh selection and one recovery retry failed again; no Colab runtime was connected or timed. Neither provider key was present in the checked process environment and no local key file was selected. No credentials were read, no Gemini/OpenAI model requests were sent, and this agent made no commit or push.

Evidence: [current correction manifest](AUDIT_FIXES_2026-10-12.json), [fresh retry checks and CPU records](../build/retry_20261013/), [student readiness findings](../build/retry_20261013/student_readiness.md), and [dated dependency verification](../build/retry_20261013/dependency_pins.md). The prior completion manifest is preserved in the retry folder.

Useful review points: the student TODO hints and stopping messages, and Week 12’s archived-results explanation.
