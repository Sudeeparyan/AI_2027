# Prompts for Claude Code and Codex

Both tools load the project rules automatically: Codex reads `AGENTS.md`, and Claude Code reads
`CLAUDE.md`, which imports it. The task board is `research/TASKS.md`. Do one task per session, then
start a fresh session; it costs far fewer tokens than one long session.
The full plan, readable from top to bottom, is `PROMPT_END_TO_END.md`. Review that one; the files
above are its compact form.

## 1. Next task (use this most of the time)
```
Read research/TASKS.md. Take the first todo task whose needs are done and that nobody is doing,
mark it doing, and complete it to its "Done when" list. Update its handoff line after each sub-step.
When finished, reply in at most 6 lines: what changed (files), checks run with results, what is not
verified, and what I should look at.
```

## 2. A named task (for parallel sessions)
```
Read research/TASKS.md and do task <ID> only (e.g. R2, E1, W03-L). Same rules and reply as prompt 1.
```

## 3. Resume after hitting a usage limit
```
Read research/TASKS.md. Task <ID> was cut off. Check its handoff line and `git status` for what is
already done, continue from there without redoing finished steps, and finish it as in prompt 1.
```

## 4. Pilot review (after W01 L, B and V are done)
```
Do not change any files. For Week 1, list the before/after slide PNGs that changed
(build/render_before_iter3/week_01 vs build/render/week_01), every new or changed diagram, the
findings table from research/weeks/w01_log.md and the lab's measured CPU time. Then run
tools/build_revision_review.py and give me the gallery path.
```

## 5. Turn my feedback into tasks
```
Do task I0: turn each point in research/TEACHER_FEEDBACK.md into task rows in research/TASKS.md.
Do not do the work itself.
```

## Running sessions in parallel
| Round | Open at the same time | Then |
|---|---|---|
| 1 | B0, R0, R1, R2, R3, R4 (in Claude Code, one session can run R1–R4 with subagents) | |
| 2 | E1 → E2 → E3, one after another (all three edit `tools/`) | |
| 3 | W01-L and W01-B, then W01-V | Pilot review (prompt 4); edit D1–D5 if needed |
| 4 | Up to 3 weeks at once: Wnn-L and Wnn-B ∥, then Wnn-V | Repeat until W12 |
| 5 | M1 → M2 → M3 | Read REVISION_2026-10.md "Iteration 3" |

Only one render or full lab run happens at a time on this machine; tasks wait on `build/.busy`.

To save your limits:
- Use the strongest model for E, L and V tasks and a faster, cheaper one for R and M tasks.
- Start a new session per task rather than continuing a long one.
- Write feedback in `research/TEACHER_FEEDBACK.md` and run prompt 5, rather than explaining it in chat.
- For another improvement round later: add your feedback, run prompt 5, then prompts 1–2 again.
  `AGENTS.md` stays the same between rounds.
