@AGENTS.md

## Claude Code only
- Use subagents only for read-only research or audits, at most 3 at a time. Ask each for ≤ 300 words
  with file:line references or URLs. They draw on the same usage limit as this session.
- Run renders and `test_labs.py --full` in the background, and meanwhile work on something that does
  not touch the same week.
- In a long task, run /compact between sub-steps. `research/TASKS.md` carries memory between sessions.
