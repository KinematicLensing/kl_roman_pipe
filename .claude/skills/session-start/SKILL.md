---
name: session-start
description: Bootstrap a new kl_roman_pipe session. Reviews recent handoffs, memory, and current project state.
disable-model-invocation: true
---

# /session-start -- Session Onboarding

> Bootstrap a new work session by reviewing previous handoffs, semantic memory, and current state.

## When to Use

Invoke `/session-start` at the beginning of a new session.

## Instructions

1. **Read recent handoffs.** Find and read the 3 most recent files in `docs/sessions/` (by filename sort). Most recent is primary; older ones provide trajectory. If <3 exist, read all.
2. **Check semantic memory.** MEMORY.md is auto-loaded but verify it matches codebase state. Flag contradictions.
3. **Audit current state.** Run tests and linting. Review recent `git log --oneline -20`. Check README.md for currency.
4. **Critically evaluate the handoffs.** Don't parrot their recommendations. Challenge priorities if the codebase state suggests different next steps. Note anything the handoffs missed or got wrong.
5. **Synthesize and present:**
   - What was done last session
   - Where we stand now (test/lint status, open branches, uncommitted work)
   - Recommended next steps (may differ from handoff's recommendations -- explain why)
   - Unresolved questions requiring human input
6. **Wait for human direction** before starting work.

## Rules

- Be critical. The handoff is a starting point, not a directive.
- If tests fail, that takes priority over any planned work.
- If docs are stale, flag it explicitly.
- If MEMORY.md contradicts current codebase state, flag it and propose corrections.
- Keep the synthesis concise -- bullet points, not paragraphs.
