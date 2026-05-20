---
name: handoff
description: Generate a structured handoff summary for kl_roman_pipe.
disable-model-invocation: true
---

# /handoff -- Agent Handoff Summary

> Generate a structured handoff summary for kl_roman_pipe.

## When to Use

Invoke `/handoff` at the end of a work session, before context gets too long, or when passing work to another agent/human.

## Output Format

Generate the following sections:

```markdown
## Handoff: [date or session identifier]

**Problem statement:** [1-2 sentences]

**What was investigated/attempted:**
- [bulleted list]

**Key findings:**
- [bulleted list]

**Current state:**
- What works: [list]
- What doesn't: [list]

**Recommended next steps:**
1. [prioritized list]

**Open questions:**
- [list]

**Relevant files/paths:**
- [list with brief descriptions]
```

## Write to File

Write the completed handoff to `docs/sessions/YYYY-MM-DD_<topic>.md` per the filename rule in PARALLAX.md (`Agent Handoff Format > Filename and storage`). Use today's date as ISO prefix; pick a short distinct kebab- or snake-case topic. If the file already exists, prefer a more specific topic over `_2` suffixing.

Do not `git add` or commit the handoff -- these are agent working notes, not project source-of-truth.

## Self-Review Pass

After writing the initial draft, re-read it critically and check for:

1. **Unverified claims.** Did you assert something works without evidence? Flag it as untested.
2. **Missing context.** Could a fresh agent pick up from this alone? What would they waste time rediscovering?
3. **Gaps in "What doesn't work."** "Nothing known" is almost always wrong. Think about what wasn't tested, what was deferred, what assumptions went unchecked.
4. **Test intent shifts.** If tests were modified, did the *meaning* of the test change, not just the assertion values?
5. **Upstream continuity.** Is there prior session context (plans, earlier sessions) that this handoff assumes but doesn't reference?

Update the draft with findings, then present to the user.

## Rules

- Be concise but complete. Another agent should be able to continue without re-investigating.
- Include file paths for all code that was touched or is relevant.
- Flag any known issues or risks explicitly.
- If a hypothesis was being tested, include its current status.
- Always write the handoff to `docs/sessions/` per PARALLAX.md naming rule. Do not just display it in chat.
