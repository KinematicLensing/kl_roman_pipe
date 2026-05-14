---
name: audit
description: Structured audit of kl_roman_pipe. Run periodically or before major milestones.
disable-model-invocation: true
---

# /audit -- Project Audit

> Structured audit of kl_roman_pipe.

## When to Use

Invoke `/audit` periodically or before major milestones to check project health.

## Checks

Perform each check and report findings:

1. **Code review:** Bugs, concerns, anti-patterns, security issues.
2. **Test gaps:** Untested features, weak assertions, missing edge cases.
3. **Doc staleness:** Do docs match current code? Are references valid?
4. **Dependency health:** Outdated or vulnerable dependencies.
5. **Architecture:** Does code match intended design? Any drift?
6. **Scientific rigor:** Are hypotheses documented? Results reproducible? Assumptions stated?

## Output Format

```markdown
## Audit: [date]

### Critical
- [findings that need immediate attention]

### Warning
- [findings that should be addressed soon]

### Info
- [observations and suggestions]

### Summary
[1-2 sentence overall assessment]
```

## Rules

- Check @CONSTITUTION.md principles against actual practice.
- Flag any tests that appear to have relaxed tolerances.
- Verify reproducibility claims (are environments locked? versions pinned?).
- Be specific: include file paths, line numbers, concrete evidence.
