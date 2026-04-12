---
name: hypothesis
description: Guides hypothesis-driven investigation for kl_roman_pipe. Invoke before starting any investigation, feature, or experiment.
memory: project
---

# /hypothesis -- Hypothesis Workflow

> Guides hypothesis-driven investigation for kl_roman_pipe.

## When to Use

Invoke `/hypothesis` before starting any new investigation, feature, or experiment.

## Protocol

1. **State your hypothesis.** What do you expect, and why? Be specific and falsifiable.
2. **AI generates alternatives.** Claude proposes 2-3 alternative hypotheses for consideration.
3. **Human selects.** Pick the hypothesis to test (may be the original or an alternative).
4. **Define success/failure criteria.** What measurable outcomes support or refute the hypothesis?
5. **Design and run the test.** Implement the experiment, collect results.
6. **Record the conclusion.** Supported, refuted, or inconclusive -- with evidence. Document regardless of outcome.

## Output Format

```markdown
## Hypothesis: [descriptive name]

**Statement:** [specific, falsifiable claim]
**Rationale:** [why you expect this]
**Success criteria:** [measurable outcome if supported]
**Failure criteria:** [measurable outcome if refuted]
**Status:** proposed | testing | supported | refuted | inconclusive
**Evidence:** [results, data, references]
```

## Rules

- Never skip the hypothesis step. No implementation without a stated expectation.
- Negative results are valuable. Document them with equal rigor.
- If inconclusive, state what additional evidence would resolve it.
