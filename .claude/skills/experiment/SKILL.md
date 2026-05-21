---
name: experiment
description: Scaffold a new experiment for kl_roman_pipe. Creates a structured manifest linked to a hypothesis.
memory: project
---

# /experiment -- Create Experiment Manifest

> Scaffold a new experiment for kl_roman_pipe.

## When to Use

Invoke `/experiment` when starting a new experiment or computational investigation.

## Protocol

1. Ask the human for the experiment name and linked hypothesis.
2. Fill out the manifest template below.
3. Create the manifest file in the appropriate location.

## Manifest Template

```markdown
## Experiment: [name]

**Hypothesis:** [reference to hypothesis being tested]
**Description:** [what will be done]
**Expected outcome:** [prediction based on hypothesis]

**Test plan:**
1. [step-by-step procedure]

**Environment:**
- Code version: [git ref]
- Config: [relevant settings]
- Data: [input data description/location]

**Status:** proposed | active | concluded

**Result:** [filled after conclusion]
- Outcome: supported | refuted | inconclusive
- Evidence: [data, plots, metrics]
- Notes: [observations, surprises, caveats]
```

## Rules

- Every experiment must link to a hypothesis. No undirected exploration without a stated expectation.
- Record environment details sufficient for exact reproduction.
- Document the result regardless of outcome.
- If the experiment reveals something unexpected, create a new hypothesis.
