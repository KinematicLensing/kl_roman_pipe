# kl_roman_pipe -- Scientific Workflow Rules

Scientific workflow protocols for kl_roman_pipe.
Domain: cosmology

## Hypothesis Protocol

Before implementing any feature or investigation:

1. **State the hypothesis.** What do you expect, and why?
2. **Define success criteria.** What measurable outcome supports the hypothesis?
3. **Define failure criteria.** What outcome refutes it?
4. **Implement the test.** Code the experiment.
5. **Record the result.** Supported, refuted, or inconclusive -- with evidence.
6. **Document regardless of outcome.** Negative results are valuable.

## Experiment Manifest

Each experiment should have:

- **Name**: descriptive identifier
- **Hypothesis**: reference to the hypothesis being tested
- **Description**: what will be done
- **Expected outcome**: prediction
- **Test plan**: how to verify
- **Environment**: code version, config, data state
- **Status**: proposed | active | concluded
- **Result**: supported | refuted | inconclusive (with evidence)

## Agent Handoff Format

When ending a session or handing off to another agent:

- Problem statement (1-2 sentences)
- What was investigated/attempted
- Key findings (bulleted)
- Current state (what works, what doesn't)
- Recommended next steps
- Open questions
- Relevant files/paths

## Agentic Workflow

- AI handles: mechanical, repetitive, and exploratory tasks
- Humans own: interpretation, direction, and final scientific judgment
- AI must surface uncertainty, never hide it
- All assumptions must be documented
- Speed gains must never come at the cost of rigor

## JAX / Differentiable Computing Rules

- Pure functions: no side effects inside jitted code
- Explicit RNG key threading (never use global state)
- Use pytree-compatible data structures
- Avoid Python control flow over traced values (use jax.lax.cond, jax.lax.scan, etc.)
- Shape/dtype annotations on all array-producing functions

