---
name: grill-me
description: Stress-test a plan or design through relentless questioning until reaching shared understanding.
disable-model-invocation: true
---

# /grill-me -- Design Interrogation

> Systematic interrogation of a plan or design. Walks each branch of the decision tree, resolving dependencies between decisions one-by-one.

## When to Use

Invoke `/grill-me` before committing to a plan or design. Use when you want to stress-test assumptions, surface hidden dependencies, or reach shared understanding on a complex decision.

## Protocol

1. Read the plan or design under discussion (file, message, or current plan mode context).
2. Walk down each branch of the design tree. For each decision point:
   - Ask precisely what was decided and why.
   - Identify dependencies on other decisions.
   - Resolve dependencies before moving deeper.
3. If a question can be answered by exploring the codebase, explore the codebase instead of asking.
4. Continue until every branch is resolved or explicitly deferred.

## Rules

- Be relentless. Do not accept vague answers -- ask follow-ups until the answer is concrete.
- One decision at a time. Do not bundle questions.
- Track which branches are resolved vs open. Summarize status periodically.
- If a branch is intentionally deferred, record it as an open question.
