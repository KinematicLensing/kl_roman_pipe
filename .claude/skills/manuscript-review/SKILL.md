---
name: manuscript-review
description: Peer review critique of LaTeX papers or Beamer presentations. Read-only.
disable-model-invocation: true
---

# /manuscript-review -- Scientific Writing Review

> Read-only peer review critique. Auto-detects paper vs presentation mode.

## When to Use

Invoke `/manuscript-review` to get a structured critique of a LaTeX manuscript or Beamer presentation. Does not edit files -- produces a review document only.

## Mode Detection

- If `\begin{frame}` appears in any `.tex` file: **presentation mode** (Beamer)
- Otherwise: **paper mode** (article/report)

Search the project for `.tex` files to determine mode before reviewing.

## Review Protocol

### Shared Criteria (both modes)

1. **Argument structure**: Is the narrative logical? Do conclusions follow from presented evidence?
2. **Citation grounding**: Are all claims backed by cited results or data? Flag unsupported assertions.
3. **Figure/table quality**: Are figures self-explanatory? Properly referenced in text? Captioned?
4. **Citation completeness**: Do all `\cite{}` keys resolve? Missing references? Orphan bib entries?
5. **LaTeX hygiene**: Undefined references, compilation warnings, package conflicts.
6. **Redundancy**: Repeated arguments, duplicated content, unnecessary sections.

### Paper Mode (additional)

7. **Section ordering**: Does the structure follow field conventions (intro, methods, results, discussion)?
8. **Coverage gaps**: Missing related work? Incomplete methodology? Unaddressed limitations?
9. **Abstract accuracy**: Does the abstract reflect actual content and results?

### Presentation Mode (additional)

7. **Slide density**: Too much text per frame? One idea per frame?
8. **Visual hierarchy**: Do slides guide the eye? Is emphasis used effectively?
9. **Narrative arc**: Does the presentation tell a coherent story from motivation to conclusion?
10. **Timing estimate**: Approximate total time at ~2 min/frame. Flag if over/under target.

## Output Format

```markdown
## Manuscript Review: [filename]
### Mode: [Paper | Presentation]

### Critical
1. [SEV:critical] [location] Issue description. Fix: recommendation.

### Major
2. [SEV:major] [location] Issue description. Fix: recommendation.

### Minor
3. [SEV:minor] [location] Issue description. Fix: recommendation.

### Suggestions
4. [SEV:suggestion] [location] Suggested improvement.

### Summary
[1-3 sentence overall assessment. Strengths and primary concern.]
```

## Rules

- Read-only: do NOT edit any files. Produce the review as text output only.
- Number all issues sequentially. Include severity, location (file:line or section), and a concrete fix recommendation.
- Compile the document (`pdflatex` or `latexmk`) to catch warnings/errors. Report compilation issues as critical.
- Reference @CONSTITUTION.md principles where relevant (reproducibility, transparency, correctness).
- Be constructively critical. A thorough review now prevents embarrassment later.
