<!-- Adapted from github.com/mattpocock/skills (MIT). Sourced from skills/deprecated/ubiquitous-language. -->
---
name: ubiquitous-language
description: Extract a DDD-style ubiquitous language glossary from the current conversation, flagging ambiguities and proposing canonical terms. Saves to UBIQUITOUS_LANGUAGE.md. Use when user wants to define domain terms, build a glossary, harden terminology, or mentions "domain model".
disable-model-invocation: true
---

# /ubiquitous-language -- Glossary Extraction

Extract and formalise domain terminology from the current conversation into a consistent glossary, saved to `UBIQUITOUS_LANGUAGE.md` in `kl_roman_pipe`'s root.

For scientific projects, the "domain" includes physical quantities, instruments, datasets, models, observables, and methods -- not just business nouns. Be opinionated about units and dimensions; a glossary that records ambiguous units is a future bug.

## When to Use

Invoke after a substantial design conversation, hypothesis discussion, or onboarding session where domain terms have surfaced. Re-invoke later in the same conversation to incorporate new terms or sharpen existing ones. Skip for short, single-purpose tasks where no new domain vocabulary appeared.

## Process

1. **Scan the conversation** for domain-relevant nouns, verbs, and concepts.
2. **Identify problems**:
   - Same word used for different concepts (ambiguity)
   - Different words used for the same concept (synonyms)
   - Vague or overloaded terms
   - Quantities used without explicit units
3. **Propose a canonical glossary** with opinionated term choices.
4. **Write to `UBIQUITOUS_LANGUAGE.md`** in the working directory using the format below.
5. **Output a summary** inline in the conversation.

## Output Format

```md
# Ubiquitous Language

## Hypothesis lifecycle

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Hypothesis** | A falsifiable claim about a system, registered before any code is written to test it | Theory, conjecture, idea |
| **Experiment** | A code + data run that produces evidence for or against one Hypothesis | Trial, simulation, test (overloaded with software tests) |
| **Result** | The output of an Experiment, archived with the exact code/data/config that produced it | Output, finding |

## Data

| Term | Definition | Units / shape |
| --- | --- | --- |
| **Lightcurve** | Time series of flux measurements for a single source | (N_epoch,) flux in microJy; epochs in MJD |
| **Measurement** | A single (time, value, uncertainty) tuple | scalar |
| **Dataset** | A versioned collection of Measurements with a fixed selection function | -- |

## Relationships

- A **Hypothesis** produces one or more **Experiments**.
- An **Experiment** consumes one or more **Datasets** and produces exactly one **Result**.
- A **Lightcurve** is a sequence of **Measurements** for one source.

## Example dialogue

> **Researcher:** "Does the new detrending change our recovery of `Hypothesis-A`?"
>
> **Agent:** "I can run an `Experiment` against `Dataset-v3`. The previous `Result` used `Dataset-v2` -- want me to re-run both for a like-for-like comparison?"
>
> **Researcher:** "Yes. And include the per-epoch uncertainty propagation -- the v2 `Result` dropped it."
>
> **Agent:** "Confirmed: the v2 `Experiment` reported point estimates only. The new `Experiment` will propagate `Measurement` uncertainties through the detrending."

## Flagged ambiguities

- "test" was used to mean both **Experiment** and a software unit test -- in this glossary, **Experiment** is the science term and "test" is reserved for code-level tests.
- "flux" was used without units in three places -- canonicalised to microJy throughout.
```

## Rules

- **Be opinionated.** When multiple words exist for the same concept, pick the best one and list the others as aliases to avoid.
- **Flag conflicts explicitly.** If a term is used ambiguously, call it out in "Flagged ambiguities" with a clear recommendation.
- **Pin units and dimensions.** For any physical quantity, record canonical units. Conflicting units in the conversation = flagged ambiguity.
- **Only include terms relevant for domain experts.** Skip the names of modules or classes unless they have meaning in the science.
- **Keep definitions tight.** One sentence max. Define what it IS, not what it does.
- **Show relationships.** Use bold term names and express cardinality where obvious.
- **Skip generic programming concepts** (array, function, endpoint) unless they have project-specific meaning.
- **Group terms into multiple tables** when natural clusters emerge (e.g. by lifecycle, by data layer, by actor). One table is fine for small glossaries.
- **Write an example dialogue.** A short conversation (3-5 exchanges) that demonstrates how the terms interact naturally and clarifies boundaries between related concepts.

## Re-running

When invoked again in the same conversation:

1. Read the existing `UBIQUITOUS_LANGUAGE.md`.
2. Incorporate any new terms from subsequent discussion.
3. Update definitions if understanding has evolved.
4. Re-flag any new ambiguities.
5. Rewrite the example dialogue to incorporate new terms.
