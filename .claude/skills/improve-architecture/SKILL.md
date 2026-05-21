<!-- Adapted from github.com/mattpocock/skills (MIT). -->
---
name: improve-architecture
description: Find deepening opportunities in a codebase. Surfaces architectural friction and proposes refactors that turn shallow modules into deep ones, improving testability and AI-navigability. User-invoked only.
disable-model-invocation: true
---

# /improve-architecture -- Codebase Architecture Review

Surface architectural friction in `kl_roman_pipe` and propose **deepening opportunities** -- refactors that turn shallow modules into deep ones. The aim is testability and AI-navigability.

## When to Use

Invoke explicitly when you want to step back from feature work and improve the architecture of a region of code: tightly-coupled modules, untested or hard-to-test seams, repeated bouncing between many small modules to understand one concept. Do *not* invoke during normal coding turns -- this is a deliberate, multi-phase grilling session, not a casual cleanup.

## A note for scientific code

Deep modules optimise for hiding implementation behind a small interface. That gospel (Ousterhout, _A Philosophy of Software Design_) is right for product code where callers should not care about internals. It is not always right for scientific code:

- **Visibility of the algorithm matters.** A reader (human or agent) must often see the integrator, sampler, loss function, or dimensional unit handling to verify correctness. Wrapping that behind a clean interface can obstruct review.
- **One-off analysis scripts are legitimately shallow.** Don't propose deepening a notebook-equivalent module just because the interface is wide.
- **The math IS the interface.** When the same physical quantity must flow through multiple call sites, the seam is in the type / units / array shape -- not in a Python class boundary. Sometimes the right answer is a stronger type, not a deeper module.

Apply the deletion test honestly. If deleting a module concentrates real complexity, deepen. If it just moves five lines of arithmetic into a class with a docstring, leave it alone.

## Language

Use these terms exactly in every suggestion. Consistent language is the point -- don't drift into "component," "service," "API," or "boundary."

- **Module** -- anything with an interface and an implementation (function, class, package, slice). Scale-agnostic.
- **Interface** -- everything a caller must know to use the module correctly: type signature, invariants, ordering constraints, error modes, required configuration, performance characteristics. Not just the type signature.
- **Implementation** -- what's inside a module. Distinct from **Adapter**: a thing can be a small adapter with a large implementation (a Postgres repo) or a large adapter with a small implementation (an in-memory fake).
- **Depth** -- leverage at the interface. **Deep** = large behaviour behind a small interface. **Shallow** = interface nearly as complex as the implementation.
- **Seam** _(Feathers)_ -- a place where behaviour can be altered without editing in that place. The *location* at which an interface lives. (Use this, not "boundary.")
- **Adapter** -- a concrete thing satisfying an interface at a seam. Describes *role*, not substance.
- **Leverage** -- what callers get from depth. More capability per unit of interface learned.
- **Locality** -- what maintainers get from depth. Change, bugs, knowledge concentrated at one place.

### Principles

- **Depth is a property of the interface, not the implementation.** A deep module can be internally composed of small, swappable parts -- they just aren't part of the interface.
- **The deletion test.** Imagine deleting the module. If complexity vanishes, it was a pass-through. If complexity reappears across N callers, it was earning its keep.
- **The interface is the test surface.** Callers and tests cross the same seam. If you want to test *past* the interface, the module is probably the wrong shape.
- **One adapter = hypothetical seam. Two adapters = real seam.** Don't introduce a seam unless something actually varies across it.

### Rejected framings

- **Depth as ratio of implementation-lines to interface-lines.** Rewards padding the implementation. Use depth-as-leverage instead.
- **"Interface" as just the type signature.** Too narrow.
- **"Boundary."** Overloaded with DDD's bounded context. Say **seam** or **interface**.

## Process

### 1. Explore

Read `kl_roman_pipe`'s `UBIQUITOUS_LANGUAGE.md` (if present) for domain vocabulary, and skim recent files under `docs/sessions/` and `docs/plans/` for decisions in the area you're touching.

Then walk the codebase. Don't follow rigid heuristics -- explore organically and note where you experience friction:

- Where does understanding one concept require bouncing between many small modules?
- Where are modules **shallow** -- interface nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called (no **locality**)?
- Where do tightly-coupled modules leak across their seams?
- Which parts of the codebase are untested, or hard to test through their current interface?

Apply the **deletion test** to anything you suspect is shallow: would deleting it concentrate complexity, or just move it? A "yes, concentrates" is the signal you want.

### 2. Present candidates

Present a numbered list of deepening opportunities. For each candidate:

- **Files** -- which files/modules are involved
- **Problem** -- why the current architecture is causing friction
- **Solution** -- plain English description of what would change
- **Benefits** -- explained in terms of locality and leverage, and how tests would improve

**Use UBIQUITOUS_LANGUAGE.md vocabulary for the domain, and the Language section above for the architecture.** If `UBIQUITOUS_LANGUAGE.md` defines "Lightcurve," talk about "the Lightcurve intake module" -- not "the FooBarHandler," and not "the Lightcurve service."

If a candidate contradicts a decision recorded in a session file or plan, surface it explicitly: _"contradicts the decision in `docs/sessions/2026-03-12_*.md` -- but worth reopening because..."_. Don't list every theoretical refactor a prior decision forbids.

Do NOT propose interfaces yet. Ask the user: "Which of these would you like to explore?"

### 3. Grilling loop

Once the user picks a candidate, drop into a grilling conversation (consider invoking `/grill-me`). Walk the design tree -- constraints, dependencies, the shape of the deepened module, what sits behind the seam, what tests survive.

Side effects happen inline as decisions crystallize:

- **Naming a deepened module after a concept not in `UBIQUITOUS_LANGUAGE.md`?** Add the term -- invoke `/ubiquitous-language` if the file does not yet exist.
- **Sharpening a fuzzy term during the conversation?** Update `UBIQUITOUS_LANGUAGE.md` right there.
- **User rejects the candidate with a load-bearing reason?** Capture it in the active session file (`docs/sessions/`) so future architecture reviews don't re-suggest the same thing. Only persist when the reason is non-obvious -- skip ephemeral ("not worth it right now") and self-evident ones.

## Output

When the grilling resolves, write the final proposal as a Parallax plan under `docs/plans/NNN_<short-name>.md` (next sequence number). Hand off to the user for implementation -- do not refactor in the same turn.

## Rules

- Never refactor in the same turn the proposal is written. Architecture decisions land as plans, not commits.
- Use the Language section's vocabulary exactly. No "component," "service," "API," or "boundary."
- Apply the deletion test honestly. Do not propose deepening for shallow modules whose complexity does not concentrate elsewhere.
- For numerical / scientific code, default toward visibility of the algorithm. Deepen only when the algorithm is stable and the seam is the bug surface, not when the code merely "looks shallow."
- Skip candidates that contradict prior decisions in `docs/plans/` or `docs/sessions/` unless the friction is real enough to warrant revisiting them -- and say so explicitly.
