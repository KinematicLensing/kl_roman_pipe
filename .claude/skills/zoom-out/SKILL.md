<!-- Adapted from github.com/mattpocock/skills (MIT). -->
---
name: zoom-out
description: Tell the agent to zoom out and give broader context or a higher-level perspective. Use when you're unfamiliar with a section of code or need to understand how it fits into the bigger picture.
disable-model-invocation: true
---

# /zoom-out -- Higher-Level Context

User invokes this skill when they don't know an area of code well and need broader context before making changes.

## When to Use

Invoke `/zoom-out` when looking at unfamiliar code, before proposing changes in an area whose call graph you have not yet mapped, or when a request touches a module whose role in the larger system is unclear.

## Protocol

Go up a layer of abstraction from the file or function in front of you. Use vocabulary from `kl_roman_pipe`'s `UBIQUITOUS_LANGUAGE.md` if it exists; otherwise name modules by their actual filenames and surface concepts as you see them.

Report:

- The role this code plays in the larger system (one sentence).
- Direct callers and direct dependencies (modules, functions, files).
- One layer up: what module/subsystem owns this, and what owns *that*.
- Any data or control flow that touches this code from outside the immediate area.
- Surprises: places the code is invoked from that don't follow the obvious path.
