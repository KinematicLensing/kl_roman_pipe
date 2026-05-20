---
name: doc-sync
description: Audit all living documentation against the actual codebase and report inconsistencies.
disable-model-invocation: true
---

# /doc-sync -- Documentation Sync Audit

> Audit all markdown documentation against the actual codebase. Identify claims that have drifted from reality, then offer to fix them.

## When to Use

- Periodically, to catch doc drift before it accumulates
- Before releases or major milestones
- After large refactors that may have invalidated documentation
- When `/audit` flags doc staleness concerns

## Step 1: Discover Documents

Collect all markdown files (`.md`) in the repository, excluding:

- `.git/`
- `node_modules/`
- `__pycache__/`
- `.pixi/`
- `.venv/`, `venv/`, `.env/`
- `build/`, `dist/`, `*.egg-info/`
- `.parallax/`
- Any directory named `vendor` or `third_party`

Present the list to the user and ask:

> "I found N markdown files to audit. Are there additional directories or files you want me to exclude?"

Wait for the user's response. Apply any additional exclusions they specify.

## Step 2: Plan Subagent Batches

Divide the documents into up to 5 batches for parallel analysis. Batching strategy:

- **Large documents** (>200 lines or >50 claims): one document per batch
- **Small/related documents** (same directory or <50 lines): group into a single batch
- **Always separate**: README.md, CLAUDE.md, and any file with "architecture" or "design" in its name get their own batch

Present the batching plan to the user but do not wait for approval -- proceed immediately.

## Step 3: Per-Batch Analysis

For each batch, launch a subagent. Systematically check every claim in every document against the seven categories below. Read the actual codebase to verify each claim. Do not guess or rely on memory.

### Category A: File and Directory Paths

Scan for any path-like references: backtick-quoted paths, paths in prose, links to local files.

For each path found:
- Does the referenced file or directory exist?
- If it is a relative path, does it resolve correctly from the document's location?
- If it is a link (`[text](path)`), does the target exist?

### Category B: Directory Tree Diagrams

Identify ASCII tree diagrams (lines with indented entries, `/` suffixes, or tree-drawing characters).

For each tree:
- Does every listed file/directory exist at the indicated location?
- Are there files/directories in the actual tree that the diagram omits? (Flag only if the diagram claims to be complete, not if it uses `...` or explicitly notes omissions.)
- Is the nesting structure accurate?

### Category C: CLI Commands and Flags

Identify documented CLI commands, flags, and usage examples.

For each command:
- If it references a project script or entry point, does that entry point exist?
- Do the documented flags/arguments match the actual CLI definition? (Check argparse, click, typer definitions, or `--help` output.)
- Are example invocations syntactically valid for the current CLI?

### Category D: Code Examples

Identify inline code blocks and code snippets that reference project code.

For each code example:
- Do referenced imports exist? Are module paths correct?
- Do function/class names match actual definitions?
- Do function signatures (parameter names, order, types) match actual code?
- Are return types or behaviors described accurately?

### Category E: Feature Inventory

Identify claims about what exists, what is planned, and what is not yet implemented. Look for phrases like "currently supports", "planned", "coming soon", "next release", "TODO", "not yet implemented", roadmap checkboxes.

For each claim:
- Items documented as "exists" or "implemented": verify they actually exist in the codebase
- Items documented as "planned" or "next": check if they have already been implemented (stale roadmap)
- Checked-off roadmap items: verify the feature is actually present
- Unchecked roadmap items: check if they are actually already done

### Category F: Convention Claims

Identify claims about tooling, configuration, or conventions. Examples: "we use ruff for linting", "tests run with pytest", "formatted with black", "CI runs on GitHub Actions".

For each claim:
- Is the referenced tool present in config files (pyproject.toml, setup.cfg, package.json, Makefile, CI configs)?
- If a configuration file is referenced, does it exist and contain the described settings?
- Are version constraints or specific configurations accurate?

### Category G: Cross-Document Consistency

After all per-document batches complete, compare claims across documents:

- Do multiple documents describe the same component consistently? (e.g., README says "3 modules" but ARCHITECTURE.md lists 4)
- Are version numbers consistent across documents?
- Do different docs agree on directory structure, entry points, and project organization?
- If CLAUDE.md or a similar config doc describes conventions, do other docs follow them?

## Step 4: Compile Report

Combine all batch findings into a single report, grouped by document:

```
## Doc Sync Report

**Documents audited:** [count]
**Total findings:** [count] (Critical: N, Warning: N, Info: N)

---

### [path/to/document.md]

#### [CRITICAL] Category A: Missing path `src/old_module/`
Line 42: References `src/old_module/handler.py` which does not exist.
**Suggested fix:** Update path to `src/new_module/handler.py` (which contains the Handler class).

#### [WARNING] Category E: Stale roadmap item
Line 78: "[ ] Add export feature" -- but `src/export.py` already implements this.
**Suggested fix:** Check the roadmap checkbox: `[x] Add export feature`.

#### [INFO] Category B: Incomplete tree diagram
Line 15: Tree diagram omits `src/utils/` directory (3 files). Diagram does not claim completeness.
**Suggested fix:** Add `utils/` to the tree, or add `...` to indicate the tree is partial.

---

### Cross-Document Issues

#### [WARNING] Category G: Inconsistent module count
README.md (line 5) says "4 core modules" but ARCHITECTURE.md (line 12) lists 5.
**Suggested fix:** Update README.md to reflect the actual count of 5 modules.

---

### Summary

| Severity | Count |
|----------|-------|
| CRITICAL | N     |
| WARNING  | N     |
| INFO     | N     |

[1-2 sentence overall assessment of documentation health.]
```

### Severity Definitions

- **CRITICAL:** A claim is factually wrong and will mislead someone following the docs. Broken paths, wrong function signatures, nonexistent imports, commands that would fail.
- **WARNING:** A claim is stale or inconsistent but may not immediately mislead. Stale roadmap items, incomplete trees, cross-doc disagreements.
- **INFO:** Cannot verify mechanically, or technically correct but could be improved. Minor omissions, ambiguous phrasing.

## Step 5: Offer Fixes

After presenting the report, ask:

> "Which findings should I fix? Options: **all**, **critical only**, **specific items** (list numbers), or **none** (report only)."

Wait for the user's response.

For the selected items:
1. Open each document and apply the suggested fix directly.
2. For ambiguous fixes (where the correct content is unclear), ask the user before modifying.
3. After applying all fixes, present a summary of changes made.

## Rules

- Always verify against the actual codebase. Read the real files. Never rely on assumptions about what "should" exist.
- Be specific: include file paths, line numbers, and exact quoted text for every finding.
- Do not flag stylistic preferences or formatting opinions -- only factual accuracy.
- If a document uses hedging language ("might", "may", "typically"), do not flag it as inaccurate unless it is clearly wrong.
- If a code example is clearly pseudocode or illustrative (not a literal usage example), note it as INFO at most.
- Keep cross-document checks (Category G) until all per-document batches are complete.
- Cap subagent batches at 5. If there are more than 20 documents, batch aggressively by directory.
