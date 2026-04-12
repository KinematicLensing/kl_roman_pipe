---
name: test-integrity
description: Review branch diff for test weakening patterns in kl_roman_pipe. Last line of defense before commit/merge.
---

# /test-integrity -- Test Integrity Review

Review the full branch diff for test weakening patterns. Comprehensive, defense-in-depth check covering both syntactic and semantic weakening.

## When to Use

- Before committing or merging a branch that modifies test files
- At the end of plan execution when tests were added or modified
- Anytime you suspect test quality may have degraded during a session

## Step 1: Determine Diff Source

If the user provided a base branch argument (e.g., `/test-integrity se/feature-a`), use that as the base instead of main/master.

Run these commands to select the appropriate diff:

1. `git diff --cached --name-only` (staged changes)
2. `git diff --name-only` (unstaged changes)
3. `git merge-base HEAD <base>` where `<base>` is the user-provided branch, or `main` (fall back to `master` if `main` does not exist)

Decision logic:
- Staged changes exist: review `git diff --cached`
- No staged but unstaged changes exist: review `git diff`
- No uncommitted changes: review `git diff $(git merge-base HEAD <base>)...HEAD`

Filter to test files only: paths containing `tests/`, or files named `test_*.py` / `*_test.py`.

If no test files are in the diff, report "No test files modified" and **PASS**.

## Step 2: Analyze Weakening Patterns

Read the filtered diff carefully. For each test file, check every category below. Use semantic understanding -- do not rely solely on regex.

### A: Skip/Disable Markers
- [ ] `@pytest.mark.skip` or `@pytest.mark.skipif` added
- [ ] `@pytest.mark.xfail` added
- [ ] `skipIf` / `skipUnless` conditions added or loosened
- [ ] Platform/environment conditions added that skip tests in CI

### B: Suppression Markers
- [ ] `# noqa` added in test files
- [ ] `# type: ignore` added in test files

### C: Trivial/Tautological Assertions
- [ ] `assert True` or `assert 1 == 1` or equivalent tautologies
- [ ] Empty test bodies: `pass`, `...` (Ellipsis), or `return`
- [ ] Assertions that can never fail given the test setup

### D: Assertion Removal/Reduction
- [ ] Assertions deleted without replacement
- [ ] Assertion count reduced in a test function
- [ ] Entire `test_*` functions or test classes deleted
- [ ] `@pytest.mark.parametrize` entries removed (test matrix shrunk)
- [ ] Test files deleted entirely

### E: Tolerance/Precision Loosening
- [ ] `atol`, `rtol`, `abs_tol`, `rel_tol`, `tolerance` arguments increased
- [ ] `pytest.approx` tolerances widened
- [ ] `numpy.testing.assert_allclose` tolerances increased
- [ ] `assertAlmostEqual` `places` argument decreased
- [ ] Floating-point comparisons changed from exact to approximate without justification

### F: Comparison Weakening
- [ ] Equality (`==`) changed to inequality (`>=`, `<=`, `>`, `<`, `!=`)
- [ ] Specific value assertion replaced with `is not None` or truthiness check
- [ ] Exact match replaced with `in`, `startswith`, or `endswith` substring check
- [ ] Numeric bounds widened (e.g., `assert x < 0.01` to `assert x < 0.1`)

### G: Mocking/Isolation Bypass
- [ ] New `@patch`, `@mock.patch`, `MagicMock` introduced where real behavior was previously tested
- [ ] `monkeypatch` replacing a function that was previously called for real
- [ ] Return values hardcoded via mock where real computation was tested

### H: Exception Handling Bypass
- [ ] Assertions wrapped in `try/except` blocks
- [ ] `pytest.raises` context scope widened to cover more code than the expected raise point
- [ ] Expected exception types broadened (e.g., `ValueError` to `Exception`)
- [ ] Error-path tests removed or weakened

### I: Commented-Out/Dead Code
- [ ] Assertions commented out (`# assert ...`)
- [ ] Test functions or blocks commented out
- [ ] Conditional `if False:` or `if 0:` wrapping test logic

### J: Fixture/Setup Weakening
- [ ] Fixture scope widened (`function` to `module` or `session`) reducing test isolation
- [ ] Setup/teardown logic removed or simplified
- [ ] Shared mutable state introduced between tests
- [ ] `autouse` fixtures added that skip or alter test behavior globally

## Step 3: Context Check

For each finding, determine if it is justified:

- **Justified removal**: The tested feature was itself removed in the same diff.
- **Justified tolerance change**: A new algorithm is intentionally less precise and this is documented.
- **Justified mocking**: Test was refactored to separate unit/integration concerns and integration tests exist elsewhere.

Mark findings as **unjustified** (default) or **justified** with a brief explanation. When in doubt, flag it -- false positives are better than missed weakening.

## Step 4: Report

Present findings inline:

```
## Test Integrity Review

**Diff source:** [branch diff vs main | staged changes | unstaged changes]
**Test files reviewed:** [count]

### Findings

#### [CATEGORY] [severity] -- [file:line-range]
**Pattern:** [which checklist item]
**Detail:** [what changed and why it weakens the test]
**Justified:** [yes/no -- explanation if yes]

### Summary

- High severity: [count]
- Medium severity: [count]
- Low severity: [count]
- Justified (excluded): [count]

### Verdict: [PASS | FAIL]

[PASS if zero unjustified high/medium findings. FAIL otherwise.]
```

## Severity

- **High**: assertion/test removal, exception swallowing, skip/xfail markers, tautological assertions
- **Medium**: tolerance loosening, comparison weakening, new mocks replacing real behavior, fixture scope widening
- **Low**: parametrize reduction, suppression markers (noqa/type:ignore), commented-out code

## Rules

- Be comprehensive. Re-check everything, trust nothing. This is the last line of defense.
- Be specific: include file paths, line numbers from the diff, and exact before/after.
- False positives are better than missed weakening. When in doubt, flag it.
- If a finding has a corresponding code change (feature removed), note it as potentially justified but still flag for human review.
- This is a focused checklist review, not an open-ended code audit. Do not expand scope beyond test integrity.
