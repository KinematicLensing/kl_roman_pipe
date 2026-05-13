#!/usr/bin/env python3
"""PreToolUse hook: block suspicious test edits.

Reads Claude Code hook JSON from stdin. Exits non-zero to block edits
that weaken tests (skip markers, trivial assertions, noqa in tests,
assertion removal).
"""

from __future__ import annotations

import json
import re
import sys


def _is_test_file(path: str) -> bool:
    """Return True if path looks like a test file."""
    return "/tests/" in path or "/test_" in path or path.endswith("_test.py")


_SUSPICIOUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"@pytest\.mark\.skip"), "@pytest.mark.skip detected"),
    (re.compile(r"@pytest\.mark\.xfail"), "@pytest.mark.xfail detected"),
    (re.compile(r"\bassert\s+True\b"), "trivial 'assert True' detected"),
    (re.compile(r"#\s*noqa"), "# noqa suppression in test file"),
    (re.compile(r"#\s*type:\s*ignore"), "# type: ignore in test file"),
]


def _check_content(content: str) -> list[str]:
    """Return list of violation descriptions found in content."""
    violations: list[str] = []
    for pattern, desc in _SUSPICIOUS_PATTERNS:
        if pattern.search(content):
            violations.append(desc)
    return violations


def _count_asserts(text: str) -> int:
    """Count lines containing assert statements."""
    return sum(1 for line in text.splitlines() if re.search(r"\bassert\b", line))


def main() -> None:
    """Entry point: read stdin JSON, check for suspicious test edits."""
    raw = sys.stdin.read().strip()
    if not raw:
        sys.exit(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = data.get("tool_input", {})
    file_path: str = tool_input.get("file_path", "")
    if not file_path or not _is_test_file(file_path):
        sys.exit(0)

    # Extract content to check (Edit: new_string, Write: content)
    new_content: str = tool_input.get("new_string", "") or tool_input.get("content", "")
    if not new_content:
        sys.exit(0)

    violations = _check_content(new_content)

    # For Edit operations: check if assertions are being removed
    old_content: str = tool_input.get("old_string", "")
    if old_content:
        old_asserts = _count_asserts(old_content)
        new_asserts = _count_asserts(new_content)
        if old_asserts > 0 and new_asserts < old_asserts:
            violations.append(
                f"assertion count reduced ({old_asserts} -> {new_asserts})"
            )

    if violations:
        print(f"BLOCKED: suspicious test edit in {file_path}")
        for v in violations:
            print(f"  - {v}")
        print("If intentional, get explicit human approval first.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
