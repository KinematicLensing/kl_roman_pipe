#!/usr/bin/env python3
"""PostToolUse hook: run ruff check on edited Python files.

Reads Claude Code hook JSON from stdin. Runs ruff on the edited file
and prints any issues as feedback. Always exits 0 (informational only).
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    """Entry point: read stdin JSON, run ruff on edited Python files."""
    raw = sys.stdin.read().strip()
    if not raw:
        sys.exit(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = data.get("tool_input", {})
    file_path: str = tool_input.get("file_path", "")
    if not file_path or not file_path.endswith(".py"):
        sys.exit(0)

    try:
        result = subprocess.run(
            ["ruff", "check", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        # ruff not installed — silently skip
        sys.exit(0)
    except subprocess.TimeoutExpired:
        sys.exit(0)

    if result.returncode != 0 and result.stdout.strip():
        print(f"ruff issues in {file_path}:")
        print(result.stdout.strip())

    sys.exit(0)


if __name__ == "__main__":
    main()
