#!/usr/bin/env python3
"""Stop hook: check for uncommitted changes and remind about session handoff.

Runs git status to detect uncommitted work. Prints a reminder about
session summaries. Always exits 0.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """Entry point: check git status, print reminders."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        # git not installed
        sys.exit(0)
    except subprocess.TimeoutExpired:
        sys.exit(0)

    if result.returncode != 0:
        # not a git repo or other error — silently skip
        sys.exit(0)

    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if lines:
        print(f"WARNING: {len(lines)} uncommitted change(s) detected.")

    print("REMINDER: Consider running /handoff if ending a session.")
    sys.exit(0)


if __name__ == "__main__":
    main()
