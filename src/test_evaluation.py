"""Backward-compatible entry point for active CNN evaluation."""

import sys

from diagnostics import main


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
