"""Run separate training, validation, test, and external CNN sanity checks."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from diagnostics import main  # noqa: E402


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
