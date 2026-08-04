#!/usr/bin/env python3
"""Start the OIP screening application with one command.

Usage:
    python run.py
"""

from scripts.app_launcher import main  # noqa: I001


if __name__ == "__main__":
    raise SystemExit(main())
