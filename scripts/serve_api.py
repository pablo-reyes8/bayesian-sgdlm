#!/usr/bin/env python3
"""Start the FastAPI service."""

from __future__ import annotations

import sys

from sgdlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
