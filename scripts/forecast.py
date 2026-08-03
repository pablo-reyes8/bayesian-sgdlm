#!/usr/bin/env python3
"""Generate forecasts from a fitted SGDLM artifact."""

from __future__ import annotations

import sys

from sgdlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["forecast", *sys.argv[1:]]))
