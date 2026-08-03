#!/usr/bin/env python3
"""Fit an SGDLM model from flags or YAML configuration."""

from __future__ import annotations

import sys

from sgdlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fit", *sys.argv[1:]]))
