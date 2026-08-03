#!/usr/bin/env python3
"""Compute terminal or dynamic SGDLM impulse responses."""

from __future__ import annotations

import sys

from sgdlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["irf", *sys.argv[1:]]))
