#!/usr/bin/env python3
"""Start the Streamlit frontend."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    environment = os.environ.copy()
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]],
            env=environment,
        )
    )
