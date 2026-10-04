"""Small helper shared by the example scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs"


def output_dir(description: str) -> Path:
    """Parse ``--out DIR`` (default: ``<repo>/docs``) and return the output directory."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the PNG")
    out: Path = parser.parse_args().out
    return out
