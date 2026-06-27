#!/usr/bin/env python
"""Build the alignment MG-vs-baryon diagnostic report."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ia_analysis.visualization.alignment_mg_baryon_report import main


if __name__ == "__main__":
    main()
