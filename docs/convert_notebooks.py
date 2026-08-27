"""Convert Jupyter notebooks in notebooks/ to Markdown for Zensical.

Run this script whenever a notebook changes:
    python docs/convert_notebooks.py

Produces a .md file for each .ipynb in docs/notebooks/ using nbconvert.
Images / attachments are placed in a <name>_files/ folder next to the .md.
"""

import subprocess
import sys
from pathlib import Path

# Input: notebooks in repo root
NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples"
# Output: Markdown files in docs/notebooks
OUTPUT_DIR = Path(__file__).parent / "notebooks"


def convert(nb: Path) -> None:
    print(f"Converting {nb.name} …")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "markdown",
            "--output-dir",
            str(OUTPUT_DIR),
            str(nb),
        ],
        check=True,
    )


if __name__ == "__main__":
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found in", NOTEBOOKS_DIR)
        sys.exit(0)

    for nb in notebooks:
        convert(nb)

    print("\nDone. Re-run this script whenever a notebook changes.")
