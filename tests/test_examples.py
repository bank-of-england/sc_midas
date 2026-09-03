"""Smoke test: every script in ``examples/`` runs to completion.

The example scripts are rendered verbatim into the docs (``docs/examples/*.md``
via snippet includes), so a broken example is broken documentation.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py"))

# Scripts that need an optional dependency stack to run.
OPTIONAL_DEPS = {
    "realtime_midas.py": ("forecast_realtime", "forecast_evaluation", "news_decomp")
}


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs(script):
    for module in OPTIONAL_DEPS.get(script, ()):
        pytest.importorskip(module)
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MPLBACKEND": "Agg"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
