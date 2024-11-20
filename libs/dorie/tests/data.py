"""Module defines common test data."""

from pathlib import Path

_THIS_DIR = Path(__file__).parent

_EXAMPLES_DIR = _THIS_DIR / "unit_tests" / "examples"

# Paths to test csv files
SENTIMATE_CSV = _EXAMPLES_DIR / "sentimate-data.csv"