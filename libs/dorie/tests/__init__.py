import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_PARENT_DIR.parent))
