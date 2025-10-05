"""Pool Projector package initializer."""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.append(str(_PKG_ROOT))

