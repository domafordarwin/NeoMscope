"""Shared pytest fixtures for NeoMscope tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from inference...`, `from tools...`, `from training...` imports
# without installing the package, so tests run from a fresh checkout.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
