"""Integration test configuration."""

import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))
