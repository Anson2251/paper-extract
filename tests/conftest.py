"""Shared pytest configuration.

The scripts under ``scripts/`` are standalone modules (not a package), so add
``scripts/`` to sys.path so ``import tidy_questions`` / ``import
validate_tidy_questions`` work from the tests.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
