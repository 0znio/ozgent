"""Tools that ship with ozgent."""

from __future__ import annotations

from pathlib import Path


def discover() -> list[str]:
    """Module names in this package, excluding private files."""
    here = Path(__file__).parent
    return sorted(
        p.stem for p in here.glob("*.py") if not p.stem.startswith("_")
    )
