"""Validate the project model catalog without importing heavy model packages."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.planning.inventory import validate_catalog
    print(f"Model catalog OK: {validate_catalog(ROOT)}")


if __name__ == "__main__":
    main()
