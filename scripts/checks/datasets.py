#!/usr/bin/env python3
"""Check that every local official demo manifest resolves to usable files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.planning.inventory import validate_datasets
    print(json.dumps({"root": str(ROOT), "datasets": validate_datasets(ROOT, require_assets=True)}, indent=2))


if __name__ == "__main__":
    main()
