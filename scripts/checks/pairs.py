#!/usr/bin/env python3
"""Validate every protocol-aware VLA × world-model pair manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.planning.inventory import validate_pairs
    print(f"Pair protocols OK: {len(validate_pairs(ROOT))} manifests")


if __name__ == "__main__":
    main()
