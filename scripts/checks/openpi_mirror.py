#!/usr/bin/env python3
"""Verify mirrored FAST weights against official metadata, without GCS downloads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vlaworldeval.openpi_mirror import verify_openpi_mirror  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT / "checkpoints/vla/openpi/pi0-fast-droid")
    print(json.dumps(verify_openpi_mirror(parser.parse_args().checkpoint), indent=2))
