"""Fingerprint actual declared model assets; no downloading or inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.validation.asset_inventory import inventory_assets  # noqa: E402
from vlaworldeval.paths import project_root  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    root = project_root()
    configs = args.config or sorted(str(p.relative_to(root)) for p in (root / "configs/experiments").glob("*.json"))
    report = inventory_assets(root, configs)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
