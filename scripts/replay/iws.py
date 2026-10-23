#!/usr/bin/env python3
"""Prepare official PushT inputs, then run the shared persistent replay worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/replay/iws_native_v1.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/iws/pusht/val/episode_0.hdf5")
    parser.add_argument("--packet", type=Path, default=ROOT / "results/integration/open_world_models_20261018/iws/inputs_fk.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/integration/open_world_models_20261018/iws/replay_fk_v1")
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--action-source", choices=("official_fk", "stored_command_diagnostic"), default="official_fk",
                        help="Default: official observed-joint FK for every action. Stored commands are diagnostic only.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--admit-only", action="store_true")
    args = parser.parse_args()
    from evaluation.datasets.iws import prepare_iws_packet

    print(json.dumps(prepare_iws_packet(args.config, args.dataset, args.packet,
                                       start=args.start, chunks=args.chunks,
                                       action_source=args.action_source), indent=2), flush=True)
    if not args.prepare_only:
        from evaluation.execution.native_replay import run_native_replay
        print(json.dumps(run_native_replay(args.config, args.packet, args.output,
                                          admit_only=args.admit_only), indent=2))


if __name__ == "__main__":
    main()
