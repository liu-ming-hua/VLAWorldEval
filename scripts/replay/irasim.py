#!/usr/bin/env python3
"""Prepare official sample, then invoke the shared native replay executor."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Configure before optional model libraries import and choose their caches.
os.environ.update(XDG_CACHE_HOME=str(ROOT / ".cache"), TORCH_HOME=str(ROOT / ".cache/torch"),
                  HF_HOME=str(ROOT / ".cache/huggingface"), HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
                  HF_ENDPOINT="https://hf-mirror.com")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/replay/irasim_native_v1.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/irasim/languagetable")
    parser.add_argument("--output", type=Path, default=ROOT / "results/integration/open_world_models_20261018/irasim/official_v2")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--admit-only", action="store_true")
    parser.add_argument("--check-context", action="store_true", help="CPU VAE reconstruction of official initial latent, then exit")
    args = parser.parse_args()
    from evaluation.datasets.irasim import prepare_sample
    if args.check_context:
        from evaluation.datasets.irasim import check_initial_context
        print(json.dumps(check_initial_context(args.dataset, ROOT / "checkpoints/world_models/irasim/sdxl_vae", args.output), indent=2))
        return
    packet = args.output.parent / f"{args.output.name}_input.npz"
    report = prepare_sample(args.config, args.dataset, packet)
    if not args.prepare_only:
        from evaluation.execution.native_replay import run_native_replay
        report = run_native_replay(args.config, packet, args.output, admit_only=args.admit_only)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
