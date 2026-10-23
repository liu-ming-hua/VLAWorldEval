#!/usr/bin/env python3
"""Download only the pinned 2B GR1 inference checkpoint using shared assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL_REVISION = "89d029e10816d2995d700cb8ba06f171e0504203"
MODEL_PREFIX = "2B_GR1_post-train"
ITERATION = "iter_000050000"
DATA_REVISION = "9f68429ad12535d30d1aae33b7945c79c830c8c4"
DATA_PREFIX = "In-lab_Eval/gr1_unified.color_puzzles_robot"
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
TOKENIZER_SIZE = 507609880
TOKENIZER_OFFICIAL_REVISION = "85f8ae7bfe8f5525c8d103429524dcf12f98bf7b"


def configure_network():
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    # Hub 0.36's hf_transfer path unlinks partials instead of resuming them.
    # Force normal HTTP resume, even if a caller inherited this setting.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("model", "tokenizer", "reason", "sample", "all"), default="model")
    args = parser.parse_args(argv)
    configure_network()
    from vlaworldeval.assets import download_snapshot
    base = ROOT / "checkpoints/world_models/dreamdojo"
    specs = {
        "model": ("nvidia/DreamDojo", MODEL_REVISION, base,
                  [f"{MODEL_PREFIX}/latest_checkpoint.txt", f"{MODEL_PREFIX}/{ITERATION}/model/*"]),
        # Historical NVIDIA registry revision is 404. The original Wan public
        # transport is byte-identical to current pinned official tokenizer.pth.
        "tokenizer": ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a",
                      base / "tokenizer", ["Wan2.1_VAE.pth"]),
        "reason": ("nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
                   base / "reason", ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]),
        "sample": ("nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1", DATA_REVISION,
                   ROOT / "datasets/dreamdojo/official",
                   [f"{DATA_PREFIX}/meta/*.json", f"{DATA_PREFIX}/meta/tasks.jsonl",
                    f"{DATA_PREFIX}/data/chunk-000/episode_000000.parquet",
                    f"{DATA_PREFIX}/videos/chunk-000/observation.images.ego_view_freq20/episode_000000.mp4"]),
    }
    for component in specs if args.component == "all" else [args.component]:
        repo, revision, target, patterns = specs[component]
        print(json.dumps({"component": component, "repo": repo, "revision": revision,
                          "destination": str(target), "patterns": patterns}), flush=True)
        print(json.dumps(download_snapshot(repo, target, root=ROOT, revision=revision,
                                           patterns=patterns,
                                           repo_type="dataset" if component == "sample" else "model")), flush=True)
        if component == "tokenizer":
            from vlaworldeval.assets import MANIFEST
            record = json.loads((target / MANIFEST).read_text())["files"]["Wan2.1_VAE.pth"]
            if record["sha256"] != TOKENIZER_SHA256 or record["size"] != TOKENIZER_SIZE:
                raise ValueError("Tokenizer differs from pinned current official Cosmos bytes")
            print(json.dumps({"official_repository": "nvidia/Cosmos-Predict2.5-2B",
                              "official_revision": TOKENIZER_OFFICIAL_REVISION,
                              "official_filename": "tokenizer.pth", "sha256": TOKENIZER_SHA256,
                              "transport_repository": repo, "transport_revision": revision,
                              "historical_registry_revision_verified": False}), flush=True)


if __name__ == "__main__":
    main()
