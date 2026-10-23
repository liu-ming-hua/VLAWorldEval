#!/usr/bin/env python3
"""Download only the pinned official PushT checkpoint and one mini episode."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_REPO = "yixuan1999/interactive-world-sim-checkpoints"
MODEL_REVISION = "6705a00b8366d2399e41154c60bf6bb5c1e42a60"
DATA_REPO = "yixuan1999/interactive-world-sim-min-data"
DATA_REVISION = "987c175bcc29ddd0f6e6e3096b2c20df300253b0"


def download(root: Path) -> None:
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    os.environ.update(HF_ENDPOINT="https://hf-mirror.com", HF_HOME=str(root / ".cache/huggingface"),
                      HF_HUB_DISABLE_XET="1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vlaworldeval.assets import download_snapshot

    for repo, revision, kind, destination, files in (
        (MODEL_REPO, MODEL_REVISION, "model", root / "checkpoints/world_models/iws",
         ("pusht_cam1/.hydra/config.yaml", "pusht_cam1/checkpoints/best.ckpt")),
        (DATA_REPO, DATA_REVISION, "dataset", root / "datasets/iws",
         ("pusht/val/episode_0.hdf5",)),
    ):
        report = download_snapshot(repo, destination, root=root, revision=revision,
                                   repo_type=kind, patterns=files)
        report["checksums"] = json.loads((destination / ".vlaworldeval-download.json").read_text())["files"]
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    download(parser.parse_args().root.resolve())
