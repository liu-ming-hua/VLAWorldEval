"""Acquire only the official SVD assets, reusing existing project weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL_REVISION = "8e502dd6b48089a40ff092e6f8d1ede9a6211fe9"
BENCHMARK_REVISION = "044100cc22ed3a92321b8b66f799e57e71abe666"
PREVIEW_REVISION = "bc4ea8f9bf88b8266c2aadf66e608b78aa035b17"
PREVIEW_PATTERNS = ["v0_2025_12_28_2000/annotation/val/9.json",
                    "v0_2025_12_28_2000/videos/val/9/*.mp4", "v0_2025_12_28_2000/metainfo/*"]


def configure_network(root: Path) -> None:
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    os.environ.update(HF_ENDPOINT="https://hf-mirror.com", HF_HOME=str(root / ".cache/huggingface"),
                      XDG_CACHE_HOME=str(root / ".cache"), HF_HUB_DISABLE_XET="1",
                      HF_HUB_ETAG_TIMEOUT="60", HF_HUB_DOWNLOAD_TIMEOUT="600")


def reuse_base(root: Path, name: str, repo: str) -> Path:
    from huggingface_hub import HfApi
    from vlaworldeval.assets import MANIFEST, download_snapshot, remote_file_record, verify_snapshot

    target = root / "checkpoints/world_models/playworld" / name
    reference_path = target.parent / f"{name}.reference.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    existing = root / "checkpoints/world_models/ctrl_world" / name
    source = target.resolve() if target.exists() else existing
    patterns = (["model_index.json", "scheduler/*", "feature_extractor/*", "vae/config.json",
                 "vae/diffusion_pytorch_model.safetensors", "unet/config.json",
                 "unet/diffusion_pytorch_model.safetensors", "image_encoder/config.json",
                 "image_encoder/model.safetensors"] if name == "svd" else
                ["config.json", "pytorch_model.bin", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt"])
    if source.is_dir():
        # Never infer validity from a config, symlink, or mere path existence.
        marker = source / MANIFEST
        if marker.exists():
            manifest = json.loads(marker.read_text())
            if manifest["repository"] != repo:
                raise ValueError(f"Reused base repository mismatch: {source}")
            verify_snapshot(source)
            revision = manifest["revision"]
        else:
            revision = reference["revision"] if reference else "main"
        if reference and (reference["repository"] != repo or reference["source"] != str(source)):
            raise ValueError(f"Reused base identity changed: {reference_path}")
        info = HfApi(endpoint=os.environ["HF_ENDPOINT"]).model_info(repo, revision=revision, files_metadata=True)
        import fnmatch
        selected = [f for f in info.siblings if any(fnmatch.fnmatchcase(f.rfilename, p) for p in patterns)]
        if not selected:
            raise ValueError(f"No official base files found: {repo}")
        if not all(any(fnmatch.fnmatchcase(f.rfilename, pattern) for f in selected) for pattern in patterns):
            raise ValueError(f"Official base snapshot is missing required components: {repo}")
        records = {remote.rfilename: remote_file_record(source / remote.rfilename, remote) for remote in selected}
        report = {"reused_base": name, "source": str(source), "repository": repo,
                  "revision": info.sha, "files_verified": len(selected), "files": records,
                  "symlink": str(target), "verification": "official remote size and LFS/Git hash"}
        target.parent.mkdir(parents=True, exist_ok=True)
        if not reference:
            from vlaworldeval.atomic_json import publish_json_exclusive
            publish_json_exclusive(reference_path, report)
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
        print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)
    else:
        if target.is_symlink():
            raise ValueError(f"Broken base symlink: {target}")
        download_snapshot(repo, target, root=root, patterns=patterns)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preview", action="store_true", help="Requires accepted HF dataset terms/token")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    configure_network(root)
    from vlaworldeval.assets import download_snapshot

    model = root / "checkpoints/world_models/playworld"
    if not args.skip_weights:
        for name, repo in (("svd", "stabilityai/stable-video-diffusion-img2vid"),
                           ("clip", "openai/clip-vit-base-patch32")):
            print(f"base {name}: {reuse_base(root, name, repo)}", flush=True)
        print("verifying/downloading official model and dynamics checkpoints", flush=True)
        download_snapshot("tennyyyin/open-world-checkpoints", model, root=root, revision=MODEL_REVISION,
                          patterns=["wm/checkpoint-120000.pt", "action_adapter/model2_15_9.pth"])
    benchmark = root / "datasets/playworld/benchmark"
    download_snapshot("tennyyyin/open-world-benchmark", benchmark, root=root, repo_type="dataset",
                      revision=BENCHMARK_REVISION, patterns=["irom_test_carrot/*", "README.md"])
    if args.preview:
        download_snapshot("tennyyyin/playworld_dataset_preview", root / "datasets/playworld/preview", root=root,
                          repo_type="dataset", revision=PREVIEW_REVISION, patterns=PREVIEW_PATTERNS)
    print(json.dumps({"model_revision": MODEL_REVISION, "benchmark_revision": BENCHMARK_REVISION,
                      "benchmark": str(benchmark), "preview_requested": args.preview}), flush=True)


if __name__ == "__main__":
    main()
