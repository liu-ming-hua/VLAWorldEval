#!/usr/bin/env python3
"""Fetch only the first official Language-Table Frame-Ada archive member and VAE.

The dataset is split gzip/tar, not a Hugging Face model snapshot. Shards are
downloaded lazily, checksum verified through the project's shared asset helper.
No training/evaluation dataset archives are selected.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vlaworldeval.assets import download_snapshot, file_record  # noqa: E402
from vlaworldeval.asset_lock import destination_locked  # noqa: E402
from vlaworldeval.atomic_json import publish_json_exclusive  # noqa: E402

SOURCE_COMMIT = "c72b6dade6fcd65971e0aa8ab49ea39b15108c90"
DATASET_COMMIT = "dfcbf85c27b2c5d041dbf5df3df272111e758b48"
VAE_COMMIT = "462165984030d82259a11f4367a4eed129e94a7b"
MEMBER = "opensource_robotdata/languagetable/checkpoints/frame_ada/0300000.pt"
CHECKPOINT_SIZE = 10869764914


def configure_environment():
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    os.environ.update(
        HF_ENDPOINT="https://hf-mirror.com",
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        HF_DATASETS_CACHE=str(ROOT / ".cache/huggingface/datasets"),
        XDG_CACHE_HOME=str(ROOT / ".cache"),
        TORCH_HOME=str(ROOT / ".cache/torch"),
        UV_CACHE_DIR=str(ROOT / ".uv-cache"),
    )


class ShardStream(io.RawIOBase):
    """Concatenate only as many verified shards as tar extraction consumes."""

    def __init__(self, revision, cache):
        self.revision, self.cache = revision, cache
        self.index, self.current = 0, None
        self.consumed = []

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded archive reads are forbidden")
        output = bytearray()
        while len(output) < size:
            if self.current is None:
                if self.index >= 6:
                    raise OSError("Checkpoint member exceeded expected shard limit")
                name = f"languagetable_checkpoints_data.tar.gz.sfpart_a{chr(97 + self.index)}"
                directory = self.cache / f"part_{self.index:02d}"
                print(f"Fetching verified checkpoint shard: {name}", flush=True)
                for attempt in range(201):
                    try:
                        report = download_snapshot("fangqi/IRASim", directory, root=ROOT,
                                                   revision=self.revision, patterns=[name], repo_type="dataset")
                        break
                    except ValueError as error:
                        if "Asset operation already running" not in str(error) or attempt == 200:
                            raise
                        time.sleep(3)
                self.revision = report["revision"]
                self.consumed.append({"name": name, "revision": self.revision})
                self.current = (directory / name).open("rb")
                self.index += 1
            block = self.current.read(size - len(output))
            if not block:
                self.current.close()
                self.current = None
            else:
                output.extend(block)
        return bytes(output)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


@destination_locked("directory")
def extract_checkpoint(directory, *, root, revision=DATASET_COMMIT):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "languagetable_frame_ada.pt"
    manifest = directory / "checkpoint.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["member"] != MEMBER or info["file"] != file_record(target):
            raise ValueError("Existing checkpoint differs from its manifest; preserving it")
        return info
    if target.exists():
        raise FileExistsError(f"Unmanifested checkpoint preserved: {target}")
    with ShardStream(revision, ROOT / ".cache/irasim/checkpoint_shards") as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name != MEMBER or member.size != CHECKPOINT_SIZE:
                    raise ValueError(f"Unexpected first checkpoint member: {member.name}, {member.size}")
                # Extract an exact approved member, never tar paths/links.
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".checkpoint-", delete=False) as output:
                    stage = Path(output.name)
                    with archive.extractfile(member) as source:
                        shutil.copyfileobj(source, output, length=8 * 1024**2)
                if stage.stat().st_size != member.size:
                    raise OSError("Incomplete checkpoint member")
                target.hardlink_to(stage)
                stage.unlink()
                info = {"repository": "fangqi/IRASim", "revision": stream.revision,
                        "member": MEMBER, "file": file_record(target), "shards": stream.consumed}
                publish_json_exclusive(manifest, info)
                return info
    raise ValueError("Checkpoint member not found")


def copy_sample():
    repo = ROOT / "third_party/irasim"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("Official sample source commit differs")
    source = repo / "robotdata/opensource_robotdata/languagetable"
    destination = ROOT / "datasets/irasim/languagetable"
    records = {}
    for name in ("annotation/test/example.json", "videos/test/example.mp4", "latent_videos/test/example.pt"):
        path, target = source / name, destination / name
        expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD:" + str(path.relative_to(repo))], text=True).strip()
        actual = subprocess.check_output(["git", "-C", str(repo), "hash-object", str(path)], text=True).strip()
        if actual != expected:
            raise ValueError(f"Modified upstream sample preserved: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_record(target) != file_record(path):
                raise ValueError(f"Modified local sample preserved: {target}")
        else:
            with path.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        records[name] = {**file_record(target), "git_blob": expected}
    provenance = {"repository": "bytedance/IRASim", "revision": commit, "files": records}
    manifest = destination / "sample.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != provenance:
            raise ValueError("Sample provenance differs; preserving it")
    else:
        publish_json_exclusive(manifest, provenance)
    return {"source_commit": commit, "directory": str(destination)}


def main():
    configure_environment()  # Before huggingface_hub captures endpoint/cache constants.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--revision", default=DATASET_COMMIT, choices=[DATASET_COMMIT], help="Pinned official HF dataset commit")
    args = parser.parse_args()
    print(json.dumps(copy_sample(), indent=2), flush=True)
    if not args.sample_only:
        print(json.dumps(extract_checkpoint(ROOT / "checkpoints/world_models/irasim", root=ROOT, revision=args.revision), indent=2), flush=True)
        print(json.dumps(download_snapshot("stabilityai/stable-diffusion-xl-base-1.0",
              ROOT / "checkpoints/world_models/irasim/sdxl_vae", root=ROOT,
              revision=VAE_COMMIT, patterns=["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
