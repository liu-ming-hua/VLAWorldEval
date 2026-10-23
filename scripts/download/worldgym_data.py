"""Download one pinned official Bridge example, proxy-free, without model deps."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

COMMIT = "709665b29f5c95124e7dbc05b0d4d47cd353de97"
FILES = {
    "000000002.mp4": "64d01ecc97a83e379474471d10787de6e56be0ed9c792dd3eea2f1a4920ae8d1",
    "000000002.npz": "71b51e11ad5d7a43d8258294bff0691d4522223b4684cf80ffea8b1253aa30a5",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(root):
    destination = root.resolve() / "datasets/worldgym/official/bridge/test"
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset destination symlink leaves the project")
    destination.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({}))
    for name, expected in FILES.items():
        path = destination / name
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"Existing official example differs: {path}; not overwriting it")
            continue
        url = f"https://raw.githubusercontent.com/world-model-eval/world-model-eval/{COMMIT}/sample_data/bridge/test/{name}"
        temporary = path.with_suffix(path.suffix + ".partial")
        with opener.open(url, timeout=600) as response, temporary.open("xb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        if sha256(temporary) != expected:
            raise ValueError(f"Official sample checksum mismatch: {temporary}")
        # Exclusive final creation, retaining a partial download on failure.
        with temporary.open("rb") as source, path.open("xb") as stream:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                stream.write(block)
        temporary.unlink()
    return {"directory": str(destination), "source_commit": COMMIT, "sha256": FILES,
            "scope": "one official test episode, not the complete training dataset"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    print(json.dumps(download(parser.parse_args().root), indent=2))
