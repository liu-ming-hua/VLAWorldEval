"""Validate source revisions, environments, manifests, and VLA weight layouts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "groot_n15": {
        "source": ROOT / "third_party/isaac-groot-n15",
        "revision": "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96",
        "environment": ROOT / "environment/vla/groot_n15/bin/python",
        "weight": ROOT / "checkpoints/vla/groot/n1.5-3b/model.safetensors.index.json",
    },
    "groot_n17": {
        "source": ROOT / "third_party/isaac-groot-n17",
        "revision": "23ace64f17aa5015259b8609d371eb61a357c776",
        "environment": ROOT / "environment/vla/groot_n17/bin/python",
        "weight": ROOT / "checkpoints/vla/groot/n1.7-3b/model.safetensors.index.json",
        "backbone_glob": ".cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/*/model.safetensors",
    },
    "pi0_droid": {
        "source": ROOT / "third_party/openpi",
        "revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "environment": ROOT / "environment/vla/openpi/bin/python",
        "weight": ROOT / "checkpoints/vla/openpi/pi0-droid/.complete.json",
    },
    "pi05_droid": {
        "source": ROOT / "third_party/openpi",
        "revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "environment": ROOT / "environment/vla/openpi/bin/python",
        "weight": ROOT / "checkpoints/vla/openpi/pi05-droid/.complete.json",
    },
    "smolvla": {
        "source": ROOT / "third_party/lerobot",
        "revision": "e595b7902714ba51f91e47523f66f89c5181b649",
        "environment": ROOT / "environment/vla/smolvla/bin/python",
        "weight": ROOT / "checkpoints/vla/smolvla/base/model.safetensors",
    },
}


def revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    errors: list[str] = []
    for manifest in ("groot.json", "openpi.json", "smolvla.json"):
        json.loads((ROOT / "configs/vlas" / manifest).read_text(encoding="utf-8"))
    for name, spec in MODELS.items():
        if not (spec["source"] / ".git").is_dir():
            errors.append(f"{name}: source missing: {spec['source']}")
        elif revision(spec["source"]) != spec["revision"]:
            errors.append(f"{name}: source revision mismatch")
        if not spec["environment"].is_file():
            errors.append(f"{name}: environment missing: {spec['environment']}")
        if not spec["weight"].exists():
            errors.append(f"{name}: weight marker missing: {spec['weight']}")
        elif spec["weight"].name == ".complete.json":
            marker = json.loads(spec["weight"].read_text(encoding="utf-8"))
            checkpoint = spec["weight"].parent
            for relative, expected in marker.get("files", {}).items():
                expected_size = expected["size"] if isinstance(expected, dict) else expected
                artifact = checkpoint / relative
                if not artifact.is_file() or artifact.stat().st_size != expected_size:
                    errors.append(f"{name}: incomplete/corrupt file: {artifact}")
            partials = list(checkpoint.rglob("*.part"))
            if partials:
                errors.append(f"{name}: unfinished partial downloads: {len(partials)}")
        backbone_glob = spec.get("backbone_glob")
        if backbone_glob and not list(ROOT.glob(backbone_glob)):
            errors.append(f"{name}: required backbone missing: {backbone_glob}")
    if errors:
        raise SystemExit("VLA integration validation failed:\n- " + "\n- ".join(errors))
    print(f"VLA integrations OK: {len(MODELS)} policies")


if __name__ == "__main__":
    main()
