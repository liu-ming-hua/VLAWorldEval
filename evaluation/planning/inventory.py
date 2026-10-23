"""Packaged, weight-free inventory validation; no external script execution."""
import json
from pathlib import Path

from evaluation.manifests import load_pair_manifest
from evaluation.planning.catalog import load_model_catalog


def validate_catalog(root):
    catalog = load_model_catalog(root)
    return {group: len(catalog[group]) for group in ("vlas", "world_models")}


def validate_datasets(root, *, require_assets=False):
    paths = sorted(root.glob("datasets/*/manifest.json"))
    if not paths:
        raise ValueError("No dataset manifests found")
    records = []
    for path in paths:
        manifest = json.loads(path.read_text())
        value = manifest.get("local_path")
        record = {"manifest": str(path.relative_to(root)), "status": "legacy_without_local_path"}
        if value is not None:
            reference = Path(value)
            if reference.is_absolute() or ".." in reference.parts:
                raise ValueError(f"Dataset path leaves project: {value}")
            # Existing official demos may be symlinks; report rather than copy
            # or silently hide their external provenance. Execution admission
            # remains stricter and independently validates each runtime path.
            local = (root / reference).resolve()
            present = local.is_dir() and any(item.is_file() and item.name not in
                                            {"manifest.json", "README.md", ".download_complete"}
                                            for item in local.rglob("*"))
            if require_assets and not present:
                raise ValueError(f"Missing or empty dataset: {local}")
            record.update(status="present" if present else "missing_assets", local_path=str(local),
                          external_symlink=not local.is_relative_to(root.resolve()))
        records.append(record)
    return records


def validate_pairs(root):
    paths = sorted((root / "configs/pairs").glob("*.json"))
    if not paths:
        raise ValueError("No pair manifests found")
    ids = set()
    for path in paths:
        pair = load_pair_manifest(path)
        if pair.pair_id in ids:
            raise ValueError(f"Duplicate pair_id {pair.pair_id!r}")
        ids.add(pair.pair_id)
        if pair.world_model.video is None:
            raise ValueError(f"{path}: executable pairs must declare world_model.video")
        for name in ("normalizer_stats", "dynamics_calibration", "kinematics"):
            value = pair.action_bridge.config.get(name)
            if value and not (root / value).is_file():
                raise FileNotFoundError(f"{path}: missing {value}")
        value = pair.metadata.get("dataset_manifest")
        if value and not (root / value).is_file():
            raise FileNotFoundError(f"{path}: missing {value}")
    return sorted(ids)
