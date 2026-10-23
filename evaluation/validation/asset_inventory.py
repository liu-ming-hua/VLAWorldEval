"""Read-only fingerprints of the actual assets behind declared suites.

This is a local reproducibility inventory, not upstream-signature verification.
It never downloads, installs, replaces files or promotes historical evidence.
"""
from pathlib import Path

from evaluation.planning.configuration import load_suite
from evaluation.manifests import load_pair_manifest
from evaluation.planning.registry import load_runtime_registry, project_path
from vlaworldeval.assets import file_record


def inventory_assets(root: Path, configs) -> dict:
    registry = load_runtime_registry(root)
    report = {"schema_version": 1, "scope": "local_declared_runtime_asset_fingerprints",
              "upstream_checksum_verified": False, "files": {}, "suites": {}, "missing": []}
    for config in configs:
        value = load_suite(project_path(root, str(config)))
        pair = load_pair_manifest(project_path(root, value["pair_manifest"]))
        paths = registry.asset_paths(root, pair, value["runtime"], value["sampling"])
        report["suites"][value["suite_id"]] = paths
        for name, raw in paths.items():
            path = Path(raw)
            if not path.exists():
                report["missing"].append({"asset": name, "path": raw})
                continue
            candidates = [path] if path.is_file() else [
                child for child in path.rglob("*")
                if child.is_file() and ".cache" not in child.relative_to(path).parts]
            for child in candidates:
                resolved = project_path(root, str(child.resolve()))
                key = str(resolved.relative_to(root.resolve()))
                if key not in report["files"]:
                    report["files"][key] = file_record(resolved)
    report["ok"] = not report["missing"] and bool(report["files"])
    report["total_bytes"] = sum(v["size"] for v in report["files"].values())
    return report
