"""Conservative, recoverable cleanup; weights and model environments are never targets."""
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import tempfile

from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
from vlaworldeval.asset_lock import asset_destination_lock


GENERATED_ROOTS = {"build", "models", "vlaworldeval.egg-info", ".pytest_cache", ".ruff_cache"}
RESULT_GROUPS = {"experiments", "delivery", "acceptance", "vla_official_smoke", "integration"}


def _path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe cleanup target: {relative}")
    if not (path.parts[0] in GENERATED_ROOTS and len(path.parts) == 1
            or len(path.parts) >= 3 and path.parts[0] == "results" and path.parts[1] in RESULT_GROUPS
            or len(path.parts) == 3 and path.parts[:2] == ("docs", "archive") and path.suffix == ".md"):
        raise ValueError(f"Cleanup target outside permitted generated outputs: {relative}")
    value = root / path
    if value.is_symlink() or value.resolve() != value.absolute():
        raise ValueError(f"Cleanup refuses symlink/escaping target: {relative}")
    return value


def plan_cleanup(root):
    root = Path(root).resolve()
    policy = json.loads((root / "configs/result_retention.json").read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported result retention policy")
    cutoff = datetime.fromisoformat(policy["retire_before_utc"]).timestamp()
    candidates = []
    def add(path, reason):
        relative = str(path.relative_to(root))
        _path(root, relative)
        candidates.append({"source": relative, "reason": reason})

    for relative in policy.get("retired_paths", []):
        path = _path(root, relative)
        if not path.exists():
            continue
        markers = [path / "report.json", path / "summary.json"] if path.is_dir() else []
        if any(p.is_file() and json.loads(p.read_text()).get("status") == "running" for p in markers):
            continue
        add(path, "explicitly superseded integration output or retired proposal; recoverable")

    # Retire only known experiment families, never unknown/new user experiments.
    for name in policy["retired_experiments"]:
        path = root / "results/experiments" / name
        if path.exists() and name not in policy["keep_experiments"]:
            if path.stat().st_mtime > cutoff:
                continue
            summaries = list(path.glob("attempt_*/summary.json"))
            if any(json.loads(p.read_text()).get("status") == "running" for p in summaries):
                continue
            add(path, "superseded experiment family")
    for group, keep in (("acceptance", policy["keep_acceptance"]), ("vla_official_smoke", policy["keep_vla_smoke"])):
        directory = root / "results" / group
        for path in sorted(directory.iterdir()) if directory.exists() else []:
            if path.name not in keep and path.is_dir() and not path.is_symlink():
                if path.stat().st_mtime > cutoff:
                    continue
                marker = path / ("report.json" if group == "acceptance" else "summary.json")
                if marker.exists() and json.loads(marker.read_text()).get("status") != "running":
                    add(path, "superseded startup/acceptance run")
    delivery = root / "results/delivery"
    for path in sorted(delivery.iterdir()) if delivery.exists() else []:
        if path.is_symlink():
            continue
        if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in policy["keep_delivery_patterns"]):
            marker = path / "report.json"
            if marker.is_file():
                report = json.loads(marker.read_text())
                if report.get("status") == "running":
                    continue
                if report.get("scope") == "fresh_weight_free_public_delivery":
                    for name in policy["gate_scratch_names"]:
                        scratch = path / name
                        if scratch.exists():
                            add(scratch, "completed delivery gate scratch workspace")
            continue
        if path.is_dir() and (path / "report.json").is_file():
            if json.loads((path / "report.json").read_text()).get("status") == "running":
                continue
        if path.stat().st_mtime > cutoff:
            continue
        add(path, "superseded engineering/debug artifact")
    for name in policy["generated_root_names"]:
        path = root / name
        if path.exists() and name in GENERATED_ROOTS:
            if name == "models" and any(p.is_file() and p.suffix not in {".pyc", ".pyo"} for p in path.rglob("*")):
                continue
            add(path, "regenerable build/cache output")
    return {"schema_version": 1, "root": str(root), "mode": "archive_not_delete",
            "protected": ["checkpoints", "datasets", "sample_data", "environment", "third_party", ".uv-cache", ".cache"],
            "candidates": candidates, "execution_performed": False}


def archive_cleanup(root, plan):
    root = Path(root).resolve()
    if plan.get("root") != str(root) or plan.get("mode") != "archive_not_delete":
        raise ValueError("Cleanup plan does not belong to this project")
    targets = [_path(root, entry["source"]) for entry in plan["candidates"]]
    if len(set(targets)) != len(targets) or any(a != b and a in b.parents for a in targets for b in targets):
        raise ValueError("Cleanup targets overlap")
    archive_root = root / ".archive"
    if archive_root.is_symlink() or not archive_root.resolve().is_relative_to(root):
        raise ValueError("Archive must be project-local and not a symlink")
    archive_root.mkdir(exist_ok=True)
    with asset_destination_lock(root, archive_root):
        directory = Path(tempfile.mkdtemp(prefix="cleanup_", dir=archive_root))
        report = {**plan, "archive": str(directory), "created_at": datetime.now(timezone.utc).isoformat(),
                  "moves": [], "execution_performed": True, "status": "running"}
        marker = directory / "manifest.json"
        publish_json_exclusive(marker, report)
        try:
            for entry, source in zip(plan["candidates"], targets):
                # Recheck immediately before moving; never follow a swapped symlink.
                _path(root, entry["source"])
                destination = directory / entry["source"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                report["moves"].append({"source": entry["source"], "archived": str(destination)})
                replace_owned_json(marker, report)
            report["status"] = "completed"
        except BaseException:
            report["status"] = "interrupted"
            raise
        finally:
            replace_owned_json(marker, report)
    return report
