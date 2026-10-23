"""Inspect isolated interpreters without importing their model stacks.

A snapshot records installed versions; it is NOT a portable dependency lock
or proof that a fresh installation works. Editable/local sources are flagged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import shutil

from adapters.environment import load_environment_registry
from vlaworldeval.paths import project_root

_PROBE = '''
import json, sys
from importlib.metadata import distributions
packages = {}
local = []
search_paths = [p for p in sys.path if "site-packages" in p]
for dist in distributions():
    name = dist.metadata.get("Name")
    if not name:
        continue
    key = name.lower().replace("_", "-").replace(".", "-")
    if key in packages:
        continue  # First metadata match follows import precedence, not last.
    packages[key] = dist.version
    direct = dist.read_text("direct_url.json")
    if direct:
        value = json.loads(direct)
        local.append({"package": key,
                      "editable": bool(value.get("dir_info", {}).get("editable")),
                      "vcs": value.get("vcs_info", {}).get("vcs"),
                      "commit": value.get("vcs_info", {}).get("commit_id"),
                      "local_file_source": value.get("url", "").startswith("file:")})
print(json.dumps({"python": list(sys.version_info[:3]),
                  "packages": dict(sorted(packages.items())), "source_requirements": local,
                  "site_package_roots": search_paths}))
'''


def compare_environment(current, reference):
    """Compare complete installed distributions and Python version, not paths."""
    before, after = reference["packages"], current["packages"]
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": {name: {"expected": before[name], "actual": after[name]}
                    for name in sorted(set(before) & set(after)) if before[name] != after[name]},
        "python_changed": reference["python"] != current["python"],
        "sources_changed": sorted(reference.get("source_requirements", []), key=lambda x: x["package"])
        != sorted(current.get("source_requirements", []), key=lambda x: x["package"]),
    }


def audit_environments(*, keys=(), baseline: Path | None = None, timeout=30, check_dependencies=False):
    registry = load_environment_registry()
    unknown = set(keys) - set(registry)
    if unknown:
        raise ValueError(f"Unknown environments: {sorted(unknown)}")
    reference = None if baseline is None else json.loads(baseline.read_text())
    if reference is not None and reference.get("schema_version") != 1:
        raise ValueError("Unsupported environment snapshot schema")
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PYTHONPATH"):
        env.pop(key, None)
    records = {}
    for key in sorted(keys or registry):
        spec = registry[key]
        record = {"path": str(spec.path), "resolved_path": str(spec.path.resolve()),
                  "package_profile": spec.package_profile, "status": "missing"}
        if spec.exists:
            try:
                output = subprocess.run([str(spec.python), "-I", "-c", _PROBE], env=env,
                                        capture_output=True, text=True, timeout=timeout, check=True)
                record.update(json.loads(output.stdout), status="inspected")
                if check_dependencies:
                    uv = shutil.which("uv")
                    if uv is None:
                        raise ValueError("uv is not on PATH")
                    checked = subprocess.run([uv, "--offline", "--cache-dir", str(project_root() / ".uv-cache"),
                                              "pip", "check", "--python", str(spec.python)], env=env,
                                             capture_output=True, text=True, timeout=timeout)
                    record["dependency_check"] = {"ok": checked.returncode == 0,
                                                  "details": (checked.stdout + checked.stderr).strip()}
                if reference is not None:
                    old = reference.get("environments", {}).get(key)
                    if old is None or old.get("status") != "inspected":
                        record["status"] = "missing_baseline"
                    else:
                        record["drift"] = compare_environment(record, old)
                        if any(record["drift"].values()):
                            record["status"] = "drift"
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                record.update(status="probe_failed", error=type(exc).__name__)
        records[key] = record
    return {"schema_version": 1, "ok": all(r["status"] == "inspected" and r.get("dependency_check", {}).get("ok", True)
                                           for r in records.values()),
            "fresh_install_verified": False, "model_inference_performed": False,
            "portable_lock": False, "proxies_removed": True, "environments": records}
