#!/usr/bin/env python3
"""Repair a copied runtime explicitly, without rewriting scientific evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

OLD_ROOT = "/data1/liuming/VLAWorldEval"
OLD_PYTHON = "/home/liuming/.local/share/uv/python"


def replacements(root: Path) -> tuple[tuple[str, str], ...]:
    return ((OLD_ROOT, str(root)),
            ("/data1/liuming/world-model-eval", str(root)),
            (OLD_PYTHON, str(root / ".tools/python")))


def rebase(value: str, root: Path) -> str:
    for old, new in replacements(root):
        value = value.replace(old, new)
    return value


def candidates(root: Path, only_core: bool):
    # os.walk does not follow model-environment symlinks or inspect model tensors.
    starts = [root / "environment/core" if only_core else root / "environment",
              root / ".tools/python", root / ".cache/uv/python"]
    for start in starts:
        for directory, dirs, files in os.walk(start, followlinks=False):
            for name in dirs + files:
                p = Path(directory) / name
                if p.is_symlink():
                    yield p, "symlink"
            for name in files:
                p = Path(directory) / name
                if p.is_symlink():
                    continue
                if (name == "pyvenv.cfg" or p.suffix in {".pth", ".egg-link"}
                        or (name.startswith("__editable__") and p.suffix == ".py")
                        or p.parent.name == "bin"):
                    yield p, "text"
    if not only_core:
        for name in ("environment/registry.toml", "configs/world_models/tau0_wm_simulator.runtime.yaml"):
            p = root / name
            if p.is_file() and not p.is_symlink():
                yield p, "text"


def relocate(root: Path, *, apply: bool = False, only_core: bool = False) -> dict:
    root = root.absolute()
    if root.is_symlink() or str(root) == OLD_ROOT:
        raise ValueError("Use a real destination directory, never the active source project")
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Destination is not a project checkout")
    backup = root / ".tools/relocation/originals"
    changes, unresolved = [], []
    for p, kind in candidates(root, only_core):
        if not p.parent.resolve().is_relative_to(root.resolve()):
            unresolved.append({"path": str(p), "reason": "parent_resolves_outside_project"})
            continue
        rel = p.relative_to(root).as_posix()
        if kind == "symlink":
            old = os.readlink(p)
            new = rebase(old, root)
            if new != old:
                changes.append({"path": rel, "kind": kind, "before": old, "after": new})
                if apply:
                    target_backup = backup / (rel + ".symlink.json")
                    target_backup.parent.mkdir(parents=True, exist_ok=True)
                    if not target_backup.exists():
                        with target_backup.open("x") as f:
                            json.dump({"target": old}, f)
                    # Atomic replacement, no traversal/deletion of link targets.
                    fd, temporary = tempfile.mkstemp(dir=p.parent, prefix=".relocate-")
                    os.close(fd)
                    os.unlink(temporary)
                    os.symlink(new, temporary)
                    os.replace(temporary, p)
            resolved = Path(new) if Path(new).is_absolute() else p.parent / new
            if not resolved.exists():
                unresolved.append({"path": rel, "target": new, "reason": "missing_link_target"})
            continue
        # Executable ELF/shared-library files in bin must never be text-rewritten.
        with p.open("rb") as f:
            header = f.read(8192)
        if b"\0" in header or header.startswith(b"\x7fELF"):
            continue
        original = p.read_bytes()
        try:
            old = original.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = rebase(old, root)
        if rel == "environment/registry.toml":
            new = new.replace('path = "/data1/liuming/giga-world-1/.venv"',
                              'path = "environment/world_models/gigaworld1"')
        if new != old:
            updated = new.encode()
            changes.append({"path": rel, "kind": kind,
                            "before_sha256": hashlib.sha256(original).hexdigest(),
                            "after_sha256": hashlib.sha256(updated).hexdigest()})
            if apply:
                saved = backup / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                if not saved.exists():
                    with saved.open("xb") as f:
                        f.write(original)
                mode = stat.S_IMODE(p.stat().st_mode)
                fd, temporary = tempfile.mkstemp(dir=p.parent, prefix=".relocate-")
                with os.fdopen(fd, "wb") as f:
                    f.write(updated)
                os.chmod(temporary, mode)
                os.replace(temporary, p)
        for line in new.splitlines():
            if any(token in line for token in ("/data0/", "/data1/liuming/", "/home/liuming/")):
                unresolved.append({"path": rel, "reason": "unmapped_external_path", "line": line[:500]})
    # Links encountered early may become valid when the managed-Python alias is
    # repaired later in this pass. Report final rather than transient failures.
    if apply:
        unresolved = [item for item in unresolved
                      if item["reason"] != "missing_link_target"
                      or not (root / item["path"]).exists()]
    return {"schema_version": 1, "root": str(root), "applied": apply,
            "scope": "core" if only_core else "runtime_metadata_and_known_templates",
            "changes": changes, "unresolved": unresolved,
            "historical_plans_results_and_weight_metadata_modified": False,
            "model_inference_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--only-core", action="store_true")
    args = parser.parse_args()
    report = relocate(args.root, apply=args.apply, only_core=args.only_core)
    if args.apply:
        destination = args.root / ".tools/relocation" / ("core.json" if args.only_core else "runtime.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep individual run reports: do not overwrite the first change audit on reapply.
        fd, name = tempfile.mkstemp(prefix=destination.stem + "-", suffix=".json", dir=destination.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(report, f, indent=2)
        report["audit_file"] = name
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
