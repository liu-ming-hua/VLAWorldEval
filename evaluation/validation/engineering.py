"""Bounded, explicit code-quality acceptance, separate from robot task quality.

Every check is executed afresh. Existing scientific/model reports are not
silently promoted to proof about current source. No model loading or downloads.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from evaluation.reporting.task_scores import _digest


def acceptance_commands(python):
    return {
        "syntax": [python, "-m", "compileall", "-q", "adapters", "cli", "evaluation", "scripts", "vlaworldeval", "tests"],
        "lint": [python, "-m", "ruff", "check", "adapters", "cli", "evaluation", "scripts", "tests", "vlaworldeval"],
        "tests": [python, "-m", "pytest", "-q"],
        "inventory": [python, "-m", "cli", "validate"],
        "capabilities": [python, "-m", "cli", "capabilities"],
        "documentation": [python, "scripts/checks/docs.py"],
    }


def checkout_fingerprints(root):
    paths = set()
    for directory, pattern in (("adapters", "*.py"), ("cli", "*.py"), ("evaluation", "*.py"),
                               ("models", "*.py"), ("vlaworldeval", "*.py"), ("tests", "*.py"),
                               ("scripts", "*.py"), ("scripts", "*.sh"), ("configs", "*.json"),
                               ("configs", "*.yaml"), ("docs", "*.md"), (".github", "*.yml"),
                               ("environment/locks", "*.txt"), ("environment/locks", "*.in"),
                               ("environment/requirements", "*.txt")):
        paths.update((root / directory).rglob(pattern))
    for name in ("pyproject.toml", "environment/registry.toml", "MANIFEST.in", ".gitignore", "README.md", "README_CN.md", "CONTRIBUTING.md", "LICENSE", "LICENSE_SCOPE.md"):
        if (root / name).is_file():
            paths.add(root / name)
    assets = root / "docs/assets"
    if assets.is_dir():
        paths.update(path for path in assets.rglob("*") if path.is_file())
    return {str(path.relative_to(root)): _digest(path) for path in sorted(paths)}


def _stop_check(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
    except ProcessLookupError:
        proc.wait(timeout=5)


def run_engineering_acceptance(root: Path, output: Path, *, timeout=180):
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 3600:
        raise ValueError("Acceptance timeout must be in (0,3600] seconds per command")
    if not (root / "tests").is_dir() or not (root / "pyproject.toml").is_file():
        raise ValueError("Engineering acceptance requires the source checkout and tests")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "scope": "current_checkout_weight_free_engineering",
              "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "model_inference_verified": False, "fresh_model_install_verified": False,
              "scientific_quality_verified": False, "checks": [], "source_sha256": {}}
    report["source_sha256"] = checkout_fingerprints(root)
    commands = acceptance_commands(sys.executable)
    shell_files = list(map(str, sorted((root / "scripts").rglob("*.sh"))))
    commands["shell_syntax"] = ["bash", "-c", 'for script_path in "$@"; do bash -n "$script_path" || exit; done',
                                "engineering-shell-check", *shell_files]
    try:
        for name, command in commands.items():
            started = time.perf_counter()
            record = {"name": name, "command": command, "log": f"{name}.log"}
            with (output / record["log"]).open("w") as handle:
                proc = None
                try:
                    proc = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    proc.wait(timeout=timeout)
                    record.update(returncode=proc.returncode, ok=proc.returncode == 0)
                except subprocess.TimeoutExpired:
                    # SIGINT lets pytest/our fixture workers run finally cleanup.
                    _stop_check(proc)
                    record.update(returncode=None, ok=False, error="timeout")
                except OSError as exc:
                    record.update(returncode=None, ok=False, error=str(exc))
                finally:
                    _stop_check(proc)
            record.update(seconds=time.perf_counter() - started, sha256=_digest(output / record["log"]))
            report["checks"].append(record)
        # Stop source drift from turning earlier checks into a misleading pass.
        report["source_unchanged"] = checkout_fingerprints(root) == report["source_sha256"]
        report["status"] = "passed" if report["source_unchanged"] and all(c["ok"] for c in report["checks"]) else "failed"
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
