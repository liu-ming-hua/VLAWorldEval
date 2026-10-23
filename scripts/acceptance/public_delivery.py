"""Fresh weight-free source/wheel delivery gate. No model stacks or publishing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
import tempfile
from evaluation.validation.engineering import checkout_fingerprints
from scripts.acceptance.export_source import PRESENTATION_ASSETS

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_archives(wheel, sdist):
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist) as archive:
        source_names = archive.getnames()
    for name in wheel_names + source_names:
        # Model guides are source documentation, not the runtime models tree.
        model_guide = "/docs/models/" in "/" + name and name.endswith(".md")
        presentation = "/".join(name.split("/")[1:]) in PRESENTATION_ASSETS
        if ("site-packages/" in name or "/.cache/" in name or "/.uv-cache/" in name
                or "/third_party/" in name or "/.archive/" in name or name.startswith("models/")
                or ("/models/" in name and not model_guide)
                or name.endswith((".safetensors", ".pt", ".npz", ".npy"))
                or (name.endswith(".mp4") and not presentation)):
            raise ValueError(f"Unexpected runtime/model/data asset in package: {name}")
    for names, suffixes in ((wheel_names, ("configs/runtime_registry.json", "configs/kinematics/droid_panda_link8_v1.json",
                                         "configs/benchmarks/native_domains_v1.json",
                                         "environment/requirements/video_metrics.txt",
                                         "environment/registry.toml", "LICENSE", "LICENSE_SCOPE.md")),
                            (source_names, ("scripts/setup/core.sh", "scripts/setup/common.sh",
                                            "environment/locks/core-py311.txt",
                                            "environment/locks/build-tools-py311.txt"))):
        for suffix in suffixes:
            if not any(name.endswith("/" + suffix) for name in names):
                raise ValueError(f"Missing distribution resource: {suffix}")
    return {"wheel_files": len(wheel_names), "sdist_files": len(source_names),
            "wheel_sha256": digest(wheel), "sdist_sha256": digest(sdist)}


def run_gate(output, *, offline=False, timeout=180):
    output = output.absolute()
    if not output.resolve().is_relative_to(ROOT.resolve()) or output.is_symlink():
        raise ValueError("Output must be a new directory inside this project")
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 3600:
        raise ValueError("Timeout must be in (0,3600] seconds")
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("uv is required")
    output.mkdir(parents=True, exist_ok=False)
    scratch_parent = ROOT / ".cache/validation"
    if scratch_parent.is_symlink() or not scratch_parent.resolve().is_relative_to(ROOT):
        raise ValueError("Validation scratch directory must stay inside the project cache")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="delivery_", dir=scratch_parent))
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PYTHONPATH", "VLAWORLDEVAL_ROOT"):
        environment.pop(key, None)
    environment.update(UV_CACHE_DIR=str(ROOT / ".uv-cache"),
                       HF_HOME=str(ROOT / ".cache/huggingface"),
                       UV_INDEX_URL=environment.get("PYPI_MIRROR", "https://pypi.tuna.tsinghua.edu.cn/simple"),
                       UV_PYTHON_DOWNLOADS="never", UV_HTTP_TIMEOUT="600")
    flags = ["--offline"] if offline else []
    report = {"schema_version": 1, "status": "running", "scope": "fresh_weight_free_public_delivery",
              "model_installation_verified": False, "model_inference_verified": False,
              "scientific_quality_verified": False, "public_release_cleared": False,
              "offline": offline, "checks": []}
    report["scratch_directory"] = str(scratch)
    report["source_sha256"] = checkout_fingerprints(ROOT)

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def execute(name, command, *, cwd=ROOT, env=None):
        started = time.perf_counter()
        record = {"name": name, "command": list(map(str, command)), "log": name + ".log"}
        process = None
        with (output / record["log"]).open("w") as handle:
            try:
                process = subprocess.Popen(command, cwd=cwd, env=env or environment, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                process.wait(timeout=timeout)
                record.update(returncode=process.returncode, ok=process.returncode == 0)
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                record.update(seconds=time.perf_counter() - started,
                              sha256=digest(output / record["log"]))
                report["checks"].append(record)
                save()
        if not record.get("ok"):
            raise ValueError(f"Delivery check failed: {name}; inspect {output / record['log']}")

    try:
        execute("build", [uv, "build", "--no-build-isolation", "--python", sys.executable,
                          "--out-dir", str(output / "distributions"), *flags])
        wheel, = (output / "distributions").glob("*.whl")
        sdist, = (output / "distributions").glob("*.tar.gz")
        report["archives"] = check_archives(wheel, sdist)
        execute("fresh_venv", [uv, "venv", str(scratch / "core_env"), "--python", sys.executable, *flags])
        python = scratch / "core_env/bin/python"
        execute("locked_dependencies", [uv, "pip", "install", "--python", str(python),
                                         "--index-url", environment["UV_INDEX_URL"], *flags,
                                         "-r", str(ROOT / "environment/locks/core-py311.txt")])
        execute("install_wheel", [uv, "pip", "install", "--python", str(python), *flags,
                                  "--no-deps", str(wheel)])
        execute("isolated_import", [str(python), "-c",
                "import cli,importlib.util; assert '/core_env/' in cli.__file__; "
                "assert not any(importlib.util.find_spec(m) for m in ['torch','jax','tensorflow','transformers'])"], cwd=scratch)
        execute("init", [str(python), "-m", "cli", "init", "--directory", str(scratch / "workspace")], cwd=scratch)
        asset_env = {**environment, "VLAWORLDEVAL_ROOT": str(scratch / "workspace")}
        execute("validate", [str(python), "-m", "cli", "validate"], cwd=scratch, env=asset_env)
        execute("portable_evaluate", [str(python), "-c",
                "import json,os,subprocess,sys; "
                "p=subprocess.run([sys.executable,'-m','cli','evaluate','--config',"
                "'configs/experiments/openvla_worldgym_v1.json','--experiment','loop_smoke','--check-only'],capture_output=True,text=True); "
                "v=json.loads(p.stdout); assert p.returncode==1 and v['status']=='preflight_failed'; "
                "assert v['preflight']['config'].startswith(os.environ['VLAWORLDEVAL_ROOT']+'/'); "
                "assert not v['execution_performed']"], cwd=scratch, env=asset_env)
        execute("configure", [str(python), "-m", "cli", "configure", "--config",
                "configs/experiments/openvla_worldgym_v1.json", "--output", "configs/experiments/new_machine.json",
                "--vla-gpus", "2", "--wm-gpus", "3", "--chunks", "40"], cwd=scratch, env=asset_env)
        execute("installed_workers", [str(python), str(ROOT / "scripts/acceptance/installed_core_smoke.py"),
                                       "--output", str(scratch / "workers")], cwd=scratch, env=asset_env)
        worker_report = json.loads((scratch / "workers/report.json").read_text())
        if worker_report.get("status") != "passed":
            raise ValueError("Installed-worker audit did not pass")
        source_parent = scratch / "source"
        source_parent.mkdir()
        with tarfile.open(sdist) as archive:
            for member in archive:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe source-distribution member: {member.name}")
            archive.extractall(source_parent, filter="data")
        source, = source_parent.iterdir()
        # Reuse downloaded wheels, never another virtual environment. The
        # extracted source's installer still creates a genuinely new venv.
        (source / ".uv-cache").symlink_to(ROOT / ".uv-cache", target_is_directory=True)
        source_env = {**environment, "UV_BIN": uv, "CORE_PYTHON": sys.executable,
                      "CORE_OFFLINE": "1" if offline else "0", "CORE_DEV": "1"}
        execute("source_setup", ["bash", str(source / "scripts/setup/core.sh")], cwd=source, env=source_env)
        source_python = source / "environment/core/bin/python"
        execute("source_validate", [str(source_python), "-m", "cli", "validate"], cwd=source, env=source_env)
        execute("source_workers", [str(source_python), str(source / "scripts/acceptance/installed_core_smoke.py"),
                                    "--output", str(scratch / "source_workers")], cwd=source, env=source_env)
        report["source_installed_workers"] = json.loads((scratch / "source_workers/report.json").read_text())
        if report["source_installed_workers"].get("status") != "passed":
            raise ValueError("Source-install worker audit did not pass")
        report["source_unchanged"] = checkout_fingerprints(ROOT) == report["source_sha256"]
        if not report["source_unchanged"]:
            raise ValueError("Source changed during delivery checks; rerun on the final source")
        report.update(status="passed", installed_workers=worker_report)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        save()
        raise
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    arguments = parser.parse_args()
    print(json.dumps(run_gate(arguments.output, offline=arguments.offline, timeout=arguments.timeout), indent=2))
