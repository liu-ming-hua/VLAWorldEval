"""Keep reusable code/configs independent of the operator's machine layout."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_shared_code_and_configs_have_no_operator_specific_paths():
    pattern = re.compile(r"/(?:data[0-9]+/|home/[A-Za-z_][A-Za-z0-9_-]*/|mnt/shared/)")
    violations = []
    for folder in ("adapters", "evaluation", "vlaworldeval", "cli", "configs"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix in {".py", ".json", ".yaml", ".toml"}:
                if pattern.search(path.read_text()):
                    violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations


def test_registered_environment_defaults_are_project_relative():
    from adapters.environment import _flatten

    registry = tomllib.loads((ROOT / "environment/registry.toml").read_text())
    for key, entry in _flatten(registry["environments"]).items():
        assert not Path(entry["path"]).is_absolute(), key


def test_public_operation_scripts_require_machine_settings():
    for name in ("migrate_server.sh", "finish_remote_relocation.sh"):
        path = ROOT / "scripts/ops" / name
        value = path.read_text()
        assert "MIGRATION_REMOTE:?" in value
        assert "MIGRATION_DEST:?" in value
        assert "MIGRATION_KEY:?" in value
        assert "117.50." not in value and "/data1/liuming" not in value
        result = subprocess.run(["bash", str(path)], capture_output=True, text=True,
                                env={"PATH": "/usr/bin:/bin"}, timeout=10)
        assert result.returncode != 0


def test_compute_deployment_never_uploads_results_or_archives(tmp_path):
    root = tmp_path/"project"
    script = root/"scripts/ops/migrate_server.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes((ROOT/"scripts/ops/migrate_server.sh").read_bytes())
    key = tmp_path/"test-key"
    key.touch()
    commands = tmp_path/"commands.jsonl"
    binaries = tmp_path/"bin"
    binaries.mkdir()
    for name in ("ssh", "rsync"):
        executable = binaries/name
        executable.write_text(f'#!{sys.executable}\n'
            'import json,os,sys\n'
            'with open(os.environ["COMMAND_LOG"], "a") as stream:\n'
            ' stream.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]])+"\\n")\n')
        executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{binaries}:/usr/bin:/bin", "UV_BIN": "/missing-test-uv",
           "MIGRATION_REMOTE": "tester@example.invalid", "MIGRATION_DEST": "/compute/project",
           "MIGRATION_KEY": str(key), "COMMAND_LOG": str(commands)}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    copies = [c for c in calls if c[0] == "rsync"]
    assert len(copies) == 4  # code, full assets, code audit, remaining-delta audit
    assert all("--exclude=/results/" in c and "--exclude=/.archive/" in c for c in copies)
    assert not any("/compute/project/results" in str(c) for c in calls)


def test_tau_server_generates_local_config_without_overwriting_template(tmp_path):
    root = tmp_path / "checkout with spaces"
    launcher = root / "scripts/serve/tau0_wm_simulator.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes((ROOT / "scripts/serve/tau0_wm_simulator.sh").read_bytes())
    source = root / "third_party/tau-0-wm"
    upstream = source / "configs/deployment/tau_simulator.yaml"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("model: /path/to/pretrained/model\nvae: /path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth\n")
    server = source / "web_infer_utils/simulator/server_sim.py"
    server.parent.mkdir(parents=True)
    server.touch()
    base = root / "checkpoints/world_models/tau0_wm_simulator"
    for rel in ("tau-0-wm/simulator/diffusion_pytorch_model.bin", "Wan2.2-TI2V-5B/Wan2.2_VAE.pth"):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    # Stub only captures launcher arguments; no real model or network execution.
    python = root / "environment/world_models/tau0_wm_simulator/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    template = root / "configs/world_models/tau0_wm_simulator.runtime.yaml"
    template.parent.mkdir(parents=True)
    template.write_text("portable-user-template\n")
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    generated = Path(args[args.index("--config") + 1])
    assert generated.is_relative_to(root / ".cache/runtime")
    assert str(base / "tau-0-wm/simulator/diffusion_pytorch_model.bin") in generated.read_text()
    assert template.read_text() == "portable-user-template\n"
    assert upstream.read_text().startswith("model: /path/to/pretrained/model")
