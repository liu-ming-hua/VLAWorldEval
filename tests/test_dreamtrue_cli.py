"""Weight-free CLI regressions, including hosts without nvidia-smi."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dreamtrue_replay_cli", Path(__file__).resolve().parents[1] / "scripts/replay/dreamtrue.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def cpu_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "file_record", lambda path: {"sha256": "fixture", "size": 0})

    def no_nvidia(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not installed")

    monkeypatch.setattr(runner.subprocess, "check_output", no_nvidia)
    return tmp_path


def test_prepare_never_queries_gpu_or_starts_worker(cpu_cli, monkeypatch):
    import evaluation.datasets.dreamtrue as dataset

    packet = {"condition": {"source": {"kind": "cpu_fixture"}, "n_real_frames": 44}}
    monkeypatch.setattr(dataset, "prepare_official", lambda **kwargs: packet)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU preparation must not query GPU or start ModelWorker")

    monkeypatch.setattr(runner, "gpu_status", forbidden)
    monkeypatch.setattr(runner, "ModelWorker", forbidden)
    output = cpu_cli / "prepare"
    assert runner.main(["prepare", "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "prepared"
    assert report["gpu"] is None
    assert not report["model_inference_performed"]
    assert runner.read_packet(output / "native_input.npz") == packet


def test_check_records_missing_nvidia_without_failing(cpu_cli, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="CPU diagnostic fixture\n", stderr="")

    class Worker:
        ready = {"cpu_fixture": True}

        def __init__(self, *args, **kwargs):
            assert kwargs["gpus"] == []

        def call(self, *args, **kwargs):
            return None

        def close(self):
            return {"closed": True}

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "ModelWorker", Worker)
    output = cpu_cli / "check"
    assert runner.main(["check", "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["gpu"] is None
    assert report["gpu_diagnostic"]["error_type"] == "FileNotFoundError"
    assert report["status"] == "checks_passed_with_blockers"
    probe = next(command[-1] for command in commands if "available=torch.cuda.is_available()" in command[-1])
    assert "if available else None" in probe
    assert not report["model_inference_performed"]


@pytest.mark.parametrize("operation", ["smoke", "camera-probe"])
def test_gpu_operations_still_require_gpu(cpu_cli, operation):
    output = cpu_cli / operation
    assert runner.main([operation, "--output", str(output)]) == 2
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "blocked"
    assert report["error_type"] == "FileNotFoundError"
    assert not report["model_inference_performed"]


def test_help_never_queries_gpu(cpu_cli, capsys):
    with pytest.raises(SystemExit) as result:
        runner.main(["--help"])
    assert result.value.code == 0
    assert "prepare" in capsys.readouterr().out
