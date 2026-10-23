"""Diagnostics use shared admission but never start a model or download assets."""

import json
import subprocess
from pathlib import Path

import pytest

from cli.main import main
from evaluation.execution.admission import ResourceAdmissionError, resource_admission, select_jobs
from evaluation.planning.configuration import ConfigurationError, load_suite
from evaluation.planning.diagnostics import diagnose_suite
from evaluation.planning.experiments import build_experiment_plan

pytestmark = pytest.mark.process

CONFIG = Path("configs/experiments/openvla_worldgym_v1.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime", []),
        ("suite_id", 2),
        ("sampling", None),
        ("experiments", [None]),
        ("dataset", "bad"),
    ],
)
def test_wrong_shape_has_file_and_field(tmp_path, field, value):
    config = json.loads(CONFIG.read_text())
    config[field] = value
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ConfigurationError) as caught:
        build_experiment_plan(path)
    assert caught.value.details["file"] == str(path)
    assert any(issue["field"].startswith(field) for issue in caught.value.details["issues"])


def test_missing_and_unknown_fields_are_aggregated(tmp_path):
    config = json.loads(CONFIG.read_text())
    del config["runtime"]["world_model_checkpoint"]
    config["runtime"]["vla_memory_gibb"] = 10
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ConfigurationError) as caught:
        load_suite(path)
    assert {issue["field"] for issue in caught.value.details["issues"]} == {
        "runtime.world_model_checkpoint",
        "runtime.vla_memory_gibb",
    }


def test_json_syntax_and_missing_file_have_clear_errors(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text("{")
    with pytest.raises(ConfigurationError, match="line 1, column 2"):
        load_suite(path)
    with pytest.raises(ConfigurationError, match="No such file"):
        load_suite(tmp_path / "absent.json")


@pytest.mark.parametrize("field,value", [("fps", 5), ("start_frame", -1), ("tail_policy", "pad")])
def test_semantic_errors_locate_exact_field(tmp_path, field, value):
    config = json.loads(CONFIG.read_text())
    config["dataset"][field] = value
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ConfigurationError) as caught:
        build_experiment_plan(path)
    assert caught.value.details["issues"][0]["field"] == f"dataset.{field}"


def test_plan_cli_returns_structured_error_without_traceback(tmp_path, capsys):
    config = json.loads(CONFIG.read_text())
    del config["sampling"]["guidance"]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    assert main(["plan", "--config", str(path)]) == 2
    output = capsys.readouterr()
    error = json.loads(output.err)["error"]
    assert error["issues"][0]["field"] == "sampling.guidance"
    assert not output.out and "Traceback" not in output.err
    with pytest.raises(ConfigurationError):
        main(["--debug", "plan", "--config", str(path)])


def test_local_doctor_never_spawns_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Local diagnosis must not launch any process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    report = diagnose_suite(CONFIG, experiments=["loop_smoke"])
    assert report["model_loading_tested"] is False
    assert report["execution_ready"] is False
    assert {item["name"] for item in report["checks"] if item["status"] == "skipped"} == {
        "data",
        "gpu",
    }


def test_gpu_failure_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(
        "evaluation.planning.diagnostics.resource_admission",
        lambda *args: (_ for _ in ()).throw(ResourceAdmissionError("insufficient free memory")),
    )
    report = diagnose_suite(CONFIG, check_gpu=True)
    assert not report["ok"]
    assert next(item for item in report["checks"] if item["name"] == "gpu")["status"] == "failed"


def test_data_subprocess_failure_is_reported(monkeypatch):
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffprobe"])

    monkeypatch.setattr("evaluation.planning.diagnostics.admit_episodes", failure)
    report = diagnose_suite(CONFIG, check_data=True)
    assert not report["ok"]
    assert next(item for item in report["checks"] if item["name"] == "data")["status"] == "failed"


def test_unknown_seed_is_not_silently_ignored():
    plan = build_experiment_plan("configs/experiments/droid_framework_v1.json")
    with pytest.raises(ValueError, match="--seed"):
        select_jobs(plan, ["loop_integrated_smoke"], [0, 1])


def test_shared_gpu_admission_adds_resident_budgets(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "0, 40000, 81920\n")
    runtime = {
        "vla_gpus": [0],
        "world_model_gpus": [0],
        "max_process_memory_gib_per_gpu": 20,
        "vla_memory_gib": 8,
    }
    report = resource_admission(runtime, True)
    assert report["budget_gib"][0] == 32
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "0, 32000, 81920\n")
    with pytest.raises(ResourceAdmissionError, match="32.5 GiB"):
        resource_admission(runtime, True)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 10
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(subprocess, "check_output", timeout)
    with pytest.raises(ResourceAdmissionError, match="GPU query failed"):
        resource_admission(runtime, True)


def test_doctor_bad_config_has_json_and_nonzero_exit(tmp_path, capsys):
    path = tmp_path / "suite.json"
    path.write_text("[]")
    assert main(["doctor", "--config", str(path), "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert not report["ok"] and not report["model_loading_tested"]
    assert report["coverage"]["data"] == report["coverage"]["gpu"] == "skipped"
