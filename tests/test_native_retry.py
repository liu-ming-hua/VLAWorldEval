"""Explicit retry preserves failed evidence, input contracts and denominator."""
import json

import pytest

from evaluation.planning.native_reference import prepare_reference_retry, prepare_reference_repartition
from evaluation.reporting.task_scores import _digest


def source(tmp_path):
    directory = tmp_path/"old"
    directory.mkdir()
    jobs = []
    for name in ("done", "failed", "not_started"):
        job = {"id": name, "chunks": 1}
        for field in ("input", "config", "reference"):
            path = directory/f"{name}_{field}"
            path.write_bytes(b"frozen fixture")
            job[field], job[field+"_sha256"] = str(path), _digest(path)
        jobs.append(job)
    study = directory/"study.json"
    study.write_text(json.dumps({"schema_version": 1, "jobs": jobs, "plan": {"allowed_gpus": [0]}}))
    completed = directory/"runs/done/summary.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps({"cleanup": {"returncode": 0, "signals": [], "errors": []}}))
    progress = {"status": "failed", "study_sha256": _digest(study), "jobs": {
        "done": {"status": "completed", "report_sha256": _digest(completed)},
        "failed": {"status": "failed", "error": "original evidence must remain"}}}
    (directory/"progress.json").write_text(json.dumps(progress))
    return directory


def test_retry_has_only_pending_exact_inputs_and_keeps_original_denominator(tmp_path):
    directory = source(tmp_path)
    original = (directory/"progress.json").read_bytes()
    result = prepare_reference_retry(tmp_path, directory, tmp_path/"retry")
    assert [j["id"] for j in result["jobs"]] == ["failed", "not_started"]
    assert result["original_expected_cases"] == 3 and result["skipped_completed_cases"] == 1
    assert (directory/"progress.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare_reference_retry(tmp_path, directory, tmp_path/"retry")


@pytest.mark.parametrize("change", ["input", "summary", "running"])
def test_retry_rejects_tampered_or_live_study(tmp_path, change):
    directory = source(tmp_path)
    if change == "input":
        (directory/"failed_input").write_bytes(b"changed")
    elif change == "summary":
        (directory/"runs/done/summary.json").write_text("{}")
    else:
        path = directory/"progress.json"
        value = json.loads(path.read_text())
        value["status"] = "running"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        prepare_reference_retry(tmp_path, directory, tmp_path/"retry")
    assert not (tmp_path/"retry").exists()


def repartition_source(tmp_path):
    directory = source(tmp_path)
    path = directory/"study.json"
    study = json.loads(path.read_text())
    env = directory/"environments.json"
    env.write_text('{"schema_version":1,"paths":{}}')
    study.update(environments=str(env), environments_sha256=_digest(env))
    for job in study["jobs"]:
        config = directory/f"{job['id']}_config"
        config.write_text('{"gpus":[0],"memory_gib":12}')
        job["config_sha256"] = _digest(config)
    path.write_text(json.dumps(study))
    progress = json.loads((directory/"progress.json").read_text())
    progress["study_sha256"] = _digest(path)
    (directory/"progress.json").write_text(json.dumps(progress))
    return directory


def test_repartition_exact_pending_union_and_preserves_original_evidence(tmp_path):
    directory = repartition_source(tmp_path)
    before = {name: (directory/name).read_bytes() for name in ("study.json", "progress.json")}
    out = tmp_path/"balanced"
    result = prepare_reference_repartition(tmp_path, [directory], out, [0, 3, 7])
    assert result["pending_cases"] == 2 and result["skipped_completed_cases"] == 1
    jobs = []
    for shard in result["shards"]:
        study = json.loads((out/f"gpu{shard['gpu']}"/"study.json").read_text())
        assert study["runtime_gpu_override_required"] == shard["gpu"]
        assert study["plan"]["allowed_gpus"] == [0]
        jobs.extend(study["jobs"])
    assert {job["id"] for job in jobs} == {"failed", "not_started"}
    assert len(jobs) == 2 and all(json.loads(open(j["config"]).read())["gpus"] == [0] for j in jobs)
    assert before == {name: (directory/name).read_bytes() for name in before}


@pytest.mark.parametrize("fault", ["live", "mutated", "overlap", "distributed", "invalid_gpu"])
def test_repartition_refuses_unsafe_sources_before_creating_outputs(tmp_path, fault):
    directory = repartition_source(tmp_path)
    sources, gpus = [directory], [7]
    if fault == "live":
        p = directory/"progress.json"
        v = json.loads(p.read_text())
        v["status"] = "running"
        p.write_text(json.dumps(v))
    elif fault == "mutated":
        (directory/"failed_input").write_bytes(b"changed")
    elif fault == "overlap":
        sources *= 2
    elif fault == "distributed":
        p = directory/"not_started_config"
        p.write_text('{"gpus":[0,1]}')
        study_path = directory/"study.json"
        study = json.loads(study_path.read_text())
        study["jobs"][-1]["config_sha256"] = _digest(p)
        study_path.write_text(json.dumps(study))
        p = directory/"progress.json"
        progress = json.loads(p.read_text())
        progress["study_sha256"] = _digest(study_path)
        p.write_text(json.dumps(progress))
    else:
        gpus = [True]
    with pytest.raises(ValueError):
        prepare_reference_repartition(tmp_path, sources, tmp_path/"new", gpus)
    assert not (tmp_path/"new").exists()
