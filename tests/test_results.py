import json

from evaluation.reporting.results import collect_results
from evaluation.execution.transport import write_packet


def test_attempt_inventory_distinguishes_running_failed_and_invalid(tmp_path):
    for name, status in (("attempt_a", "completed"), ("attempt_b", "running"), ("attempt_c", "failed")):
        folder = tmp_path / "suite" / name
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"status": status, "jobs": [], "worker_loads": [1, 2],
                                                        "error_info": {"code": "request_timeout"} if status == "failed" else None,
                                                        "worker_cleanup": []}))
    broken = tmp_path / "attempt_bad"
    broken.mkdir()
    (broken / "summary.json").write_text("{")
    report = collect_results(tmp_path)
    assert report["status_counts"] == {"running": 1, "completed": 1, "failed": 1}
    assert len(report["errors"]) == 1
    assert report["attempts"][0]["model_loads"] == 2
    failed = next(item for item in report["attempts"] if item["status"] == "failed")
    assert failed["error_info"]["code"] == "request_timeout"


def test_inventory_counts_resident_workers_before_attempt_finishes(tmp_path):
    attempt = tmp_path / "attempt_running"
    worker = attempt / "workers" / "vla"
    worker.mkdir(parents=True)
    (attempt / "summary.json").write_text(json.dumps({"status": "running", "jobs": []}))
    write_packet(worker / "ready.npz", {"ok": True})
    assert collect_results(tmp_path)["attempts"][0]["model_loads"] == 1
