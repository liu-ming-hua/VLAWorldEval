import json
from pathlib import Path

import pytest

from vlaworldeval.maintenance import archive_cleanup, plan_cleanup


def project(tmp_path):
    (tmp_path / "configs").mkdir()
    policy = json.loads(Path("configs/result_retention.json").read_text())
    policy["retire_before_utc"] = "2100-01-01T00:00:00+00:00"
    (tmp_path / "configs/result_retention.json").write_text(json.dumps(policy))
    return tmp_path


def test_cleanup_preserves_inputs_latest_unknown_and_running_outputs(tmp_path):
    root = project(tmp_path)
    for name in (
        "checkpoints",
        "datasets",
        "environment",
        "results/experiments/new_user_run",
        "results/experiments/long_openvla_20261012",
        "results/experiments/openvla_worldgym_v1/attempt_live",
    ):
        (root / name).mkdir(parents=True)
    (root / "results/experiments/openvla_worldgym_v1/attempt_live/summary.json").write_text(
        '{"status":"running"}'
    )
    old = root / "results/delivery/old_debug.log"
    old.parent.mkdir(parents=True)
    old.write_text("old result")
    plan = plan_cleanup(root)
    assert [c["source"] for c in plan["candidates"]] == ["results/delivery/old_debug.log"]
    assert old.exists() and not (root / ".archive").exists()
    report = archive_cleanup(root, plan)
    assert report["status"] == "completed" and not old.exists()
    assert Path(report["moves"][0]["archived"]).read_text() == "old result"


@pytest.mark.parametrize(
    "target", ["checkpoints", "environment", "../outside", "/tmp/outside", "results"]
)
def test_cleanup_rejects_broad_or_protected_targets(tmp_path, target):
    plan = {"root": str(tmp_path), "mode": "archive_not_delete", "candidates": [{"source": target}]}
    with pytest.raises(ValueError, match="target"):
        archive_cleanup(tmp_path, plan)
    assert not (tmp_path / ".archive").exists()


def test_cleanup_rejects_overlapping_and_symlink_targets(tmp_path):
    root = project(tmp_path)
    path = root / "results/delivery/old"
    path.mkdir(parents=True)
    plan = {
        "root": str(root),
        "mode": "archive_not_delete",
        "candidates": [
            {"source": "results/delivery/old"},
            {"source": "results/delivery/old/child"},
        ],
    }
    with pytest.raises(ValueError, match="overlap"):
        archive_cleanup(root, plan)
    (root / "results/delivery/link").symlink_to(path)
    plan["candidates"] = [{"source": "results/delivery/link"}]
    with pytest.raises(ValueError, match="symlink"):
        archive_cleanup(root, plan)


def test_future_delivery_and_new_model_sources_are_preserved(tmp_path):
    import os

    root = project(tmp_path)
    file = root / "results/delivery/new_user_report.json"
    file.parent.mkdir(parents=True)
    file.write_text("{}")
    os.utime(file, (4200000000, 4200000000))
    source = root / "models/custom.py"
    source.parent.mkdir()
    source.write_text("# user source")
    assert not plan_cleanup(root)["candidates"]
