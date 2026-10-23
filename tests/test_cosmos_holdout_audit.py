"""Dataset declarations cannot silently shrink the documented2157-episode holdout."""
import json
import sys
from types import SimpleNamespace

import pytest

from evaluation.planning.checkpoint import audit_cosmos_droid_holdout
from evaluation.datasets.droid import CAMERAS


def fixture(tmp_path, monkeypatch, *, duplicate=False, missing_meta=False):
    from vlaworldeval import paths
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed),
        randperm=lambda n, generator: SimpleNamespace(tolist=lambda: list(range(n)))))
    tables = {}
    for subset, total in (("success", 57639), ("failure", 14268)):
        directory = tmp_path/subset
        meta = directory/"meta/episodes/chunk-000/file-000.parquet"
        meta.parent.mkdir(parents=True)
        meta.write_bytes(b"synthetic metadata")
        (directory/"meta/info.json").write_text(json.dumps({"total_episodes": total, "fps": 15}))
        rows = []
        for index in range(round(total*.03)-(1 if missing_meta else 0)):
            row = {"episode_index": index, "length": 100, "data/chunk_index": 0, "data/file_index": 0}
            for key in CAMERAS.values():
                row.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0})
            rows.append(row)
        if duplicate:
            rows.append(rows[0])
        tables[str(meta)] = rows
    parquet = SimpleNamespace(read_table=lambda path, **kwargs: SimpleNamespace(to_pylist=lambda: tables[str(path)]))
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)


def test_all_heldout_ids_preserved_despite_missing_payload(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    output = tmp_path/"audit.json"
    result = audit_cosmos_droid_holdout(tmp_path, output)
    report = json.loads(output.read_text())
    assert result["expected_episodes"] == len(report["episodes"]) == 2157
    assert result["locally_available_episodes"] == 0
    assert len(report["missing_files"]) == 8
    assert all(e["complete16_action_chunks"] == 6 for e in report["episodes"])
    assert not report["full_test_set_completed"]


@pytest.mark.parametrize("issue", ["duplicate", "missing_meta"])
def test_bad_metadata_does_not_redefine_cohort(tmp_path, monkeypatch, issue):
    fixture(tmp_path, monkeypatch, **{issue: True})
    output = tmp_path/"audit.json"
    with pytest.raises(ValueError):
        audit_cosmos_droid_holdout(tmp_path, output)
    assert not output.exists()
