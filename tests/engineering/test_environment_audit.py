import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.validation.environment_audit import audit_environments, compare_environment

pytestmark = pytest.mark.process


def test_environment_drift_catches_packages_python_and_sources():
    baseline = {"python": [3, 11, 2], "packages": {"a": "1", "b": "2"}, "source_requirements": []}
    current = {
        "python": [3, 11, 3],
        "packages": {"a": "2", "c": "1"},
        "source_requirements": [{"package": "a", "commit": "changed"}],
    }
    drift = compare_environment(current, baseline)
    assert drift["added"] == ["c"] and drift["removed"] == ["b"]
    assert drift["changed"]["a"] == {"expected": "1", "actual": "2"}
    assert drift["python_changed"] and drift["sources_changed"]
    assert not any(compare_environment(baseline, baseline).values())


def test_environment_probe_is_stdlib_only(monkeypatch, tmp_path):
    import evaluation.validation.environment_audit as module

    monkeypatch.setattr(
        module,
        "load_environment_registry",
        lambda: {
            "test": SimpleNamespace(
                path=Path(sys.executable).parent,
                python=Path(sys.executable),
                package_profile="test",
                exists=True,
            )
        },
    )
    report = audit_environments(keys=["test"])
    assert report["ok"] and not report["portable_lock"] and not report["fresh_install_verified"]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report))
    assert audit_environments(keys=["test"], baseline=baseline)["ok"]
    with pytest.raises(ValueError, match="Unknown environments"):
        audit_environments(keys=["other"])
