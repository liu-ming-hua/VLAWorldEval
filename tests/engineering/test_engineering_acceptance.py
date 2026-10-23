import json
import sys

import pytest

from evaluation.validation.engineering import run_engineering_acceptance

pytestmark = pytest.mark.process


def checkout(tmp_path):
    root = tmp_path / "checkout"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text("# fixture")
    (root / "scripts").mkdir()
    return root


def test_engineering_acceptance_runs_fresh_and_records_failures(tmp_path, monkeypatch):
    import evaluation.validation.engineering as module

    monkeypatch.setattr(
        module,
        "acceptance_commands",
        lambda python: {
            "pass": [sys.executable, "-c", "print('actual execution')"],
            "fail": [sys.executable, "-c", "raise SystemExit(3)"],
        },
    )
    report = run_engineering_acceptance(checkout(tmp_path), tmp_path / "output")
    assert report["status"] == "failed"
    assert report["checks"][0]["ok"] and report["checks"][1]["returncode"] == 3
    assert not report["scientific_quality_verified"]
    assert json.loads((tmp_path / "output/report.json").read_text())["status"] == "failed"


def test_engineering_acceptance_refuses_invalid_timeout(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="timeout"):
        run_engineering_acceptance(checkout(tmp_path), tmp_path / "output", timeout=float("nan"))


def test_shell_gate_checks_every_script_not_only_first(tmp_path, monkeypatch):
    import evaluation.validation.engineering as module

    root = checkout(tmp_path)
    (root / "scripts/a.sh").write_text("true\n")
    (root / "scripts/z.sh").write_text("if true; then\n")
    monkeypatch.setattr(module, "acceptance_commands", lambda python: {})
    report = run_engineering_acceptance(root, tmp_path / "output")
    assert report["status"] == "failed"
    assert report["checks"][0]["name"] == "shell_syntax" and not report["checks"][0]["ok"]


def test_check_timeout_is_failed_and_reaped(tmp_path, monkeypatch):
    import evaluation.validation.engineering as module

    monkeypatch.setattr(
        module,
        "acceptance_commands",
        lambda python: {"slow": [sys.executable, "-c", "import time; time.sleep(60)"]},
    )
    report = run_engineering_acceptance(checkout(tmp_path), tmp_path / "output", timeout=0.1)
    assert report["status"] == "failed"
    assert report["checks"][0]["error"] == "timeout"


def test_inventory_declarations_do_not_require_downloads(tmp_path):
    import pytest

    from evaluation.planning.inventory import validate_datasets

    path = tmp_path / "datasets/example/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"local_path": "datasets/example/official/demo"}))
    assert validate_datasets(tmp_path)[0]["status"] == "missing_assets"
    with pytest.raises(ValueError, match="Missing or empty"):
        validate_datasets(tmp_path, require_assets=True)


def test_vendored_runtime_does_not_require_original_git_checkout(tmp_path):
    from evaluation.execution.executor import git_checkout_evidence

    evidence = git_checkout_evidence(tmp_path / "missing-original-repository")
    assert evidence["head"] is None and evidence["status"] == "not_a_git_checkout"
    source = tmp_path / "exported-source"
    source.mkdir()
    (source / "network.py").write_text("# exported network")
    assert git_checkout_evidence(source)["head"] is None
