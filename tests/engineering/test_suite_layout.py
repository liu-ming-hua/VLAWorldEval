"""Cost tiers must select real tests and never import optional integrations implicitly."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.checks.test_suite import inventory

pytestmark = pytest.mark.process
ROOT = Path(__file__).resolve().parents[2]


def miniature_suite(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_bytes((ROOT / "tests/conftest.py").read_bytes())
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\naddopts = --strict-markers\n"
        "markers =\n unit: CPU\n process: subprocess\n integration: opt-in\n"
    )
    for group in ("core", "engineering", "integration"):
        (tests / group).mkdir()
    (tests / "core/test_unit.py").write_text("def test_contract():\n    assert True\n")
    (tests / "engineering/test_process.py").write_text(
        "import pytest\npytestmark = pytest.mark.process\ndef test_process():\n    assert True\n"
    )
    optional = tests / "integration/test_native.py"
    optional.write_text("raise RuntimeError('optional module must not be imported')\n")
    return optional


def run_suite(tmp_path, *arguments):
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    output = tmp_path / "suite.json"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--suite-report", str(output), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result, json.loads(output.read_text()) if output.exists() else None


def test_default_suite_includes_both_cpu_tiers_without_importing_native_module(tmp_path):
    miniature_suite(tmp_path)
    result, report = run_suite(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {case["tier"] for case in report["cases"]} == {"unit", "process"}
    assert len(report["cases"]) == 2 and not report["integration_enabled"]
    assert all(
        case["phases"]["call"] == "passed" and case["duration_seconds"] >= 0
        for case in report["cases"]
    )


@pytest.mark.parametrize("tier", ["unit", "process"])
def test_cost_selector_applies_before_pytest_marker_filtering(tmp_path, tier):
    miniature_suite(tmp_path)
    result, report = run_suite(tmp_path, "-m", tier)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(report["cases"]) == 1 and report["cases"][0]["tier"] == tier


def test_integration_collection_and_execution_require_explicit_opt_in(tmp_path):
    optional = miniature_suite(tmp_path)
    result, _ = run_suite(tmp_path, "--run-integration", "-m", "integration")
    assert result.returncode == 2 and "optional module must not be imported" in result.stdout
    optional.write_text("def test_native_fixture():\n    assert True\n")
    result, report = run_suite(tmp_path, "--run-integration", "-m", "integration")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(report["cases"]) == 1 and report["cases"][0]["tier"] == "integration"
    assert report["integration_enabled"]


def test_conflicting_cost_annotations_fail_instead_of_running_in_two_jobs(tmp_path):
    miniature_suite(tmp_path)
    (tmp_path / "tests/core/test_unit.py").write_text(
        "import pytest\npytestmark = [pytest.mark.unit, pytest.mark.process]\n"
        "def test_contract():\n    assert True\n"
    )
    result, _ = run_suite(tmp_path)
    assert result.returncode == 4 and "Exactly one cost tier" in result.stderr


def test_inventory_maps_every_module_to_a_group_and_cost_without_claiming_coverage():
    report = inventory(ROOT)
    assert set(report["groups"]) == {"core", "models", "engineering", "integration"}
    assert set(report["tiers"]) == {"unit", "process", "integration"}
    assert sum(report["groups"].values()) == report["modules"]
    assert report["exact_duplicate_function_bodies"] == []
    assert "not measured" in report["scope"]
    assert any(
        "evaluation.evaluator" in record["production_imports"] for record in report["records"]
    )


def test_shared_builders_are_not_imported_from_collected_test_modules():
    import ast

    for path in (ROOT / "tests").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("tests.")
            ):
                assert ".test_" not in node.module, str(path)
