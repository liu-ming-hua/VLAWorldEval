"""Explicit test costs; optional native checks are never enabled by local assets."""

import json
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parent
TIERS = ("unit", "process", "integration")


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Collect optional native-runtime/data checks; does not install or download assets",
    )
    parser.addoption(
        "--suite-report",
        type=Path,
        default=None,
        help="Write selected cases, test tiers and observed durations as JSON",
    )


def pytest_ignore_collect(collection_path, config):
    relative = (
        Path(collection_path).relative_to(TEST_ROOT)
        if Path(collection_path).is_relative_to(TEST_ROOT)
        else None
    )
    if (
        relative
        and relative.parts[0] == "integration"
        and not config.getoption("--run-integration")
    ):
        return True
    return None


class SuiteReport:
    def __init__(self, config):
        self.config = config
        self.cases = []
        self.results = {}
        self.collection = []

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection.append(
                {"nodeid": report.nodeid, "outcome": report.outcome, "reason": str(report.longrepr)}
            )

    def pytest_runtest_logreport(self, report):
        record = self.results.setdefault(report.nodeid, {"duration_seconds": 0.0, "phases": {}})
        record["duration_seconds"] += report.duration
        record["phases"][report.when] = report.outcome

    def pytest_collection_finish(self, session):
        self.cases = [
            {
                "nodeid": item.nodeid,
                "group": Path(item.path).relative_to(TEST_ROOT).parts[0],
                "tier": next(tier for tier in TIERS if item.get_closest_marker(tier)),
            }
            for item in session.items
        ]

    def pytest_sessionfinish(self, session, exitstatus):
        output = self.config.getoption("--suite-report")
        if output is None:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": 1,
            "exit_status": int(exitstatus),
            "integration_enabled": self.config.getoption("--run-integration"),
            "cases": [{**case, **self.results.get(case["nodeid"], {})} for case in self.cases],
            "collection_events": self.collection,
            "scope": "Test selection and observed cost, not executed production-line coverage or model quality",
        }
        output.write_text(json.dumps(value, indent=2) + "\n")


def pytest_configure(config):
    config.pluginmanager.register(SuiteReport(config), "suite-report")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        relative = Path(item.path).relative_to(TEST_ROOT)
        group = relative.parts[0]
        if group == "integration":
            item.add_marker(pytest.mark.integration)
        explicit = [tier for tier in TIERS if item.get_closest_marker(tier)]
        if not explicit:
            item.add_marker(pytest.mark.unit)
            explicit = ["unit"]
        if len(explicit) != 1:
            raise pytest.UsageError(f"Exactly one cost tier is required: {item.nodeid}: {explicit}")
        if explicit[0] == "integration" and not config.getoption("--run-integration"):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
