"""One public entry point: preflight, existing executor, artifact inspection."""
from pathlib import Path
from contextlib import redirect_stdout
import sys

from evaluation.planning.diagnostics import diagnose_suite
from vlaworldeval.paths import project_root


def evaluate_suite(config, *, experiments=(), seeds=(), execute=False, on_attempt=None):
    config = Path(config)
    if not config.is_absolute():
        config = project_root() / config
    report = {"schema_version": 1, "status": "preflight", "execution_performed": False,
              "scientific_quality_verified": False, "artifacts": []}
    report["preflight"] = diagnose_suite(config, experiments=experiments, seeds=seeds,
                                         check_data=True, check_gpu=execute)
    if not report["preflight"]["ok"]:
        report.update(ok=False, status="preflight_failed")
        return report
    if not execute:
        report.update(ok=True, status="checked_not_executed")
        return report
    from evaluation.execution.executor import execute_suite
    from evaluation.reporting.artifact_checks import audit_job, audit_attempt
    with redirect_stdout(sys.stderr):
        kwargs = {"on_attempt": on_attempt} if on_attempt is not None else {}
        summary = execute_suite(config, experiments=experiments, seeds=seeds, **kwargs)
    report.update(execution_performed=True, execution=summary, status=summary["status"])
    for job in summary["jobs"]:
        path = Path(job["output"])
        report["artifacts"].append({"job": job["id"], "directory": str(path),
                                    "audit": audit_job(job),
                                    "videos": sorted(str(p) for p in path.glob("*.mp4"))})
    report["audit"] = audit_attempt(summary)
    report["ok"] = report["audit"]["ok"]
    if summary["status"] == "completed" and not report["ok"]:
        report["status"] = "artifact_audit_failed"
    return report
