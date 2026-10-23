"""Version-local evaluator evidence; no writes to model inference artifacts."""
import json
from pathlib import Path

from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest


def admit_metric(path, *, summary_sha256, reference_sha256, implementation):
    value = json.loads(Path(path).read_text())
    if (not isinstance(value, dict) or value.get("status") != "completed" or value.get("summary_sha256") != summary_sha256
            or value.get("reference_sha256") != reference_sha256):
        raise ValueError("Metric artifact does not bind the same completed replay/reference")
    if not isinstance(implementation, dict) or not implementation or value.get("implementation_sha256") != implementation:
        raise ValueError("Metric artifact implementation differs from its scoring journal")
    return {"path": str(Path(path).resolve()), "sha256": _digest(path)}


def load_metric_store(root, directory, plan_path, plan, selected):
    """Admit a completed journal without requiring today's implementation hash.

    An old, explicit, checksum-bound metric version can be reported later. A
    changed journal, cross-plan artifact or unversioned sidecar cannot substitute
    for that version. Adjacent legacy files are never used as a fallback.
    """
    directory = project_path(root, str(directory))
    path = directory/"progress.json"
    digest = _digest(path)
    journal = json.loads(path.read_text())
    if not isinstance(journal, dict):
        raise ValueError("Metric journal must be an object")
    names = {case["id"] for case in plan["cases"]}
    provider = journal.get("provider")
    if (not isinstance(journal.get("jobs"), dict)
            or not isinstance(journal.get("case_input_sha256"), dict)
            or journal.get("sidecar_layout") != "version_local"
            or journal.get("status") != "completed"
            or journal.get("source_plan_sha256") != _digest(plan_path)
            or journal.get("expected_cases") != len(names)
            or set(journal.get("jobs", {})) != names or set(selected) != names
            or provider not in {"none", "pusht_color", "ewmbench"}):
        raise ValueError("Metric store requires a completed, exact-plan version-local journal")
    required = {"lpips"} if provider == "none" else {"lpips", "motion"}
    implementations = journal.get("metric_implementation_sha256", {})
    if not isinstance(implementations, dict) or set(implementations) != required:
        raise ValueError("Metric journal is missing declared implementation versions")
    index = {}
    for case in plan["cases"]:
        name = case["id"]
        summary, reference = selected[name]
        report = json.loads(summary.read_text())
        expected = {"input_sha256": report["input_sha256"],
                    "config_sha256": _digest(project_path(root, case["prepared"])/"config.json"),
                    "reference_sha256": report["reference_evaluation"]["reference_sha256"]}
        job = journal["jobs"][name]
        summary_hash = _digest(summary)
        if (not isinstance(job, dict) or job.get("status") != "completed" or job.get("summary_sha256") != summary_hash
                or project_path(root, job.get("summary")) != summary.resolve()
                or journal.get("case_input_sha256", {}).get(name) != expected
                or set(job.get("metrics", {})) != required):
            raise ValueError("Metric journal job does not bind the admitted replay and prepared inputs")
        index[name] = {}
        for kind, record in job["metrics"].items():
            if not isinstance(record, dict):
                raise ValueError("Metric artifact record must include path and checksum")
            artifact = project_path(root, record.get("path"))
            if not artifact.is_relative_to(directory) or artifact == directory or _digest(artifact) != record["sha256"]:
                raise ValueError("Metric artifact changed or leaves its version directory")
            admit_metric(artifact, summary_sha256=summary_hash,
                         reference_sha256=expected["reference_sha256"], implementation=implementations[kind])
            index[name][kind] = artifact
    if _digest(path) != digest:
        raise ValueError("Metric journal changed during admission")
    return index, {"path": str(path), "sha256": digest, "provider": provider}
