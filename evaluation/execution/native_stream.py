"""One resident WM across immutable, incrementally delivered native studies."""
import json
from pathlib import Path
import time

from evaluation.execution.admission import native_gpu_lease
from evaluation.execution.native_study import run_native_study
from evaluation.execution.worker_session import NativeWorkerSession
from evaluation.planning.native_batches import validate_ready_batch
from evaluation.planning.native_reference import prepare_reference_study
from evaluation.planning.registry import project_path
from evaluation.reporting.checkpoint_metrics import select_replays
from evaluation.reporting.task_scores import _digest
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json


def run_native_stream(root, plan_path, batches, output, *, gpus, previous_directory=None):
    root = Path(root).resolve()
    plan_path, batches, output = [project_path(root, str(p)) for p in (plan_path, batches, output)]
    if gpus != [0, 1, 2, 3]:
        raise ValueError("Current resident native stream requires explicit physical GPUs0–3")
    cohort_digest = _digest(plan_path)
    cohort = json.loads(plan_path.read_text())
    identifiers = {c["id"] for c in cohort["cases"]}
    if len(identifiers) != len(cohort["cases"]) or not identifiers:
        raise ValueError("Unique frozen cohort required")
    if output.exists():
        raise FileExistsError("Resident stream needs a fresh output version; old attempts remain intact")
    output.mkdir(parents=True)
    skipped = {}
    if previous_directory is not None:
        selected, _ = select_replays(root, previous_directory, cohort)
        for identifier, (path, _) in selected.items():
            report = json.loads(path.read_text())
            if any(_digest(path.parent/name) != sha for name, sha in report["files"].items()):
                raise ValueError("Existing completed replay artifacts changed")
            skipped[identifier] = {"summary": str(path), "summary_sha256": _digest(path)}
    publish_json_exclusive(output/"stream.json", {
        "schema_version": 1, "cohort_plan": str(plan_path), "cohort_plan_sha256": cohort_digest,
        "expected_cases": len(identifiers), "previous_completed": skipped, "gpus": gpus,
        "completion_policy": "generated cases certified only after resident worker exits cleanly"})
    progress = {"status": "running", "expected_cases": len(identifiers), "previous_completed": len(skipped),
                "generated_in_session": 0, "studies": [], "model_loads": 0}
    journal = output/"progress.json"
    publish_json_exclusive(journal, progress)
    generated = set()
    consumed = set()

    def generated_case(path):
        if path.parent.name not in identifiers or path.parent.name in skipped:
            raise ValueError("Resident worker produced a case outside pending ownership")
        generated.add(path.parent.name)
        progress.update(generated_in_session=len(generated), model_loads=1, last_generated_case=path.parent.name)
        replace_owned_json(journal, progress)

    try:
        with asset_destination_lock(root, output), native_gpu_lease(root, gpus):
            with NativeWorkerSession(output/"resident_session", stop_path=output/"stop_after_case",
                                     on_generated=generated_case) as session:
                while len(generated | set(skipped)) < len(identifiers):
                    if _digest(plan_path) != cohort_digest:
                        raise ValueError("Frozen cohort changed during resident stream")
                    if (output/"stop_after_case").exists():
                        break
                    available = sorted(batches.glob("batch*/prepared_manifest.json"))
                    found = False
                    for manifest in available:
                        if manifest.parent.name in consumed:
                            continue
                        found = True
                        source = manifest.parent/"plan.json"
                        validate_ready_batch(root, source, manifest)
                        evidence = json.loads(manifest.read_text())
                        if evidence["cohort_plan_sha256"] != cohort_digest:
                            raise ValueError("Delivered batch belongs to another cohort")
                        data = json.loads(source.read_text())
                        incoming = {c["id"] for c in data["cases"]}
                        if generated & incoming:
                            raise ValueError("Overlapping delivered batches would duplicate model inference")
                        cases = [c for c in data["cases"] if c["id"] not in skipped]
                        if cases:
                            plans = output/"plans"
                            plans.mkdir(exist_ok=True)
                            derived = plans/(manifest.parent.name+".json")
                            publish_json_exclusive(derived, {**data, "cases": cases,
                                "scope": "Resident stream: missing cases only; completed prefix retained separately"})
                            directory = output/"studies"/manifest.parent.name
                            prepare_reference_study(root, derived, directory)
                            progress["studies"].append(str(directory))
                            replace_owned_json(journal, progress)
                            result = run_native_study(root, directory, watch=True, gpu_map=gpus,
                                                       worker_session=session)
                            if result["status"] != "generated_pending_worker_close":
                                raise ValueError("Resident batch did not finish generation")
                            state = json.loads((directory/"progress.json").read_text())
                            generated.update(i for i, j in state["jobs"].items()
                                             if j["status"] == "generated_pending_worker_close")
                        consumed.add(manifest.parent.name)
                        progress.update(generated_in_session=len(generated), model_loads=int(session.worker is not None))
                        replace_owned_json(journal, progress)
                        if (output/"stop_after_case").exists():
                            break
                    if not found:
                        progress["status"] = "waiting_for_prepared_batch_model_resident"
                        replace_owned_json(journal, progress)
                        time.sleep(15)
                    else:
                        progress["status"] = "running"
            if not session.clean:
                raise ValueError("Resident session did not exit cleanly; results remain uncertified")
            progress["status"] = "completed" if (generated | set(skipped)) == identifiers else "stopped_partial"
            progress["certified_completed"] = len(generated)+len(skipped)
            replace_owned_json(journal, progress)
    except BaseException as exc:
        progress.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        replace_owned_json(journal, progress)
        raise
    return progress
