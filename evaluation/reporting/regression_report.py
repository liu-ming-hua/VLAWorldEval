"""Re-audit executed workflows and summarize engineering/visual evidence separately."""
import json
import math
from pathlib import Path

from evaluation.reporting.artifact_checks import audit_attempt
from evaluation.planning.registry import project_path
from evaluation.reporting.video_diagnostics import diagnose_artifact


def summarize_regressions(root: Path, reports):
    if not reports:
        raise ValueError("At least one executed workflow report is required")
    root = Path(root).resolve()
    result = {"schema_version": 1, "kind": "closed_loop_regression",
              "task_success_evaluated": False, "scientific_quality_verified": False,
              "entries": [], "errors": []}
    seen = set()
    for raw in reports:
        path = project_path(root, str(raw))
        if path in seen:
            raise ValueError(f"Duplicate regression report: {path}")
        seen.add(path)
        entry = {"report": str(path), "ok": False, "jobs": []}
        result["entries"].append(entry)
        try:
            value = json.loads(path.read_text())
            if (not isinstance(value, dict) or value.get("schema_version") != 1
                    or value.get("execution_performed") is not True
                    or value.get("status") != "completed"):
                raise ValueError("Expected a completed, executed evaluate report; preflight is not execution")
            summary = value["execution"]
            if not isinstance(summary, dict) or not isinstance(summary.get("jobs"), list) or not summary["jobs"]:
                raise ValueError("Execution summary must contain jobs")
            for job in summary["jobs"]:
                project_path(root, job["output"])
            selection = value["preflight"]["selection"]
            if not isinstance(selection, list) or not selection:
                raise ValueError("Missing preflight job selection")
            entry["audit"] = audit_attempt(summary, expected_jobs=selection)
            for job in summary["jobs"]:
                directory = project_path(root, job["output"])
                metadata = json.loads((directory / "metadata.json").read_text())
                job_config = json.loads((directory / "job.json").read_text())
                dataset = json.loads((directory / "dataset.json").read_text())
                if not all(isinstance(item, dict) for item in (metadata, job_config, dataset)):
                    raise ValueError("Artifact metadata, job and dataset must be objects")
                fps, frames = metadata.get("video_fps"), metadata.get("num_video_frames")
                if (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
                        or type(frames) is not int or frames < 1
                        or not isinstance(metadata.get("view_files"), dict)):
                    raise ValueError("Invalid artifact FPS, frame count or view mapping")
                # Decode the current videos; do not trust a stale cached diagnostics JSON.
                visual = diagnose_artifact(directory)
                entry["jobs"].append({
                    "id": job["id"], "directory": str(directory),
                    "chunks": job["completed_chunks"], "vla_calls": job["vla_calls"],
                    "world_model_calls": job["world_model_calls"],
                    "wall_seconds_excluding_startup": job.get("seconds"),
                    "frames": metadata["num_video_frames"], "fps": metadata["video_fps"],
                    "duration_seconds": metadata["num_video_frames"] / metadata["video_fps"],
                    "state_provider": job_config["state_provider"],
                    "evaluation_mode": job_config.get("mode"),
                    "imagined_beyond_sample": dataset.get("imagined_beyond_sample"),
                    "videos": {name: str(project_path(root, str(directory / relative)))
                               for name, relative in metadata["view_files"].items()},
                    "visual_diagnostics": visual,
                })
            entry["ok"] = entry["audit"]["ok"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["errors"].append({"report": str(path), "error": str(exc)})
    result["ok"] = not result["errors"] and all(entry["ok"] for entry in result["entries"])
    result["status"] = "passed" if result["ok"] else "failed"
    return result
