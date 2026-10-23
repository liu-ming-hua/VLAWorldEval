"""Read-only engineering audit of unified executor artifacts, not model quality."""
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation.execution.reliability import clean_worker_exit


def audit_job(record):
    directory = Path(record["output"])
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    try:
        job = json.loads((directory / "job.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        pair = json.loads((directory / "pair_manifest.json").read_text())
        steps = job["chunks"]
        wm = pair["world_model"]["action"]
        from evaluation.manifests import VideoSpec
        declared_video = pair["world_model"].get("video")
        video_spec = VideoSpec.from_dict(declared_video) if declared_video else VideoSpec(wm["rate_hz"], wm["horizon"])
        from evaluation.modes import POLICY_MODES, pilworld_alignment
        from evaluation.manifests import PairManifest
        policy = job["mode"] in POLICY_MODES
        require(record["status"] == "completed" and record["completed_chunks"] == steps, "Incomplete job")
        require(record["world_model_calls"] == steps, "Wrong world-model call count")
        require(record["vla_calls"] == (steps if policy else 0), "Wrong VLA call count")
        require(metadata["format_version"] == 2 and metadata["num_rollout_steps"] == steps, "Wrong artifact version/steps")
        require(not metadata["trial"].get("partial"), "Partial artifact cannot pass acceptance")
        require(metadata["video_fps"] == video_spec.fps, "Metadata FPS does not match pair video protocol")
        with np.load(directory / "actions.npz", allow_pickle=False) as arrays:
            for prefix, spec, count in (("bridged", wm, steps), ("native", pair["vla"]["action"], steps if policy else 0)):
                values = arrays[f"{prefix}_actions"]
                require(values.shape == ((count, spec["horizon"], spec["dimension"]) if count else (0, 0, 0)),
                        f"Wrong {prefix} action shape")
                require(np.isfinite(values).all(), f"Nonfinite {prefix} actions")
                require(np.all(arrays[f"{prefix}_action_lengths"] == spec["horizon"]), f"Wrong {prefix} action lengths")
                require(arrays[f"{prefix}_action_mask"].shape == values.shape[:2]
                        and arrays[f"{prefix}_action_mask"].all(), f"Wrong {prefix} action mask")
        with np.load(directory / "states.npz", allow_pickle=False) as arrays:
            values, mask = arrays["state_values"], arrays["state_mask"]
            require(len(values) == steps + 1 and mask.shape == values.shape, "Wrong state observation count/mask")
            require(mask.dtype == np.bool_, "State mask must be boolean")
            require(np.isfinite(values[mask]).all(), "Nonfinite valid state values")
            if not policy or job["state_provider"] == "none":
                # Replay may retain the measured initial state, but must not invent future state.
                require(not mask[1:].any(), "Unexpected recurrent state for replay/image-only policy")
            if policy and job["state_provider"] == "none":
                require(not mask.any(), "Image-only policy must not synthesize initial state")
            if policy and job["state_provider"] != "none":
                require(np.all(arrays["state_sources"][1:] == job["state_provider"]), "Wrong recurrent state provenance")
        expected_frames = video_spec.frame_count(steps)
        require(metadata["num_video_frames"] == expected_frames, "Wrong video frame count in metadata")
        files = [directory / "rollout.mp4"]
        require(bool(metadata["view_files"]), "Missing per-view videos")
        output_views = pair.get("view_bridge", {}).get("output_views") or pair["world_model"]["output_views"]
        require(set(metadata["view_files"]) == set(output_views), "Wrong/missing output views")
        for relative in metadata["view_files"].values():
            path = (directory / relative).resolve()
            require(path.is_relative_to(directory.resolve()), "Video path leaves artifact directory")
            if path.is_relative_to(directory.resolve()):
                files.append(path)
        for path in files:
            info = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=nb_frames,avg_frame_rate", "-of", "json", str(path),
            ], text=True, timeout=15))["streams"][0]
            require(int(info["nb_frames"]) == expected_frames, f"Wrong actual frame count: {path.name}")
            numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
            require(abs(numerator / denominator - metadata["video_fps"]) < 1e-6, f"Wrong FPS: {path.name}")
        journal = sorted((directory / "partial").glob("chunk_*.json"))
        require(len(journal) == steps, "Missing per-chunk audit records")
        if job["mode"] == "pilworld":
            declared = PairManifest.from_dict(pair)
            endpoint = pilworld_alignment(declared.vla.action, declared.world_model.action)
            require(job["state_provider"] == "pilworld_action_target", "Wrong pilworld state provider")
            for path in journal:
                row = json.loads(path.read_text())
                state_meta = row["state_update_metadata"]
                require(row["evaluation_mode"] == "pilworld", "Missing pilworld mode provenance")
                require(all(state_meta.get(key) == value for key, value in endpoint.items()), "Wrong pilworld endpoint alignment")
                require(state_meta.get("wm_predicted_state") is False, "PiL proxy mislabeled as WM-predicted state")
        for previous, current in zip(journal, journal[1:]):
            prev, curr = json.loads(previous.read_text()), json.loads(current.read_text())
            require(prev["output_terminal_view_hashes"] == curr["input_view_hashes"], "Generated image handoff mismatch")
            require(curr["observation_input_source"] == "previous_generated_terminal_views", "Wrong recurrent input source")
            offsets = pair["vla"].get("history_offsets")
            if policy and offsets and min(offsets) < 0:
                depth = 1 - min(offsets)
                if video_spec.advance_frames >= depth:
                    require(prev["output_history_hashes"] == curr["input_history_hashes"],
                            "Generated image history handoff mismatch")
                else:
                    require(prev["output_observation_history_hashes"] == curr["input_history_hashes"],
                            "Carried generated image history handoff mismatch")
        require(not list((directory / "partial").glob("chunk_*.npz")), "Successful partial arrays were not removed")
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as exc:
        failures.append(f"Artifact audit error: {exc}")
    return {"output": str(directory), "ok": not failures, "failures": failures}


def audit_attempt(summary, *, expected_jobs=None):
    jobs = [audit_job(record) for record in summary["jobs"]]
    cleanup = summary.get("worker_cleanup", [])
    cleanup_ok = bool(cleanup) and all(item.get("status") == "closed"
                                      and clean_worker_exit(item.get("cleanup")) for item in cleanup)
    selection_ok = True
    if expected_jobs is not None:
        expected = sorted((job["id"], job["episode_index"], job["seed"]) for job in expected_jobs)
        actual = sorted((job["id"], job["episode"], job["seed"]) for job in summary["jobs"])
        selection_ok = expected == actual
    return {"ok": summary["status"] == "completed" and bool(jobs) and all(job["ok"] for job in jobs) and cleanup_ok and selection_ok,
            "jobs": jobs, "cleanup_ok": cleanup_ok, "selection_ok": selection_ok, "quality_evaluated": False}
