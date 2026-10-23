#!/usr/bin/env python3
"""Prepare official inputs and delegate all inference to native_replay."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def environment_snapshot(output: Path):
    """Preserve exact package metadata and raw dependency-check outcome."""
    command = [shutil.which("uv") or "uv", "pip", "check", "--python", sys.executable]
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".uv-cache"))
    for key in list(env):
        if key.lower().endswith("_proxy"):
            env.pop(key)
    check = subprocess.run(command, env=env, capture_output=True, text=True)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=index,driver_version,name,memory.free", "--format=csv"],
                            capture_output=True, text=True)
    report = {"python": sys.version, "executable": sys.executable,
              "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
              "dependency_check": {"command": command, "exit_code": check.returncode,
                                   "stdout": check.stdout, "stderr": check.stderr,
                                   "passed": check.returncode == 0, "known_exceptions": []},
              "nvidia_smi": {"exit_code": driver.returncode, "stdout": driver.stdout, "stderr": driver.stderr},
              "model_inference_performed": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_action_probe(baseline_dir: Path, probe_dir: Path, output: Path):
    """Compare shared-executor artifacts; no model calls or rollout loop."""
    import numpy as np
    from evaluation.execution.transport import read_packet

    baseline = json.loads((baseline_dir / "summary.json").read_text())
    probe = json.loads((probe_dir / "summary.json").read_text())
    configs = [json.loads((path / "config.json").read_text()) for path in (baseline_dir, probe_dir)]
    packets = [read_packet(path.with_suffix(".inputs.npz")) for path in (baseline_dir, probe_dir)]
    if any(s["status"] != "completed" or s["world_model_calls"] != 1 for s in (baseline, probe)):
        raise ValueError("Both runs must complete exactly one world-model call")
    if configs[0] != configs[1]:
        raise ValueError("Hold and action probe must use identical config/seed")
    for name in configs[0]["output_views"]:
        if not np.array_equal(packets[0]["initial_views"][name], packets[1]["initial_views"][name]):
            raise ValueError("Hold and probe initial pixels differ")
    if not np.array_equal(packets[0]["reset_args"]["value"]["initial_state"], packets[1]["reset_args"]["value"]["initial_state"]):
        raise ValueError("Hold and probe initial robot state differs")
    adapter_key = "adapters/world_models/playworld.py"
    if baseline["framework_code_sha256"][adapter_key] != probe["framework_code_sha256"][adapter_key]:
        raise ValueError("Model Adapter code changed between paired runs")
    views = {}
    for name in configs[0]["output_views"]:
        filenames = [path / f"{name}.mp4" for path in (baseline_dir, probe_dir)]
        frames = []
        for filename in filenames:
            pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(filename),
                                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"])
            frames.append(np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 192, 320, 3))
        if any(frame.shape != (6, 192, 320, 3) for frame in frames):
            raise ValueError("Expected one initial plus five generated frames per camera")
        delta = frames[1].astype(np.float32) - frames[0].astype(np.float32)
        views[name] = {
            "baseline_video_sha256": hashlib.sha256(filenames[0].read_bytes()).hexdigest(),
            "probe_video_sha256": hashlib.sha256(filenames[1].read_bytes()).hexdigest(),
            "raw_generated_chunk_hash_changed": baseline["chunk_prediction_sha256"][0][name] != probe["chunk_prediction_sha256"][0][name],
            "baseline_raw_generated_chunk_sha256": baseline["chunk_prediction_sha256"][0][name],
            "probe_raw_generated_chunk_sha256": probe["chunk_prediction_sha256"][0][name],
            "decoded_generated_frames_mae_255": float(np.abs(delta[1:]).mean()),
            "decoded_terminal_mae_255": float(np.abs(delta[-1]).mean()),
            "decoded_terminal_rmse_255": float(np.sqrt(np.square(delta[-1]).mean())),
            "decoded_terminal_max_abs_255": float(np.abs(delta[-1]).max()),
            "decoded_terminal_changed_pixel_fraction": float(np.any(delta[-1] != 0, axis=-1).mean()),
            "baseline_decoded_terminal_sha256": hashlib.sha256(frames[0][-1].tobytes()).hexdigest(),
            "probe_decoded_terminal_sha256": hashlib.sha256(frames[1][-1].tobytes()).hexdigest(),
        }
    report = {"status": "completed", "synthetic_probe": True, "recorded_action_replay": False,
              "task_success_evaluated": False, "physical_direction_accuracy_verified": False,
              "seed": configs[0]["seed"], "identical_config_and_initial_pixels_state": True,
              "identical_adapter_sha256": baseline["framework_code_sha256"][adapter_key],
              "baseline_summary": str((baseline_dir / "summary.json").resolve()),
              "probe_summary": str((probe_dir / "summary.json").resolve()),
              "action_provenance": probe["source"]["synthetic_probe"], "views": views,
              "nonzero_action_response_observed": all(v["raw_generated_chunk_hash_changed"] and v["decoded_terminal_mae_255"] > 0 for v in views.values()),
              "limits": "Single same-seed paired synthetic probe; no repeatability control, task-success or physical motion-direction validation. Pixel metrics use decoded lossy videos; raw chunk hashes are from pre-encoding model outputs."}
    output.mkdir(parents=True, exist_ok=False)
    report["comparison_video_layout"] = "left: hold; right: synthetic x+2cm"
    for name in configs[0]["output_views"]:
        filename = output / f"{name}_hold_vs_x_plus_2cm.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(baseline_dir / f"{name}.mp4"),
                        "-threads", "1", "-i", str(probe_dir / f"{name}.mp4"),
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2", "-filter_complex_threads", "1",
                        "-c:v", "libx264", "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", str(filename)],
                       capture_output=True, check=True)
        views[name]["comparison_video"] = str(filename.resolve())
        views[name]["comparison_video_sha256"] = hashlib.sha256(filename.read_bytes()).hexdigest()
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/replay/playworld_native_v1.json")
    parser.add_argument("--case-dir", type=Path,
                        default=ROOT / "datasets/playworld/benchmark/irom_test_carrot/init_0")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/integration/open_world_models_20261018/playworld/benchmark_hold")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--actions", type=Path, help="Audited raw absolute Cartesian [chunks,15,7] .npy")
    mode.add_argument("--hold-probe", action="store_true", help="Stationary input smoke; NOT trajectory replay")
    mode.add_argument("--environment-snapshot", action="store_true", help="Record exact distributions and raw uv pip check")
    mode.add_argument("--make-x-probe", action="store_true", help="Prepare a bounded synthetic x+2cm NPY and provenance")
    mode.add_argument("--compare-probe", action="store_true", help="Compare completed hold and synthetic action artifacts")
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "results/integration/open_world_models_20261018/playworld/benchmark_hold_v2")
    parser.add_argument("--probe-run", type=Path, help="Completed synthetic action run for comparison")
    parser.add_argument("--actions-provenance", type=Path, help="Synthetic probe provenance JSON matching --actions hash")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--admit-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.environment_snapshot:
        print(json.dumps(environment_snapshot(args.output), indent=2))
        return
    if args.make_x_probe:
        from evaluation.datasets.playworld import prepare_single_axis_probe
        report = prepare_single_axis_probe(args.baseline, args.output,
                                          ROOT / "third_party/playworld/openworld/world_models/vidwm_world_model.py")
        print(json.dumps(report, indent=2))
        return
    if args.compare_probe:
        if args.probe_run is None:
            parser.error("--compare-probe requires --probe-run")
        print(json.dumps(compare_action_probe(args.baseline, args.probe_run, args.output), indent=2))
        return
    packet = args.output.with_suffix(".inputs.npz")
    from evaluation.datasets.playworld import prepare_benchmark_packet
    preparation = prepare_benchmark_packet(args.config, args.case_dir, packet,
                                           actions_path=args.actions, hold_probe=args.hold_probe,
                                           actions_provenance_path=args.actions_provenance)
    if args.prepare_only:
        print(json.dumps(preparation, indent=2))
        return
    from evaluation.execution.native_replay import run_native_replay
    report = run_native_replay(args.config, packet, args.output, admit_only=args.admit_only)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
