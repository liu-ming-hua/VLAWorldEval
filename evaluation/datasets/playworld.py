"""Prepare official PlayWorld initialization inputs for the shared executor.

The public benchmark is an initialization suite, not action trajectories. A hold
probe is explicit and never described as recorded-action replay or task success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import ast

import numpy as np
from PIL import Image

VIEW_NAMES = ("exterior_right", "exterior_left", "wrist")


def prepare_benchmark_packet(config_path: Path, case_dir: Path, output: Path, *,
                             actions_path: Path | None = None, hold_probe: bool = False,
                             actions_provenance_path: Path | None = None):
    from evaluation.execution.transport import write_packet
    import yaml

    if output.exists():
        raise FileExistsError(output)
    if (actions_path is None) == (not hold_probe):
        raise ValueError("Provide recorded actions_path OR explicitly choose hold_probe")
    if actions_provenance_path is not None and actions_path is None:
        raise ValueError("Actions provenance requires an actions file")
    config = json.loads(config_path.read_text())
    if (config["adapter"] != "PlayWorldAdapter" or config["reset_operation"] != "reset"
            or config["action"]["horizon"] != 15 or config["action"]["dimension"] != 7
            or config["output_views"] != list(VIEW_NAMES)):
        raise ValueError("Expected the official PlayWorld 15x7 three-view profile")
    case_dir = case_dir.resolve()
    annotation = case_dir / "initialization.yaml"
    payload = yaml.safe_load(annotation.read_text())
    robot = payload["initial_state"]["robot"]
    if robot["state_representation"] != "cartesian_position_with_gripper":
        raise ValueError("Benchmark must provide absolute Cartesian XYZ/RPY/gripper state")
    state = np.asarray(robot["state"], dtype=np.float32)
    instruction = payload.get("instruction")
    if state.shape != (7,) or not np.isfinite(state).all() or not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Benchmark requires a finite 7D initial robot state and instruction")
    views = {}
    files = [annotation]
    for name in VIEW_NAMES:
        path = case_dir / f"{name}.png"
        with Image.open(path) as image:
            views[name] = np.asarray(image.convert("RGB"))
        if views[name].shape != (192, 320, 3):
            raise ValueError(f"Official benchmark view must already be 320x192: {path}")
        files.append(path)
    if hold_probe:
        actions = np.repeat(state[None, None], 15, axis=1)
        conversion = "explicit stationary absolute-state probe from official initialization; NOT recorded actions"
    else:
        actions = np.load(actions_path, allow_pickle=False)
        conversion = "user-supplied raw absolute XYZ/RPY/gripper [chunks,15,7]; benchmark provides initialization only"
        files.append(actions_path)
    if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[1:] != (15, 7) or not np.isfinite(actions).all():
        raise ValueError("Actions must be finite [chunks,15,7] absolute robot-base XYZ/RPY/gripper")
    source = {"dataset": "tennyyyin/open-world-benchmark", "case": str(case_dir),
              "instruction": instruction, "conversion": conversion,
              "recorded_action_replay": False,
              "action_provenance": "stationary_probe" if hold_probe else "user_supplied_not_independently_verified",
              "task_success_evaluated": False,
              "runtime_protocol": "OpenWorld SVD evaluation: nonoverlapping 15 source ticks; sampled offsets [0,3,6,9,12]",
              "source_tick_hz": 15, "source_advance_ticks": 15, "sampled_action_offsets": [0, 3, 6, 9, 12],
              "source_advance_seconds": 1.0, "sampled_span_seconds": 0.8,
              "video_note": "Official evaluator playback at 5fps; model fps conditioning remains 7; not Ctrl inclusive timing",
              "calibration_required": False,
              "calibration_note": "SVD checkpoint uses ordered RGB cameras and raw robot-base poses; no camera rays/extrinsics input",
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    if actions_provenance_path is not None:
        provenance = json.loads(actions_provenance_path.read_text())
        if (provenance.get("synthetic") is not True or provenance.get("recorded_action_replay") is not False
                or provenance.get("actions_sha256") != hashlib.sha256(actions_path.read_bytes()).hexdigest()):
            raise ValueError("Synthetic action provenance must declare synthetic/non-recorded and match raw actions hash")
        source.update(action_provenance="synthetic_single_axis_absolute_eef_probe", synthetic=True,
                      synthetic_probe=provenance,
                      conversion="synthetic absolute EEF x-axis +2cm smoothstep from official initial state; NOT recorded actions or task success")
        source["input_sha256"][str(actions_provenance_path)] = hashlib.sha256(actions_provenance_path.read_bytes()).hexdigest()
    packet = {"protocol": config["action"], "actions": np.asarray(actions, dtype=np.float32),
              "initial_views": views,
              "reset_args": {"value": {"views": views, "initial_state": state, "instruction": instruction}},
              "source": source}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_packet(output, packet)
    return {"ok": True, "model_inference_performed": False, "output": str(output),
            "action_shape": list(actions.shape), "source": source}


def single_axis_probe_actions(initial_state, droid_stats, *, displacement_m=0.02):
    """Bounded synthetic absolute actions; selected tick 12 reaches the target."""
    initial = np.asarray(initial_state, dtype=np.float32)
    lower = np.asarray(droid_stats["state_01"], dtype=np.float32)
    upper = np.asarray(droid_stats["state_99"], dtype=np.float32)
    if (initial.shape != (7,) or lower.shape != (7,) or upper.shape != (7,)
            or not np.isfinite([initial, lower, upper]).all() or np.any(lower >= upper)):
        raise ValueError("Expected finite initial state and DROID percentile bounds, each [7]")
    if not np.isfinite(displacement_m) or not 0 < displacement_m <= 0.02:
        raise ValueError("This minimal diagnostic allows only positive x displacements up to 2cm")
    t = np.minimum(np.arange(15, dtype=np.float32) / 12, 1)
    smooth = t * t * (3 - 2 * t)
    actions = np.repeat(initial[None], 15, axis=0)
    actions[:, 0] += np.float32(displacement_m) * smooth
    if np.any(actions < lower) or np.any(actions > upper):
        raise ValueError("Synthetic probe leaves official DROID state_01/state_99 bounds; do not clip it")
    return actions[None]


def prepare_single_axis_probe(baseline_dir: Path, output: Path, stats_source: Path):
    """Generate an NPY and provenance, never model predictions or a rollout."""
    from evaluation.execution.transport import read_packet

    baseline = json.loads((baseline_dir / "summary.json").read_text())
    if baseline["status"] != "completed" or baseline["world_model_calls"] != 1:
        raise ValueError("Probe requires a completed one-chunk hold baseline")
    packet = read_packet(baseline_dir.with_suffix(".inputs.npz"))
    initial = packet["reset_args"]["value"]["initial_state"]
    if not np.array_equal(packet["actions"], np.repeat(initial[None, None], 15, axis=1)):
        raise ValueError("Baseline must be an explicit stationary initial-state probe")
    # Read the literal official preset without importing torch or the model.
    tree = ast.parse(stats_source.read_text())
    stats = next(ast.literal_eval(node.value)["droid"] for node in tree.body
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                 and node.target.id == "DATASET_STATS")
    actions = single_axis_probe_actions(initial, stats)
    config = json.loads((baseline_dir / "config.json").read_text())
    for filename, expected in packet["source"]["input_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Official initialization changed since hold: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    action_path = output / "actions.npy"
    np.save(action_path, actions, allow_pickle=False)
    report = {"kind": "single_axis_absolute_eef_smoothstep", "synthetic": True,
              "recorded_action_replay": False, "task_success_evaluated": False,
              "axis": "robot_base_x", "target_displacement_m": 0.02,
              "schedule": "smoothstep 3t^2-2t^3, t=min(raw_tick/12,1); ticks 13/14 hold target",
              "action_shape": list(actions.shape), "initial_state": np.asarray(initial).tolist(),
              "selected_ticks": [0, 3, 6, 9, 12], "selected_x": actions[0, ::3, 0].tolist(),
              "other_dimensions_unchanged": bool(np.array_equal(actions[0, :, 1:], np.repeat(initial[None, 1:], 15, axis=0))),
              "normalization_clipping_required": False,
              "stats_source": str(stats_source.resolve()), "stats_source_sha256": hashlib.sha256(stats_source.read_bytes()).hexdigest(),
              "state_01": list(stats["state_01"]), "state_99": list(stats["state_99"]),
              "raw_min": actions.min(axis=(0, 1)).tolist(), "raw_max": actions.max(axis=(0, 1)).tolist(),
              "seed": config["seed"], "baseline_summary": str((baseline_dir / "summary.json").resolve()),
              "baseline_input_sha256": baseline["input_sha256"],
              "actions_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
