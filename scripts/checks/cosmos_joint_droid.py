#!/usr/bin/env python3
"""Native recorded-command DROID test; reuse the framework's replay executor."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def joint_replay_source(provenance):
    """Replace inherited EEF-loader labels; this never transforms model inputs."""
    return {**provenance,
        "source_frames_per_chunk": 32,
        "replay_action_source": "recorded action.joint_position + action.gripper_position",
        "conversion": "Recorded action.joint_position and action.gripper_position; NOT future measured states. Official joint_pos RoboLab use_state prefix, 32 commands at 15Hz.",
        "recurrent_image_source": "generated_terminal_frame",
        "recurrent_state_source": "last_command_target_proxy_not_predicted_state",
        "geometry": "540x640; latent-decoded missing bottom/right borders edge-filled, not predicted pixels"}


def main():
    import numpy as np
    import pyarrow.parquet as pq
    from evaluation.datasets.droid import load_openpi_droid_episode
    from evaluation.execution.native_replay import run_native_replay
    from evaluation.execution.transport import write_packet
    from adapters.observations.droid import compose_droid_cosmos_view
    from adapters.backends.cosmos_actions import validate_joint_action
    from vlaworldeval.assets import verify_snapshot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=350)
    parser.add_argument("--start", type=int, default=15)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists() or args.chunks < 1:
        raise ValueError("Require a fresh project-local output and positive chunk count")
    asset = verify_snapshot(ROOT / "checkpoints/world_models/cosmos3_nano_droid_joint_fd/community_anshks_v1_best5600")
    if asset["repository"] != "anshks/cosmos3-nano-droid-fd-wm":
        raise ValueError("Joint FD checkpoint identity mismatch")
    data = load_openpi_droid_episode(ROOT / "datasets/cosmos3_droid/success", args.episode,
                                    args.start, args.chunks * 2)
    rows = pq.read_table(data.provenance["data_path"], filters=[("episode_index", "=", args.episode)],
                         columns=["action.joint_position", "action.gripper_position"]).to_pylist()
    selected = rows[args.start:args.start + args.chunks * 32]
    commands = np.asarray([list(row["action.joint_position"]) + [float(row["action.gripper_position"])]
                           for row in selected], dtype=np.float32).reshape(args.chunks, 32, 8)
    for chunk in commands:
        validate_joint_action(chunk, 32)
    views = data.trial["observation_bundle"]["views"]
    composite = compose_droid_cosmos_view(views["droid.exterior_1"], views["droid.exterior_2"], views["droid.wrist"])[0]
    if composite.shape != (540, 640, 3):
        raise ValueError("This native profile admits the official 540x640 DROID composite only")
    config_path = ROOT / "configs/replay/cosmos_joint_droid_native_v1.json"
    config = json.loads(config_path.read_text())
    packet = {"protocol": config["action"], "actions": commands,
        "reset_args": {"value": {"observation/image": composite,
            **data.trial["initial_state"].values, "prompt": data.trial["instruction"]}},
        "initial_views": {"droid.cosmos_composite": composite},
        "source": joint_replay_source(data.provenance)}
    prepared = output.with_name(output.name + "_input.npz")
    if prepared.exists():
        raise FileExistsError(prepared)
    prepared.parent.mkdir(parents=True, exist_ok=True)
    write_packet(prepared, packet)
    report = run_native_replay(config_path, prepared, output)
    print(json.dumps({"status": report["status"], "world_model_calls": report["world_model_calls"],
                      "output": str(output)}), flush=True)

if __name__ == "__main__":
    main()
