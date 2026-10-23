"""Prepare observed-joint FK actions for the shared IWS replay executor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import inspect
import subprocess
import re

import numpy as np
from vlaworldeval.assets import verify_snapshot

FK_REVISION = "805ea8bbb58eb2bf727d05970b7e287c99b82b42"
URDFPY_REVISION = "90737eab7af4399c97b23cda81152994c07a35cd"
DATASET_REVISIONS = {
    "yixuan1999/interactive-world-sim-min-data": "987c175bcc29ddd0f6e6e3096b2c20df300253b0",
    "yixuan1999/interactive-world-sim-data": "6c3010b4d8c0fae8b683856ca1c550869d5bdb04",
}


def _official_fk_dependencies():
    """Import the unmodified official helpers and record their robot assets."""
    from vlaworldeval.paths import project_root
    root = project_root()
    revisions = {
        "iws": (root / "third_party/iws", "3ba69b41d070d7a56e79c287da5b24a7d90742aa"),
        "yixuan_utilities": (root / "third_party/iws_dependencies/yixuan_utilities", FK_REVISION),
        "urdfpy": (root / "third_party/iws_dependencies/urdfpy", URDFPY_REVISION),
    }
    for name, (directory, expected) in revisions.items():
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"IWS FK dependency revision mismatch: {name}")
        subprocess.run(["git", "-C", str(directory), "diff", "--quiet", "HEAD", "--"], check=True)
    from yixuan_utilities.kinematics_helper import KinHelper
    from interactive_world_sim.utils.action_utils import joint_pos_to_action_primitive
    from interactive_world_sim.utils.aloha_conts import (
        MASTER_GRIPPER_JOINT_UNNORMALIZE_FN, PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
    )
    import urdfpy
    kin_path = Path(inspect.getfile(KinHelper)).resolve()
    action_path = Path(inspect.getfile(joint_pos_to_action_primitive)).resolve()
    if kin_path != revisions["yixuan_utilities"][0] / "src/yixuan_utilities/kinematics_helper.py":
        raise ValueError("IWS FK imported a different yixuan_utilities package")
    if action_path != revisions["iws"][0] / "interactive_world_sim/utils/action_utils.py":
        raise ValueError("IWS FK imported a different IWS action_utils package")
    if not Path(urdfpy.__file__).resolve().is_relative_to(revisions["urdfpy"][0]):
        raise ValueError("IWS FK imported a different urdfpy package")
    urdf = kin_path.parent / "assets/robot/trossen_description/vx300s.urdf"
    audit = {
        "source_revisions": {name: revision for name, (_, revision) in revisions.items()},
        "robot": "trossen_vx300s", "eef_link": "vx300s/ee_arm_link",
        "backend": "official KinHelper / SAPIEN 2.2.2 create_pinocchio_model",
        "files_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (kin_path, action_path, urdf)},
    }
    return (KinHelper, joint_pos_to_action_primitive,
            PUPPET_GRIPPER_JOINT_NORMALIZE_FN, MASTER_GRIPPER_JOINT_UNNORMALIZE_FN, audit)


def official_fk_actions(joint_positions, world_t_robot_base):
    """Exactly the real_aloha_dataset bimanual_push path, using observed joints.

    The upstream primitive duplicates the final finger joint for SAPIEN, clips
    robot-frame EEF x/y, applies world_t_robot_base, and selects world x/y.
    Stored action commands are neither accepted nor consulted here.
    """
    joints = np.asarray(joint_positions)
    bases = np.asarray(world_t_robot_base)
    if joints.ndim != 2 or joints.shape[1] != 14 or len(joints) < 1 or joints.dtype.kind not in "fi" or not np.isfinite(joints).all():
        raise ValueError("IWS official FK requires finite observed joint_pos [T,14]")
    if bases.shape != (len(joints), 2, 4, 4) or bases.dtype.kind not in "fi" or not np.isfinite(bases).all():
        raise ValueError("IWS official FK requires finite world_t_robot_base [T,2,4,4]")
    if not np.allclose(bases[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("IWS robot bases must be homogeneous world transforms")
    KinHelper, primitive, puppet_normalize, master_unnormalize, audit = _official_fk_dependencies()
    kin = KinHelper("trossen_vx300s")
    actions = []
    for raw_joint, base in zip(joints, bases):
        joint = raw_joint.copy()
        for robot in range(2):
            joint[robot * 7 + 6] = master_unnormalize(puppet_normalize(joint[robot * 7 + 6]))
        actions.append(primitive(joint_pos=joint, ctrl_mode="bimanual_push",
                                 base_pose_in_world=base, kin_helper=kin))
    actions = np.concatenate(actions).astype(np.float32)
    if actions.shape != (len(joints), 4) or not np.isfinite(actions).all():
        raise ValueError("Official IWS FK returned invalid bimanual world XY actions")
    audit = {**audit, "joint_pos_sha256": hashlib.sha256(np.ascontiguousarray(joints).tobytes()).hexdigest(),
             "world_t_robot_base_sha256": hashlib.sha256(np.ascontiguousarray(bases).tobytes()).hexdigest()}
    return actions, audit


def prepare_iws_packet(config_path: Path, dataset: Path, output: Path, *, start=0, chunks=20,
                       action_source="official_fk"):
    import h5py
    from adapters.world_models.iws import prepare_image
    from evaluation.execution.transport import write_packet

    if type(start) is not int or type(chunks) is not int or start < 0 or chunks < 1:
        raise ValueError("start/chunks must be nonnegative/positive integers")
    if output.exists():
        raise FileExistsError(output)
    if action_source not in {"official_fk", "stored_command_diagnostic"}:
        raise ValueError("Unknown IWS action source")
    config = json.loads(config_path.read_text())
    if config["adapter"] != "IWSAdapter" or config["action"]["horizon"] != 1 or config["action"]["dimension"] != 4:
        raise ValueError("IWS packet requires the one-step 4-D PushT protocol")
    dataset = dataset.resolve()
    verified = verify_snapshot(dataset.parent.parent.parent)
    if DATASET_REVISIONS.get(verified["repository"]) != verified["revision"]:
        raise ValueError("IWS preparation requires a pinned official mini or full dataset")
    relative = dataset.relative_to(dataset.parent.parent.parent).as_posix()
    if not re.fullmatch(r"pusht/val/episode_[0-9]+\.hdf5", relative):
        raise ValueError("IWS preparation requires a verified official PushT val episode")
    with h5py.File(dataset) as f:
        count = len(f["obs/joint_pos"] if action_source == "official_fk" else f["action"])
        if start + chunks >= count:
            raise ValueError("Requested future actions exceed episode; no repetition or padding")
        window = slice(start, start + chunks + 1)
        if action_source == "official_fk":
            commands, audit = official_fk_actions(f["obs/joint_pos"][window], f["obs/world_t_robot_base"][window])
            conversion = "Official real_aloha_dataset observed joint_pos -> puppet/master gripper conversion -> KinHelper(trossen_vx300s) FK -> bimanual_push robot XY clip -> world_t_robot_base -> absolute world XY metres; initial and all future actions use FK, no stored action commands."
        else:
            commands = np.asarray(f["action"][window], dtype=np.float32)
            audit = None
            conversion = "DIAGNOSTIC ONLY: stored action[start] seeds history and stored action[start+1:] drives future frames; no observed-joint FK; not the official real_aloha_dataset route."
        if commands.shape != (chunks + 1, 4) or not np.isfinite(commands).all():
            raise ValueError("PushT actions must be finite [T,4]")
        image = prepare_image(f["obs/images/camera_1_color"][start])
    digest = hashlib.sha256()
    with dataset.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    packet = {
        "protocol": config["action"],
        "actions": commands[1:, None].copy(),
        "reset_args": {"value": {"image": image, "initial_action": commands[0].copy()}},
        "initial_views": {"camera_1_color": image},
        "source": {
            "dataset": str(dataset.resolve()), "sha256": digest.hexdigest(),
            "repository": verified["repository"],
            "revision": verified["revision"],
            "start_frame": start, "fps": 10,
            "action_source": action_source, "official_fk_audit": audit,
            "initial_action_source_frame_index": start,
            "action_source_frame_indices": list(range(start+1, start+chunks+1)),
            "conversion": conversion + " Aspect-centre crop/area resize camera_1_color to 128x128; predicted latent feedback only.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_packet(output, packet)
    return {"ok": True, "model_inference_performed": False,
            "action_shape": list(packet["actions"].shape), "source": packet["source"]}
