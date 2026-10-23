"""Native DROID reference preparation; all inference stays in the shared executor."""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from evaluation.datasets.droid import load_droid_episode
from evaluation.datasets.video import decode_source_views
from evaluation.execution.transport import write_packet
from evaluation.reporting.task_scores import _digest
from vlaworldeval.assets import MANIFEST, verify_snapshot
from vlaworldeval.atomic_json import publish_json_exclusive

VIEW = "droid.cosmos_composite"


class SnapshotChecks:
    """One preparation process hashes shared payloads once, never trusts a disk cache."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest_sha256 = _digest(self.directory/MANIFEST)
        self.checked = {}

    @staticmethod
    def signature(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def verify(self, names):
        if _digest(self.directory/MANIFEST) != self.manifest_sha256:
            raise ValueError("Snapshot manifest changed during preparation")
        signatures = {name: self.signature(self.directory/name) for name in names}
        for name, signature in signatures.items():
            if name in self.checked and self.checked[name] != signature:
                raise ValueError(f"Previously verified payload changed: {name}")
        # Linux can assign two very fast same-size writes the same timestamp.
        # Never reuse a stat cache for recently created/modified files: hash
        # again until both timestamps predate this verification by a full second.
        cutoff = time.time_ns()-1_000_000_000
        pending = [name for name in names if name not in self.checked
                   or max(signatures[name][-2:]) >= cutoff]
        if pending:
            verify_snapshot(self.directory, names=pending)
        if any(self.signature(self.directory/name) != value for name, value in signatures.items()):
            raise ValueError("Payload changed while its checksum was being verified")
        self.checked.update(signatures)
        return {"verification_scope": "case_files_verified_in_this_process", "files_verified": len(names),
                "newly_hashed_files": len(pending), "snapshot_manifest_sha256": self.manifest_sha256}


def resolve_chunks(dataset, episode, start, chunks):
    if type(episode) is not int or episode < 0 or type(start) is not int or start < 0:
        raise ValueError("Episode/start must be nonnegative integers")
    if chunks != "all":
        if type(chunks) is not int or chunks < 1:
            raise ValueError("Choose a positive chunk count or all complete chunks")
        return chunks
    import pyarrow.parquet as pq
    matches = []
    for path in sorted((Path(dataset)/"meta/episodes").glob("chunk-*/file-*.parquet")):
        matches.extend(pq.read_table(path, columns=["episode_index", "length"],
                                    filters=[("episode_index", "=", episode)]).to_pylist())
    if len(matches) != 1 or type(matches[0]["length"]) is not int:
        raise ValueError("Exactly one valid episode metadata row is required")
    count = (matches[0]["length"]-start-1)//16
    if count < 1:
        raise ValueError("Episode has no complete16-transition window; no padding")
    return count


def compose_native_views(views):
    import torch
    from torch.nn import functional as F
    data = {name: torch.from_numpy(np.asarray(views[f"droid.{name}"]).copy()).permute(0, 3, 1, 2).float()/255
            for name in ("wrist", "exterior_1", "exterior_2")}
    wrist = data["wrist"]
    half = (wrist.shape[-2]//2, wrist.shape[-1]//2)
    bottom = torch.cat([F.interpolate(data[name], size=half, mode="bilinear", align_corners=False)
                        for name in ("exterior_1", "exterior_2")], dim=-1)
    result = torch.cat([wrist, bottom], dim=-2)
    return (result*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def compare_source_compositor(views, expected, repo):
    """Execute the reviewed upstream composition method with augmentation off."""
    import torch
    from torch.nn import functional as F
    path = Path(repo)/"cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DROIDLeRobotDataset")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compose_multi_view")
    namespace = {"torch": torch, "F": F, "Any": Any}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    sample = {name: torch.from_numpy(np.asarray(views[key]).copy()).permute(0, 3, 1, 2).float()/255
              for name, key in (("wrist", "droid.wrist"), ("left", "droid.exterior_1"), ("right", "droid.exterior_2"))}
    context = SimpleNamespace(_image_features={key: key for key in sample},
                              _use_image_augmentation=False, _color_augmentor=None)
    raw = namespace["_compose_multi_view"](context, sample)
    official = (raw*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    np.testing.assert_array_equal(expected, official)
    return {"exact_equal": True, "upstream_file": str(path), "sha256": _digest(path),
            "method": "DROIDLeRobotDataset._compose_multi_view", "training_augmentation": False}


def convert_episode(dataset, episode, start, chunks, repo):
    chunks = resolve_chunks(dataset, episode, start, chunks)
    data = load_droid_episode(Path(dataset), episode, start, chunks, protocol="cosmos", require_instruction=False)
    views = decode_source_views(data.provenance, 16*chunks+1)
    for name, frames in views.items():
        if not np.array_equal(frames[0], data.trial["observation_bundle"]["views"][name]):
            raise ValueError("Independent initial camera decode differs from the reference timeline")
    composed = compose_native_views(views)
    parity = compare_source_compositor({key: value[:1] for key, value in views.items()}, composed[:1], repo)
    height, width = composed.shape[1:3]
    region = composed[:, :height//16*16, :width//16*16].copy()
    source = {**data.provenance, "composition_parity": parity,
        "instruction_conditioning": False,
        "source_instruction_available": bool(data.trial["instruction"]),
        "conversion": "official no-augmentation float bilinear composite; model keeps full conditioning image; score actual16-grid decoder region only",
        "model_conditioning_hwc": list(composed.shape[1:]), "scored_hwc": list(region.shape[1:]),
        "edge_filled_pixels_scored": False,
        "unscored_tail_transitions": data.provenance["length"]-start-16*chunks-1}
    packet = {"actions": np.stack(data.replay_actions), "reset_args": {"value": composed[0]},
              "initial_views": {VIEW: region[0]}, "source": source}
    return packet, region


def prepare_native_case(root, dataset, config_path, output, *, episode=0, start=0, chunks=1,
                        _checks=None, _cohort=None):
    """Requires a complete pinned download marker; verifies every referenced file."""
    from evaluation.datasets.native_reference import save_reference
    from evaluation.planning.registry import project_path
    from evaluation.reporting.replay_metrics import validate_reference
    from evaluation.execution.transport import read_packet
    root = Path(root).resolve()
    dataset, config_path, output = [project_path(root, str(p)) for p in (dataset, config_path, output)]
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((dataset.parent/MANIFEST).read_text())
    if (manifest["repository"] != "nvidia/Cosmos3-DROID" or manifest["revision"] != "5c11a20accb11497270a5247a7f1e66ad04c956c"):
        raise ValueError("Native20k-EMA requires the pinned Cosmos3-DROID snapshot")
    config = json.loads(config_path.read_text())
    if config["adapter"] != "Cosmos3DroidFDNativeAdapter" or config["action"]["dimension"] != 10 or config["action"]["horizon"] != 16:
        raise ValueError("Use the explicit native20k-EMA config, not the policy border-restoration adapter")
    if config.get("instruction_conditioning") is not False:
        raise ValueError("Native20k-EMA replay requires explicit language-free forward dynamics")
    repo = project_path(root, "third_party/cosmos-framework")
    packet, frames = convert_episode(dataset, episode, start, chunks, repo)
    paths = [dataset/"meta/info.json", *[Path(packet["source"][key]) for key in ("metadata_path", "data_path")]]
    paths += [Path(info["path"]) for info in packet["source"]["videos"].values()]
    names = [str(path.relative_to(dataset.parent)) for path in paths]
    if _checks is not None and _checks.directory != dataset.parent:
        raise ValueError("Snapshot verification session belongs to another dataset")
    packet["source"]["snapshot_verification"] = (_checks.verify(names) if _checks is not None
                                                  else verify_snapshot(dataset.parent, names=names))
    packet["source"]["snapshot_manifest_sha256"] = _digest(dataset.parent/MANIFEST)
    packet["protocol"] = config["action"]
    output.mkdir(parents=True)
    write_packet(output/"input.npz", packet)
    save_reference(output/"input.npz", output/"reference.npz", {VIEW: frames}, {VIEW: range(start, start+len(frames))},
        domain="droid", split="cosmos20k_loader_holdout" if _cohort else "source_snapshot_case_membership_not_yet_cohort_audited",
        episode=f"{dataset.name}/{episode}",
        condition_kind="future_measured_kinematics", conversion=packet["source"]["conversion"],
        native_region=packet["source"]["scored_hwc"], camera_offsets=packet["source"]["videos"],
        cohort=_cohort, frame_axis="episode_frame")
    # Only this new, unfrozen prepared artifact is rewritten. IPC stays fast,
    # while potentially hundreds of GB of real-reference pixels are lossless.
    write_packet(output/"reference.npz", read_packet(output/"reference.npz"), compressed=True)
    publish_json_exclusive(output/"config.json", config)
    validate_reference(read_packet(output/"reference.npz"), config, packet,
        {VIEW: frames[:1]}, input_sha256=_digest(output/"input.npz"))
    return {"ok": True, "prepared": str(output), "chunks": len(packet["actions"]),
            "source_length": packet["source"]["length"],
            "action_shape": list(packet["actions"].shape), "model_inference_performed": False}
