"""Prepare every20k-EMA holdout episode; model inference uses native-study."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Sequence

from evaluation.datasets.cosmos_reference import SnapshotChecks, prepare_native_case
from evaluation.reporting.task_scores import _digest
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json


def validate_selection(selection, dataset):
    """Check deterministic ordered membership and the frozen source metadata."""
    import torch
    if (selection.get("model") != "cosmos3_nano_droid_fd_20k_ema"
            or selection.get("expected_episodes") != 2157):
        raise ValueError("This full holdout protocol belongs only to20k-EMA")
    expected = [(subset, int(index)) for subset, count in (("success", 57639), ("failure", 14268))
                for index in torch.randperm(count, generator=torch.Generator().manual_seed(42)).tolist()[:round(count*.03)]]
    rows = selection["episodes"]
    if [(row["subset"], row["episode"]) for row in rows] != expected:
        raise ValueError("Selection differs from all2157 ordered official-loader holdout IDs")
    metadata = selection.get("metadata_sha256", {})
    actual_paths = set()
    for subset in ("success", "failure"):
        actual_paths.add(str(dataset/subset/"meta/info.json"))
        actual_paths.update(str(p) for p in (dataset/subset/"meta/episodes").glob("chunk-*/file-*.parquet"))
    if set(metadata) != actual_paths or any(_digest(Path(path)) != sha for path, sha in metadata.items()):
        raise ValueError("Full source metadata membership/checksums differ from selection audit")
    for row in rows:
        if (row["id"] != f"cosmos20k_{row['subset']}_{row['episode']:06d}"
                or type(row["length"]) is not int or row["length"] < 1
                or type(row["complete16_action_chunks"]) is not int
                or row["complete16_action_chunks"] != (row["length"]-1)//16):
            raise ValueError("Invalid full-episode identity/length/chunk count; cannot omit short cases")


def partition_selection(selection):
    """Account for every selected episode, including native-ineligible short ones."""
    eligible, short = [], []
    for row in selection["episodes"]:
        if row["complete16_action_chunks"]:
            eligible.append(row)
        else:
            short.append({"id": row["id"], "subset": row["subset"], "episode": row["episode"],
                          "length": row["length"], "status": "no_complete_native_window",
                          "available_transitions": row["length"]-1,
                          "reason": "Official16-action loader requires at least17 recorded frames; no padding or fabricated future"})
    return eligible, short


def native_eligibility(root, dataset, selection):
    """Execute upstream span construction against all real source metadata."""
    import pyarrow.parquet as pq
    source = root/"third_party/cosmos-framework/cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_episode_spans")
    namespace = {"Any": Any, "Sequence": Sequence}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    eligible, short = partition_selection(selection)
    actual, sliding_starts = [], 0
    columns = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    for subset, count in (("success", 57639), ("failure", 14268)):
        rows = [row for path in sorted((dataset/subset/"meta/episodes").glob("chunk-*/file-*.parquet"))
                for row in pq.read_table(path, columns=columns).to_pylist()]
        rows.sort(key=lambda row: row["episode_index"])
        if [row["episode_index"] for row in rows] != list(range(count)):
            raise ValueError("Native eligibility requires complete unique source episode metadata")
        selected = [row for row in selection["episodes"] if row["subset"] == subset]
        for row in selected:
            original = rows[row["episode"]]
            if (row["length"] != original["length"]
                    or original["dataset_to_index"]-original["dataset_from_index"] != row["length"]):
                raise ValueError("Audited episode length differs from native source metadata")
        metadata = {name: [row[name] for row in rows] for name in columns}
        spans, starts, _ = namespace[method.name](metadata, [row["episode"] for row in selected], 16)
        actual.extend((subset, int(episode)) for episode, _, _ in spans)
        sliding_starts += starts
    if actual != [(row["subset"], row["episode"]) for row in eligible]:
        raise ValueError("Preparation eligibility differs from the actual official span constructor")
    return {"source": str(source.relative_to(root)), "source_sha256": _digest(source),
            "method": method.name, "chunk_length": 16, "exact_eligible_ids_equal": True,
            "selected_episodes": len(selection["episodes"]), "eligible_episodes": len(eligible),
            "short_episodes": short, "official_overlapping_window_starts": sliding_starts,
            "replay_protocol": "all nonoverlapping complete16-action chunks per eligible episode, not every sliding start"}


def prepare_holdout(root, dataset, selection_path, config_path, output):
    """Resumable preparation, no smaller fallback cohort or padded final chunk.

    The completion plan appears only after every declared case is prepared.
    A partial case directory is deliberately not silently replaced on restart.
    """
    from evaluation.planning.registry import project_path
    root = Path(root).resolve()
    dataset, selection_path, config_path, output = [project_path(root, str(p))
        for p in (dataset, selection_path, config_path, output)]
    selection = json.loads(selection_path.read_text())
    validate_selection(selection, dataset)
    eligible, short = partition_selection(selection)
    if not eligible:
        raise ValueError("No complete native window; retain the selection audit, do not invent inputs")
    eligibility = native_eligibility(root, dataset, selection)
    checks = SnapshotChecks(dataset)  # Requires the complete download marker.
    frozen = {"selection_sha256": _digest(selection_path), "config_sha256": _digest(config_path),
              "snapshot_manifest_sha256": checks.manifest_sha256, "dataset": str(dataset),
              "preparation_code_sha256": {str(path.relative_to(root)): _digest(path) for path in (
                  root/"evaluation/planning/cosmos.py", root/"evaluation/datasets/cosmos_reference.py",
                  root/"evaluation/datasets/droid.py", root/"evaluation/datasets/video.py",
                  root/"adapters/actions/droid.py", root/"evaluation/execution/transport.py")},
              "native_eligibility": eligibility,
              "protocol": f"all2157 selected episodes accounted; all{len(eligible)} native-eligible episodes, all complete16-transition AR chunks; seed0"}
    config = json.loads(config_path.read_text())
    if config.get("seed") != 0 or config.get("kwargs", {}).get("seed") != 0:
        raise ValueError("Frozen initial full holdout protocol requires seed0")
    with asset_destination_lock(root, output):
        output.mkdir(parents=True, exist_ok=True)
        journal = output/"progress.json"
        if journal.exists():
            progress = json.loads(journal.read_text())
            if progress.get("frozen") != frozen:
                raise ValueError("Cannot resume preparation with changed inputs")
        else:
            if any(output.iterdir()):
                raise ValueError("New preparation directory must be empty")
            progress = {"schema_version": 1, "frozen": frozen, "status": "preparing",
                        "selected_episodes": 2157, "expected_cases": len(eligible),
                        "native_ineligible_episodes": short, "completed": {}}
            publish_json_exclusive(journal, progress)
        cases, total_chunks = [], 0
        unscored_tail = sum(row["available_transitions"] for row in short)
        try:
            for row in eligible:
                identifier = row["id"]+"_seed0"
                target = output/"cases"/identifier
                if identifier in progress["completed"]:
                    record = progress["completed"][identifier]
                    if any(_digest(target/name) != sha for name, sha in record["sha256"].items()):
                        raise ValueError(f"Previously prepared evidence changed: {identifier}")
                    count = record["chunks"]
                else:
                    result = prepare_native_case(root, dataset/row["subset"], config_path, target,
                        episode=row["episode"], chunks="all", _checks=checks,
                        _cohort={"subset": row["subset"], "selection_sha256": frozen["selection_sha256"],
                                 "membership": "reconstructed20k_loader_holdout_not_scene_disjoint"})
                    count = result["chunks"]
                    if count != row["complete16_action_chunks"] or result["source_length"] != row["length"]:
                        raise ValueError("Prepared episode length differs from frozen selection")
                    progress["completed"][identifier] = {"chunks": count, "sha256": {
                        name: _digest(target/name) for name in ("input.npz", "reference.npz", "config.json")}}
                    progress["status"] = "preparing"
                    progress.pop("error", None)
                    replace_owned_json(journal, progress)
                total_chunks += count
                unscored_tail += row["length"]-1-count*16
                cases.append({"id": identifier, "prepared": str(target.relative_to(root))})
            if (len(cases) != len(eligible) or len(cases)+len(short) != 2157
                    or set(progress["completed"]) != {c["id"] for c in cases}):
                raise ValueError("Full holdout preparation membership mismatch")
            if not (output/"plan.json").exists():
                publish_json_exclusive(output/"plan.json", {"schema_version": 1, "scope": frozen["protocol"],
                    "cases": cases, "unavailable": {"scene_disjointness": "not established by episode split",
                    "native_short_episodes": short,
                    "incomplete_tails": "explicitly unscored; no fabricated future or action padding"}})
            coverage = {"schema_version": 1, **frozen, "selected_episodes": 2157,
                        "expected_cases": len(eligible), "prepared_cases": len(cases),
                        "native_ineligible_episodes": short, "all_selected_episodes_accounted": True,
                        "world_model_calls_planned": total_chunks, "future_frames_planned": total_chunks*16,
                        "unscored_tail_transitions": unscored_tail, "model_inference_performed": False,
                        "full_test_set_completed": False}
            if not (output/"coverage.json").exists():
                publish_json_exclusive(output/"coverage.json", coverage)
            progress["status"] = "prepared_not_evaluated"
            replace_owned_json(journal, progress)
            return {"plan": str(output/"plan.json"), **coverage}
        except BaseException as exc:
            progress.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error=f"{type(exc).__name__}: {exc}")
            replace_owned_json(journal, progress)
            raise
