"""Evidence-bound native-domain summaries, never a cross-domain leaderboard."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.execution.reliability import clean_worker_exit
from evaluation.reporting.task_scores import _digest
from evaluation.planning.registry import project_path
from vlaworldeval.atomic_json import publish_json_exclusive
from evaluation.reporting.motion import episode_bootstrap


def admitted_config(root, summary_path, report):
    """One config evidence check shared by pixel and motion reporting."""
    config_path = Path(summary_path).parent/"config.json"
    config = json.loads(config_path.read_text())
    if _digest(config_path) != report["config_sha256"]:
        binding = config_path.parent/"config_provenance.json"
        if not binding.is_file():
            raise ValueError("Replay configuration changed or needs explicit legacy serialization provenance")
        evidence = json.loads(binding.read_text())
        source = project_path(Path(root), evidence["source_config"])
        if (_digest(source) != report["config_sha256"] or _digest(config_path) != evidence["saved_config_sha256"]
                or json.loads(source.read_text()) != config):
            raise ValueError("Legacy config binding does not match original admitted content")
    return config


def scientific_protocol(config):
    """Separate checkpoints/protocols, not repeated seeds or allocator budgets."""
    value = {key: item for key, item in config.items()
             if key not in {"gpus", "memory_gib", "worker_options", "seed"}}
    if "kwargs" in value:
        value["kwargs"] = {key: item for key, item in value["kwargs"].items()
                           if key not in {"runtime_dir", "seed"}}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, value


def summarize_native_references(root, directory, output, *, plan_path=None, metrics_directory=None,
                               additional_directories=()):
    root = Path(root).resolve()
    directory, output = project_path(root, str(directory)), project_path(root, str(output))
    additional_directories = tuple(project_path(root, str(p)) for p in additional_directories)
    if output.exists():
        raise FileExistsError(output)
    plan = None
    admitted = None
    plan_sha256 = None
    budget_selection = None
    metric_index, metric_journal = None, None
    if metrics_directory is not None and plan_path is None:
        raise ValueError("A version-local metric journal requires the exact --plan")
    if plan_path is not None:
        from evaluation.reporting.checkpoint_metrics import select_replays
        plan_path = project_path(root, str(plan_path))
        plan_sha256 = _digest(plan_path)
        plan = json.loads(plan_path.read_text())
        from evaluation.planning.native_subset import audit_reference_subset
        budget_selection = audit_reference_subset(root, plan_path, plan)
        selected, _ = select_replays(root, directory, plan, additional_directories=additional_directories)
        if metrics_directory is not None:
            from evaluation.reporting.metric_store import load_metric_store
            metric_index, metric_journal = load_metric_store(root, metrics_directory, plan_path, plan, selected)
        admitted = {path.resolve(): _digest(path) for path, _ in selected.values()}
        # A pending inference is valid only against a real prepared case; a
        # nonexistent reference is not a verified full evaluation manifest.
        for case in plan["cases"]:
            if case["id"] not in selected:
                prepared = project_path(root, case["prepared"])
                for filename in ("input.npz", "reference.npz", "config.json"):
                    _digest(prepared/filename)
    rows, motion_rows, event_rows, excluded = [], [], [], []
    protocols, perceptual_implementations = {}, {}
    from evaluation.reporting.checkpoint_metrics import replay_summary_paths
    for path in replay_summary_paths(root, directory, additional_directories):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Replay summary leaves the project")
        if admitted is not None and path.resolve() not in admitted:
            excluded.append({"path": str(path), "reason": "Not a clean admitted replay from the exact plan"})
            continue
        if admitted is not None and _digest(path) != admitted[path.resolve()]:
            raise ValueError("Admitted replay summary changed during reporting")
        report = json.loads(path.read_text())
        if report.get("status") != "completed" or not clean_worker_exit(report.get("cleanup")):
            excluded.append({"path": str(path), "reason": "Inference or cleanup failed; inspect original evidence",
                             "status": report.get("status"), "error": report.get("error")})
            continue
        if not report.get("reference_evaluation"):
            excluded.append({"path": str(path), "reason": "No paired reference scores; not counted"})
            continue
        config = admitted_config(root, path, report)
        protocol, canonical = scientific_protocol(config)
        protocols[protocol] = canonical
        for name, digest in report["files"].items():
            if _digest(path.parent / name) != digest:
                raise ValueError("Replay output changed after scoring")
        source = report["reference_evaluation"]["reference_source"]
        perceptual = (path.parent/"perceptual.json" if metric_index is None
                      else metric_index[path.parent.name]["lpips"])
        learned = json.loads(perceptual.read_text()) if perceptual.exists() else None
        if learned and (learned["summary_sha256"] != _digest(path)
                        or learned["reference_sha256"] != report["reference_evaluation"]["reference_sha256"]):
            raise ValueError("Perceptual scores are not bound to the same replay")
        perceptual_version = None
        if learned:
            implementation = learned.get("implementation_sha256")
            if not implementation:
                if plan is not None:
                    raise ValueError("Unversioned perceptual metric cannot verify a planned report")
                perceptual_version = "legacy_unversioned"
            else:
                perceptual_version = hashlib.sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest()
            perceptual_implementations[perceptual_version] = implementation
        motion_path = (path.parent/"motion/motion.json" if metric_index is None
                       else metric_index[path.parent.name].get("motion"))
        observed = json.loads(motion_path.read_text()) if motion_path is not None and motion_path.exists() else None
        if observed:
            annotation = project_path(root, observed["annotations_path"])
            if (observed["summary_sha256"] != _digest(path)
                    or observed["reference_sha256"] != report["reference_evaluation"]["reference_sha256"]
                    or observed["annotations_sha256"] != _digest(annotation)):
                raise ValueError("Motion scores are not bound to the same replay observations")
            for view, entities in observed["tracks"].items():
                if view not in report["reference_evaluation"]["metrics"]:
                    raise ValueError("Motion camera is not part of the original replay")
                for entity, scores in entities.items():
                    motion_rows.append({"case": path.parent.name, "adapter": config["adapter"],
                        "protocol_sha256": protocol,
                        "domain": source["dataset_domain"], "split": source["split"], "episode": source["episode"],
                        "condition_kind": source["condition_kind"], "view": view, "entity": entity,
                        "future_frames": scores["total_future_frames"], "fps": config["video"]["fps"],
                        "seed": config.get("seed", 0), **scores,
                        "evaluator": observed["evaluator"]["identity"],
                        "human_review_status": observed["evaluator"].get("human_review_status", "not_declared"),
                        "motion_sha256": _digest(motion_path)})
            if observed.get("events"):
                event_rows.append({"case": path.parent.name, "episode": source["episode"],
                                   "evaluator": observed["evaluator"]["identity"], **observed["events"]})
        for view, values in report["reference_evaluation"]["metrics"].items():
            rows.append({"case": path.parent.name, "adapter": config["adapter"],
                "protocol_sha256": protocol,
                "perceptual_implementation": perceptual_version,
                "domain": source["dataset_domain"], "split": source["split"], "episode": source["episode"],
                "condition_kind": source["condition_kind"], "view": view, "frames": values["frames_scored"],
                "seed": config.get("seed", 0),
                "fps": config["video"]["fps"], "mse_01": values["mse_01"],
                "mean_frame_psnr_db": values["mean_frame_psnr_db"],
                "psnr_db_from_mean_mse": values["psnr_db_from_mean_mse"],
                "ssim": values["ssim_rgb_gaussian11_sigma1p5_population"],
                "improvement_over_frozen": values["relative_mse_improvement_over_frozen"],
                "lpips_vgg_decoded": None if learned is None else learned["views"][view]["lpips_vgg"],
                "summary": str(path), "summary_sha256": _digest(path)})
    grouped = {}
    for row in rows:
        # Never average short-window and long-AR errors into a single score.
        key = (row["adapter"], row["domain"], row["split"], row["condition_kind"], row["frames"], row["fps"],
               row["protocol_sha256"], row["perceptual_implementation"])
        grouped.setdefault(key, {}).setdefault(str(row["episode"]), []).append(row)
    groups = []
    for key, episodes in grouped.items():
        item = dict(zip(("adapter", "domain", "split", "condition_kind", "future_frames", "fps",
                         "protocol_sha256", "perceptual_implementation"), key))
        item.update(episodes=len(episodes), cases=len({r["case"] for e in episodes.values() for r in e}),
                    seeds=sorted({r["seed"] for e in episodes.values() for r in e}),
                    uncertainty="insufficient_sample_count_for_reliability_claim", paper_test_set_reproduced=False)
        for metric in ("mse_01", "mean_frame_psnr_db", "ssim", "improvement_over_frozen", "lpips_vgg_decoded"):
            means = [float(np.mean([r[metric] for r in group]))
                     for group in episodes.values() if all(r[metric] is not None for r in group)]
            item[metric] = float(np.mean(means)) if len(means) == len(episodes) else None
        item["episode_uncertainty"] = {metric: episode_bootstrap([
            (episode, row[metric]) for episode, group in episodes.items() for row in group if row[metric] is not None])
            for metric in ("mse_01", "ssim", "lpips_vgg_decoded")}
        groups.append(item)
    motion_groups = {}
    for row in motion_rows:
        key = tuple(row[k] for k in ("adapter", "domain", "split", "condition_kind", "future_frames", "fps", "entity", "evaluator", "protocol_sha256"))
        motion_groups.setdefault(key, []).append(row)
    motion_summaries = []
    for key, members in motion_groups.items():
        item = dict(zip(("adapter", "domain", "split", "condition_kind", "future_frames", "fps", "entity", "evaluator", "protocol_sha256"), key))
        item.update(scored_observations=sum(r["status"] == "scored" for r in members),
                    unscored_observations=sum(r["status"] != "scored" for r in members),
                    mean_paired_coverage=float(np.mean([r["paired_coverage"] for r in members])),
                    human_review_status=sorted({r["human_review_status"] for r in members}))
        for metric in ("ade", "fde", "symmetric_hausdorff", "dtw_mean_path_distance", "speed_wasserstein", "relative_ade_improvement_over_frozen"):
            item[metric] = episode_bootstrap([(str(r["episode"]), r[metric]) for r in members if r[metric] is not None])
        motion_summaries.append(item)
    if plan_path is not None and _digest(plan_path) != plan_sha256:
        raise ValueError("Frozen plan changed during reporting")
    if budget_selection is not None and (
            _digest(plan_path.parent/"selection.json") != budget_selection["selection_sha256"]
            or _digest(project_path(root, budget_selection["source_plan"])) != budget_selection["source_plan_sha256"]):
        raise ValueError("Budget population/selection changed during reporting")
    if metric_journal is not None and _digest(Path(metric_journal["path"])) != metric_journal["sha256"]:
        raise ValueError("Scoring journal changed during reporting")
    completed = sorted({r["case"] for r in rows})
    planned = [] if plan is None else [case["id"] for case in plan["cases"]]
    result = {"schema_version": 1, "status": "completed", "rows": rows, "groups": groups,
              "replay_directories": [str(directory), *map(str, additional_directories)],
              "motion_rows": motion_rows, "motion_groups": motion_summaries, "event_rows": event_rows,
              "excluded": excluded, "cross_domain_ranking_valid": False, "physical_action_tracking_verified": False,
              "completed_cases": completed, "pending_or_failed_cases": sorted(set(planned)-set(completed)),
              "plan_coverage_verified": plan is not None,
              "all_planned_replays_completed": plan is not None and set(planned) == set(completed),
              "source_plan_sha256": plan_sha256, "protocols": protocols,
              "evaluation_scope": None if plan is None else plan.get("scope"),
              "budget_selection": budget_selection,
              "full_test_set_completed": False,
              "perceptual_implementations": perceptual_implementations,
              "scoring_journal": metric_journal,
              "perceptual_scored_cases": sorted({row["case"] for row in rows if row["lpips_vgg_decoded"] is not None}),
              "unavailable": {} if plan is None else plan["unavailable"],
              "aggregation": "views/cases macro-averaged within source episode, then episodes; no frame-wise pseudo-replication",
              "primary_domain": "raw_rgb_before_encoding", "lpips_domain": "decoded_prediction_mp4_vs_raw_reference"}
    output.mkdir(parents=True, exist_ok=False)
    publish_json_exclusive(output / "report.json", result)
    with (output / "per_view.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    if motion_rows:
        with (output/"motion_per_entity.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(motion_rows[0]))
            writer.writeheader()
            writer.writerows(motion_rows)
    return result
