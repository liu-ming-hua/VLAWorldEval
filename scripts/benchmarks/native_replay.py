"""Prepare an audited native input/reference pair; inference uses `vlaworldeval replay`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["dreamdojo", "iws", "irasim", "dreamtrue", "worldgym", "ctrl_world", "cosmos20k"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--chunks", type=lambda value: "all" if value == "all" else int(value), default=1,
        help="positive count; Cosmos20k also accepts all complete16-action chunks (unpaired/incomplete tail explicitly excluded)")
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--full-iws-validation", action="store_true", help="prepare all10 official PushT validation episodes, not mini/debug")
    parser.add_argument("--full-dreamdojo-inlab", action="store_true", help="prepare the complete released100-sample49-frame protocol")
    parser.add_argument("--audit-cosmos-holdout", action="store_true", help="reconstruct2157 held-out episode IDs and audit local payload availability, no inference")
    parser.add_argument("--compare-dreamdojo-selection", action="store_true", help="check all100 selected identities against pinned upstream indexing methods")
    parser.add_argument("--extract-irasim-annotations", action="store_true", help="extract and verify complete4562/5243 test selection from a verified archive prefix; no inference")
    parser.add_argument("--extract-irasim-assets", action="store_true", help="extract all frozen test RGB/latent files from a complete verified multipart archive")
    parser.add_argument("--probe-cosmos-protocol", action="store_true", help="CPU-only local conversion/source parity probe; not a prepared or scored evaluation case")
    parser.add_argument("--full-cosmos-holdout", action="store_true", help="prepare all2157 frozen20k-EMA holdout episodes with every complete16-action chunk")
    parser.add_argument("--selection", type=Path, help="frozen Cosmos holdout audit required for full preparation")
    parser.add_argument("--cosmos-subset-plan", type=Path, help="frozen400-case plan; preserve all native AR chunks")
    parser.add_argument("--prepared-cache", type=Path, help="immutable prior native preparation progress.json")
    parser.add_argument("--compare-irasim-selection", action="store_true", help="CPU-only comparison of all5243 frozen windows with actual upstream indexing")
    parser.add_argument("--full-irasim-short", action="store_true", help="prepare all5243 official Language-Table short test windows; long trajectories remain separate")
    parser.add_argument("--full-irasim-long", action="store_true", help="prepare all4562 complete Language-Table trajectories, with native padded tails excluded from scoring")
    parser.add_argument("--compare-irasim-long", action="store_true", help="compare all4562 trajectories against upstream pure tail-action statements; no model inference")
    parser.add_argument("--full-worldgym-released-test", action="store_true", help="all released sample-test actions,1 episode, explicitly NOT full OXE paper validation")
    parser.add_argument("--full-ctrl-released-validation", action="store_true", help="all97 published droid_subset validation windows, NOT paper256 clips; --dataset is the pinned source checkout")
    parser.add_argument("--full-ctrl-released-long", action="store_true", help="both entire released validation episodes; clamp padded tail conditions and exclude padded predictions")
    parser.add_argument("--metadata", type=Path, default=Path("datasets/dreamdojo/eval_metadata_v1"))
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(argv)
    if args.chunks == "all" and args.model != "cosmos20k":
        raise ValueError("All-chunk resolution is currently declared only for Cosmos20k")
    from vlaworldeval.cache import runtime_cache_environment
    import os
    os.environ.update(runtime_cache_environment(ROOT), HF_HUB_OFFLINE="1")
    if sum((args.full_iws_validation, args.full_dreamdojo_inlab, args.audit_cosmos_holdout,
            args.compare_dreamdojo_selection, args.extract_irasim_annotations, args.extract_irasim_assets,
            args.probe_cosmos_protocol, args.full_cosmos_holdout,
            args.compare_irasim_selection, args.full_irasim_short, args.full_irasim_long, args.compare_irasim_long,
            args.full_worldgym_released_test, args.full_ctrl_released_validation, args.full_ctrl_released_long,
            bool(args.cosmos_subset_plan))) > 1:
        raise ValueError("Select one complete native-domain protocol")
    if args.compare_irasim_selection or args.compare_irasim_long:
        if args.model != "irasim":
            raise ValueError("IRASim selection comparison requires --model irasim")
        from evaluation.datasets.irasim_reference import short_selection, compare_long_protocol
        from evaluation.planning.registry import project_path
        from vlaworldeval.atomic_json import publish_json_exclusive
        output = project_path(ROOT, str(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        compare = compare_long_protocol if args.compare_irasim_long else short_selection
        result = compare(project_path(ROOT, str(args.metadata)), ROOT/"third_party/irasim")
        publish_json_exclusive(output, result)
        keys = (("test_episodes", "native_model_calls", "all_padded_scaled_actions_exact_equal", "model_inference_performed")
                if args.compare_irasim_long else ("test_episodes", "official_short_clips", "all_indices_equal", "model_inference_performed"))
        print(json.dumps({key: result[key] for key in keys}))
        return
    if args.probe_cosmos_protocol:
        if args.model != "cosmos20k":
            raise ValueError("Native DROID20k protocol probe requires --model cosmos20k")
        import torch
        torch.set_num_threads(4)
        from evaluation.datasets.cosmos_reference import convert_episode
        from evaluation.planning.registry import project_path
        from vlaworldeval.atomic_json import publish_json_exclusive
        output = project_path(ROOT, str(args.output))
        if output.exists():
            raise FileExistsError(output)
        packet, frames = convert_episode(project_path(ROOT, str(args.dataset)), args.episode, args.start,
                                          args.chunks, ROOT/"third_party/cosmos-framework")
        report = {"schema_version": 1, "kind": "protocol_probe_not_evaluation", "source": packet["source"],
            "action_shape": list(packet["actions"].shape), "reference_shape": list(frames.shape),
            "source_snapshot_verification": "not performed by this protocol-only probe",
            "cuda_initialized": torch.cuda.is_initialized(), "model_inference_performed": False,
            "prepared_evaluation_case": False, "full_test_set_completed": False}
        output.parent.mkdir(parents=True, exist_ok=True)
        publish_json_exclusive(output, report)
        print(json.dumps({key: report[key] for key in ("kind", "action_shape", "reference_shape", "cuda_initialized")}))
        return
    if args.extract_irasim_assets:
        if args.model != "irasim":
            raise ValueError("Native test assets require --model irasim")
        from evaluation.datasets.native_archives import extract_irasim_test_assets
        print(json.dumps(extract_irasim_test_assets(ROOT, args.dataset, args.metadata, args.output)))
        return
    if args.extract_irasim_annotations:
        if args.model != "irasim":
            raise ValueError("Native test annotations require --model irasim")
        from evaluation.planning.checkpoint import extract_irasim_test_annotations
        print(json.dumps(extract_irasim_test_annotations(args.dataset, args.output)))
        return
    if args.compare_dreamdojo_selection:
        if args.model != "dreamdojo":
            raise ValueError("Official In-lab indexing comparison requires --model dreamdojo")
        from evaluation.planning.checkpoint import compare_dreamdojo_official_selection
        print(json.dumps(compare_dreamdojo_official_selection(args.metadata, args.output)))
        return
    if args.audit_cosmos_holdout:
        if args.model != "cosmos20k":
            raise ValueError("Holdout reconstruction is documented only for --model cosmos20k")
        from evaluation.planning.checkpoint import audit_cosmos_droid_holdout
        print(json.dumps(audit_cosmos_droid_holdout(args.dataset, args.output)))
        return
    if args.config is None:
        raise ValueError("Preparation requires --config and a supported native conversion")
    if args.cosmos_subset_plan:
        if args.model != "cosmos20k" or args.selection is None or args.prepared_cache is None:
            raise ValueError("Subset requires cosmos20k, frozen --selection and --prepared-cache")
        import torch
        torch.set_num_threads(4)
        from evaluation.planning.native_subset import prepare_cosmos_subset
        print(json.dumps(prepare_cosmos_subset(ROOT, args.cosmos_subset_plan, args.dataset,
            args.selection, args.config, args.prepared_cache, args.output)))
        return
    if args.full_ctrl_released_validation or args.full_ctrl_released_long:
        if args.model != "ctrl_world" or args.start != 0 or args.episode != 0 or args.chunks != 1 or (args.gpu is not None and not args.full_ctrl_released_long) or args.seeds is not None:
            raise ValueError("Full released Ctrl validation freezes all97 windows, seed0 and config GPUs")
        from evaluation.datasets.ctrl_world import prepare_released_validation
        print(json.dumps(prepare_released_validation(ROOT, args.dataset, args.config, args.output,
            mode="long" if args.full_ctrl_released_long else "short", gpu=args.gpu)))
        return
    if args.full_worldgym_released_test:
        if args.model != "worldgym" or args.start != 0 or args.episode != 0 or args.chunks != 1 or args.gpu is not None:
            raise ValueError("Released WorldGym test uses all published test IDs/actions; set GPUs in config")
        from evaluation.planning.checkpoint import prepare_worldgym_released_test
        print(json.dumps(prepare_worldgym_released_test(ROOT, args.dataset, args.config, args.output,
            seeds=args.seeds or [0, 1, 2])))
        return
    if args.full_irasim_short or args.full_irasim_long:
        if args.model != "irasim" or args.gpu is not None or args.seeds is not None or args.start != 0 or args.episode != 0 or args.chunks != 1:
            raise ValueError("Full IRASim freezes every official case, seed3407 and round-robin GPU0..3")
        import torch
        torch.set_num_threads(4)
        from evaluation.datasets.irasim_reference import prepare_full_test
        print(json.dumps(prepare_full_test(ROOT, args.dataset, args.metadata, args.config, args.output,
            mode="long" if args.full_irasim_long else "short")))
        return
    if args.full_cosmos_holdout:
        if args.model != "cosmos20k" or args.selection is None or args.gpu is not None:
            raise ValueError("Full holdout requires --model cosmos20k --selection; use multi-rank config GPUs")
        if args.seeds is not None or args.start != 0 or args.chunks != 1 or args.episode != 0:
            raise ValueError("Full holdout freezes seed0, every episode, start0 and all complete chunks")
        import torch
        torch.set_num_threads(4)
        from evaluation.planning.cosmos import prepare_holdout
        print(json.dumps(prepare_holdout(ROOT, args.dataset, args.selection, args.config, args.output)))
        return
    if args.model == "cosmos20k":
        if args.gpu is not None:
            raise ValueError("Set Cosmos multi-rank GPUs in its config rather than collapsing them with --gpu")
        from evaluation.datasets.cosmos_reference import prepare_native_case
        print(json.dumps(prepare_native_case(ROOT, args.dataset, args.config, args.output,
            episode=args.episode, start=args.start, chunks=args.chunks)))
        return
    if args.full_dreamdojo_inlab:
        if args.model != "dreamdojo":
            raise ValueError("Full In-lab selection requires --model dreamdojo")
        from evaluation.planning.checkpoint import prepare_dreamdojo_eval100
        print(json.dumps(prepare_dreamdojo_eval100(ROOT, args.dataset, args.metadata, args.config,
            args.output, seeds=args.seeds or [0])))
        return
    if args.full_iws_validation:
        if args.model != "iws":
            raise ValueError("Full IWS validation requires --model iws")
        from evaluation.planning.checkpoint import prepare_iws_validation
        print(json.dumps(prepare_iws_validation(ROOT, args.dataset, args.config, args.output, seeds=args.seeds or [0, 1, 2])))
        return
    from evaluation.datasets.native_reference import prepare_case
    print(json.dumps(prepare_case(args.model, args.dataset, args.config, args.output,
        start=args.start, chunks=args.chunks, episode=args.episode, gpu=args.gpu), indent=2))


if __name__ == "__main__":
    main()
