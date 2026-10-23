"""Command-line arguments; no model imports or execution."""

import argparse
from pathlib import Path


def build_parser(root):
    parser = argparse.ArgumentParser(
        prog="vlaworldeval",
        description="Validate, execute and inspect VLA × world-model experiment suites.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="show original traceback for command errors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    policy_parser = subparsers.add_parser("policy-check", help="one native DROID policy call; not a closed-loop score")
    policy_parser.add_argument("--model", choices=["cosmos_droid", "dreamzero_droid"], required=True)
    policy_parser.add_argument("--output", type=Path, required=True)
    policy_parser.add_argument("--dataset", type=Path, default=Path("datasets/cosmos3_droid/success"))
    policy_parser.add_argument("--episode", type=int, default=350)
    policy_parser.add_argument("--start", type=int, default=15)
    policy_parser.add_argument("--gpus", type=int, nargs="+")
    policy_parser.add_argument("--memory-gib", type=float)
    policy_parser.add_argument("--attention-backend", choices=["TE", "torch"])
    policy_parser.add_argument("--execute", action="store_true", help="opt in to full asset verification and GPU inference")
    policy_parser.add_argument("--runtime-check", action="store_true", help="also check CPU imports in the isolated runtime; no GPU/model loading")
    study_parser = subparsers.add_parser("droid-study", help="fixed DROID positive-control and gated policy queues")
    study_parser.add_argument("operation", choices=["prepare", "run", "status", "compare"])
    study_parser.add_argument("--directory", type=Path, required=True)
    study_parser.add_argument("--config", type=Path, default=Path("configs/studies/droid_positive_controls_v1.json"))
    study_parser.add_argument("--phase", choices=["controls", "policies", "all"], default="controls")
    study_parser.add_argument("--watch", action="store_true", help="wait for GPU/review changes, never evict processes")
    study_parser.add_argument("--output", type=Path, help="fresh comparison output directory, required for compare")
    init_parser = subparsers.add_parser(
        "init", help="create a portable empty asset workspace; no downloads or model installs"
    )
    init_parser.add_argument("--directory", type=Path, required=True)
    asset_parser = subparsers.add_parser(
        "verify-assets", help="offline SHA256 verification of a downloaded HF snapshot"
    )
    asset_parser.add_argument("--directory", type=Path, required=True)
    configure_parser = subparsers.add_parser(
        "configure", help="make a validated suite for your GPUs without overwriting templates"
    )
    configure_parser.add_argument("--config", type=Path, required=True)
    configure_parser.add_argument("--output", type=Path, required=True)
    configure_parser.add_argument("--vla-gpus", type=int, nargs="+")
    configure_parser.add_argument("--wm-gpus", type=int, nargs="+")
    configure_parser.add_argument("--budget-gib", type=float)
    configure_parser.add_argument("--vla-budget-gib", type=float)
    configure_parser.add_argument("--chunks", type=int)
    subparsers.add_parser("models", help="list catalogued VLA and world-model adapters")
    cleanup_parser = subparsers.add_parser(
        "cleanup-results",
        help="preview retention policy; --apply archives old generated outputs without deleting",
    )
    cleanup_parser.add_argument("--apply", action="store_true")
    regression_parser = subparsers.add_parser(
        "regression-report", help="re-audit executed evaluate reports and decode videos; no models"
    )
    regression_parser.add_argument("--report", type=Path, action="append", required=True)
    subparsers.add_parser("environments", help="show registered runtime environments")
    env_parser = subparsers.add_parser(
        "env-audit", help="inspect actual isolated package versions; optionally detect drift"
    )
    env_parser.add_argument("--environment", action="append", default=[])
    env_parser.add_argument("--baseline", type=Path)
    env_parser.add_argument(
        "--dependencies", action="store_true", help="also run offline uv pip check; never installs"
    )
    score_parser = subparsers.add_parser(
        "scores", help="validate evidence-bound local outcomes and matched rankings; no cloud calls"
    )
    score_parser.add_argument("--annotations", type=Path, required=True)
    replay_parser = subparsers.add_parser(
        "replay", help="native-protocol world-model-only autoregressive replay"
    )
    replay_parser.add_argument("--config", type=Path, required=True)
    replay_parser.add_argument("--input", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--admit-only", action="store_true")
    replay_parser.add_argument("--reference", type=Path,
        help="bound recorded-future packet, scored after inference only; never passed to the model")
    native_study = subparsers.add_parser("native-study", help="freeze/run recorded-action reference cases using shared workers")
    native_study.add_argument("operation", choices=["stream", "batch", "subset", "prepare", "run", "status", "retry", "repartition", "remainder", "score"])
    native_study.add_argument("--directory", type=Path, required=True)
    native_study.add_argument("--plan", type=Path)
    native_study.add_argument("--case-limit", type=int, default=400, help="subset only: maximum native cases; never duplicate to fill")
    native_study.add_argument("--sample-seed", type=int, default=42, help="subset only: outcome-independent cohort sampling seed")
    native_study.add_argument("--prepared-progress", type=Path, help="batch only: checksum-bound preparation progress")
    native_study.add_argument("--batch-limit", type=int, default=32, help="batch only: maximum newly ready cases")
    native_study.add_argument("--prepared-manifest", type=Path, help="prepare only: validate transferred ready batch against frozen parent cohort")
    native_study.add_argument("--batch-directory", type=Path, help="stream only: immutable delivered ready batches")
    native_study.add_argument("--previous-directory", type=Path, help="stream only: reuse validated completed native replay results")
    native_study.add_argument("--watch", action="store_true")
    native_study.add_argument("--exclusive-gpu", action="store_true",
        help="run only: one cooperating native study per GPU; waiting queues load no model")
    native_study.add_argument("--reuse-worker", action="store_true", help="run only: keep one supported WM resident across episodes; certify results after clean session close")
    native_study.add_argument("--gpu", type=int,
        help="run only: explicitly relocate single-GPU jobs; preserves frozen scientific config and records device provenance")
    native_study.add_argument("--source-directory", type=Path, action="append", default=[],
        help="repartition only: stopped source study, repeat for disjoint shards")
    native_study.add_argument("--gpus", type=int, nargs="+",
        help="repartition: authorized GPU pool; run: explicit rank-to-physical-GPU map of unchanged size")
    native_study.add_argument("--additional-directory", type=Path, action="append", default=[],
        help="score only: additional disjoint replay root after GPU migration; repeat as needed")
    native_study.add_argument("--memory-gib", type=float, help="run only: explicit allocator-budget override, journalled separately; does not change scientific inputs")
    native_study.add_argument("--output", type=Path, help="new retry directory; original failed attempts are preserved")
    native_study.add_argument("--workers", type=int, default=2, help="bounded CPU evaluator concurrency for score")
    native_study.add_argument("--metric-timeout-seconds", type=float, default=3600,
        help="score only: explicit per-metric subprocess deadline,1..604800 seconds; recorded in the journal")
    native_study.add_argument("--metric-gpus", type=int, nargs="+", default=[],
        help="score only: explicit GPU0..3 assignment, one evaluator per GPU; default CPU")
    native_study.add_argument("--metric-frame-batch-size", type=int, default=1,
        help="score only: LPIPS future-frame batch1..32; no resizing or temporal subsampling")
    native_study.add_argument("--motion-provider", choices=["ewmbench", "pusht_color", "none"], default="none",
        help="none scores LPIPS only and explicitly records absent motion evaluation; domain observers require calibration")
    reference_report = subparsers.add_parser("replay-report", help="group paired-reference metrics without cross-domain ranking")
    reference_report.add_argument("--directory", type=Path, required=True)
    reference_report.add_argument("--additional-directory", type=Path, action="append", default=[],
        help="additional disjoint replay root; does not allow choosing between duplicate clean cases")
    reference_report.add_argument("--output", type=Path, required=True)
    reference_report.add_argument("--plan", type=Path, help="also report pending/unavailable planned cases")
    reference_report.add_argument("--metrics", type=Path, help="completed version-local scoring directory; requires --plan, never falls back to legacy sidecars")
    motion_parser = subparsers.add_parser("motion-eval", help="evidence-bound 2-D tracks/events; no VLA or world-model calls")
    motion_parser.add_argument("--summary", type=Path, required=True)
    motion_parser.add_argument("--reference", type=Path, required=True)
    motion_parser.add_argument("--output", type=Path, required=True)
    motion_parser.add_argument("--annotations", type=Path, help="score existing bound observations without loading detector")
    motion_parser.add_argument("--provider", choices=["ewmbench", "pusht_color"], default="ewmbench")
    replay_parser.add_argument("--image-anchors", type=Path,
        help="diagnostic only: recorded images at chunk boundaries, never a closed-loop score")
    prepare_parser = subparsers.add_parser(
        "prepare-replay", help="prepare Ctrl-World native replay packet from local DROID"
    )
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--episode", type=int, default=0)
    prepare_parser.add_argument("--start", type=int, default=0)
    prepare_parser.add_argument("--chunks", type=int, default=20)
    subparsers.add_parser(
        "runtimes", help="list executable pair profiles and data loaders (no model loading)"
    )
    subparsers.add_parser(
        "capabilities",
        help="list explicit pair contracts and unmigrated catalogue entries; no inference",
    )
    validate_parser = subparsers.add_parser(
        "validate", help="validate declarations; optionally require local demo assets"
    )
    validate_parser.add_argument(
        "--assets", action="store_true", help="fail when local demo files are missing"
    )
    engineering_parser = subparsers.add_parser(
        "engineering-accept",
        help="execute fresh weight-free checkout acceptance; no downloads/models",
    )
    engineering_parser.add_argument("--output", type=Path, required=True)
    engineering_parser.add_argument("--timeout", type=float, default=180)
    doctor_parser = subparsers.add_parser(
        "doctor", help="read-only suite diagnostics; never loads models"
    )
    doctor_parser.add_argument("--config", type=Path, required=True)
    doctor_parser.add_argument("--experiment", action="append", default=[])
    doctor_parser.add_argument("--seed", type=int, action="append", default=[])
    doctor_parser.add_argument(
        "--data", action="store_true", help="check selected video/action windows"
    )
    doctor_parser.add_argument("--gpu", action="store_true", help="check live GPU memory budgets")
    doctor_parser.add_argument(
        "--json", action="store_true", help="print machine-readable diagnostic report"
    )
    acceptance_parser = subparsers.add_parser(
        "accept", help="staged integration acceptance; inference is explicit opt-in"
    )
    acceptance_parser.add_argument(
        "--matrix", type=Path, default=root / "configs/acceptance/framework_v1.json"
    )
    acceptance_parser.add_argument("--pair", action="append", default=[])
    acceptance_parser.add_argument(
        "--level", choices=["local", "preflight", "smoke", "long"], default="local"
    )
    acceptance_parser.add_argument("--execute", action="store_true")
    results_parser = subparsers.add_parser(
        "results", help="summarize executed suite attempts without loading models"
    )
    results_parser.add_argument("--root", type=Path, default=root / "results/experiments")
    inspect_parser = subparsers.add_parser(
        "inspect", help="read-only local video diagnostics; no models or task scores"
    )
    inspect_parser.add_argument("--artifact", type=Path, required=True)
    inspect_parser.add_argument("--near-static-mae", type=float, default=0.5)
    inspect_parser.add_argument("--jump-mae", type=float, default=35.0)
    demo_parser = subparsers.add_parser(
        "demo", help="CPU-only synthetic engineering example; no model weights"
    )
    demo_parser.add_argument("--output", type=Path, required=True)
    demo_parser.add_argument("--chunks", type=int, default=5)
    plan_parser = subparsers.add_parser(
        "plan", help="validate and expand an experiment suite; does not execute it"
    )
    plan_parser.add_argument(
        "--config", type=Path, default=root / "configs/experiments/droid_framework_v1.json"
    )
    run_parser = subparsers.add_parser(
        "run", help="admit and execute a suite using persistent model workers"
    )
    run_parser.add_argument(
        "--config", type=Path, default=root / "configs/experiments/droid_framework_v1.json"
    )
    run_parser.add_argument("--experiment", action="append", default=[])
    run_parser.add_argument("--seed", type=int, action="append", default=[])
    run_parser.add_argument("--admit-only", action="store_true")
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="preflight, execute and audit through one entry point; no auto-install/download",
    )
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--experiment", action="append", default=[])
    evaluate_parser.add_argument("--seed", type=int, action="append", default=[])
    evaluate_mode = evaluate_parser.add_mutually_exclusive_group(required=True)
    evaluate_mode.add_argument("--check-only", action="store_true")
    evaluate_mode.add_argument("--execute", action="store_true")
    return parser
