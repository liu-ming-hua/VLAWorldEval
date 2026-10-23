#!/usr/bin/env python3
"""Read-only DreamZero resource diagnosis; never loads weights onto a GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from adapters.environment import get_environment
    from evaluation.planning.registry import project_path
    from evaluation.validation.safetensors import inspect_sharded_tensors
    from vlaworldeval.atomic_json import publish_json_exclusive

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="query current GPU0–3 free memory; no CUDA initialization")
    parser.add_argument("--reserve-gib", type=float, default=8, help="conservative extra headroom, not a measured peak")
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        raise ValueError("reserve-gib must be finite and nonnegative")
    output = project_path(ROOT, str(args.output))
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / "third_party/dreamzero"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "ab790c198fbce33503358efbbd4187ce9a89adf3":
        raise ValueError("DreamZero resource contract requires the pinned upstream source revision")
    sources = {}
    for name in ("groot/vla/model/n1_5/sim_policy.py",
                 "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py"):
        actual = (source / name).read_bytes()
        original = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{name}"])
        if actual != original:
            raise ValueError(f"Resource contract cannot assume unreviewed loader changes: {name}")
        sources[name] = hashlib.sha256(actual).hexdigest()
    model = ROOT / "checkpoints/vla/dreamzero/droid"
    tensors = inspect_sharded_tensors(model)
    if set(tensors["dtype_bytes"]) != {"BF16"}:
        raise ValueError("This DreamZero resource profile only covers the BF16 release")
    for name in ("spiece.model", "tokenizer_config.json"):
        if not (ROOT / "checkpoints/vla/dreamzero/umt5-tokenizer" / name).is_file():
            raise FileNotFoundError(f"Missing local DreamZero tokenizer: {name}")
    environment = get_environment("vla.dreamzero")
    if not environment.exists:
        raise ValueError("Install the isolated environment with bash scripts/setup/dreamzero.sh")
    # No model/CUDA import: distinguish this package owner from Isaac-GR00T.
    package = subprocess.check_output([str(environment.python), "-c",
        "import importlib.util,json; s=importlib.util.find_spec('groot'); print(json.dumps({'groot':s.origin if s else None}))"], text=True)
    package = json.loads(package)
    if not package["groot"] or not Path(package["groot"]).resolve().is_relative_to(source):
        raise ValueError("DreamZero environment resolves a different groot package")
    payload_gib = tensors["tensor_bytes"] / 2**30
    report = {"schema_version": 1, "scope": "native_readiness_not_model_acceptance",
              "model_inference_performed": False, "asset_integrity_verified": False,
              "source_revision": revision, "source_sha256": sources, "tensors": tensors,
              "environment": str(environment.path), "package": package,
              "parallelism": {"weight_sharding": False, "inference_parallel_sizes": [1, 2],
                              "full_weights_per_rank": True},
              "weight_payload_gib_per_rank": payload_gib,
              "extra_headroom_gib": args.reserve_gib,
              "estimated_required_free_gib_per_rank": payload_gib + args.reserve_gib,
              "estimate_is_measured_peak": False, "native_inference_verified": False,
              "resource_status": "not_checked"}
    if args.gpu:
        values = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
        free = {int(row.split(",")[0]): int(row.split(",")[1]) / 1024 for row in values.splitlines()}
        report["observed_free_gib"] = {str(g): free.get(g) for g in range(4)}
        report["eligible_gpu_ids"] = [g for g in range(4) if free.get(g, 0) >= payload_gib + args.reserve_gib]
        report["resource_status"] = "eligible_for_attempt_not_verified" if report["eligible_gpu_ids"] else "insufficient_free_memory"
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, report)
    print(json.dumps(report, indent=2))
    return 2 if report["resource_status"] == "insufficient_free_memory" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}), file=sys.stderr)
        raise SystemExit(2)
