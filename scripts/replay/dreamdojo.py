#!/usr/bin/env python3
"""Prepare GR1 data, then invoke the shared native replay executor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def audit_environment(output: Path, config: Path, packet: Path):
    """Write actual diagnostics, including failures; never manufacture a video."""
    from vlaworldeval.assets import MANIFEST, verify_snapshot
    from evaluation.execution.native_replay import run_native_replay
    output.mkdir(parents=True, exist_ok=False)
    python = ROOT / "environment/world_models/dreamdojo/bin/python"
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv")
    environment = os.environ.copy()
    from vlaworldeval.cache import runtime_cache_environment
    environment.update(runtime_cache_environment(ROOT))
    for key in list(environment):
        if key.lower().endswith("_proxy"):
            environment.pop(key)
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_ENDPOINT="https://hf-mirror.com",
                       PYTHONPATH=f"{ROOT}:{ROOT / 'third_party/dreamdojo'}")
    libs = python.parent.parent / "lib/python3.10/site-packages/nvidia"
    environment["LD_LIBRARY_PATH"] = ":".join(str(libs / name / "lib") for name in ("cudnn", "cublas", "cuda_runtime"))
    checks = {
        "cpu_tests": [str(ROOT / "environment/core/bin/python"), "-m", "pytest",
                      str(ROOT / "tests/models/test_dreamdojo.py"), "-q"],
        "ruff": [str(ROOT / "environment/core/bin/python"), "-m", "ruff", "check",
                 *[str(ROOT / name) for name in (
                     "adapters/world_models/dreamdojo.py", "scripts/download/dreamdojo.py",
                     "scripts/replay/dreamdojo.py", "evaluation/datasets/dreamdojo.py", "tests/models/test_dreamdojo.py")]],
        "distributions": [str(python), "-c", "import importlib.metadata as m,json; "
                          "print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True,indent=2))"],
        "pip_check": [uv, "pip", "check", "--python", str(python)],
        "runtime_import": [str(python), "-c", "import torch; print(torch.__version__,torch.version.cuda); "
                           "from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference"],
        "encoding_compare": [str(python), str(ROOT / "scripts/replay/dreamdojo.py"), "--compare-encoding",
                             "--packet", str(packet)],
        "gpu": ["nvidia-smi", "--query-gpu=index,driver_version,memory.free,utilization.gpu", "--format=csv"],
    }
    evidence = {}
    for name, command in checks.items():
        result = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=120)
        (output / f"{name}.txt").write_text(result.stdout + result.stderr)
        evidence[name] = {"returncode": result.returncode, "command": command, "log": f"{name}.txt"}
    assets = {}
    for name, directory in {
        "model": ROOT / "checkpoints/world_models/dreamdojo",
        "tokenizer": ROOT / "checkpoints/world_models/dreamdojo/tokenizer",
        "reason": ROOT / "checkpoints/world_models/dreamdojo/reason",
        "sample": ROOT / "datasets/dreamdojo/official",
    }.items():
        if (directory / MANIFEST).is_file():
            assets[name] = verify_snapshot(directory)
        else:
            assets[name] = {"ok": False, "status": "incomplete_or_not_downloaded", "directory": str(directory)}
    try:
        admission = run_native_replay(config, packet, output / "native", admit_only=True)
    except Exception as exc:
        admission = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    report = {"status": "inference_not_performed", "model_inference_performed": False,
              "compatibility_profile": "dreamdojo_2b_gr1_wan_vae_torch271_cu126_v1",
              "exact_paper_reproduction": False,
              "generated_video": None, "checks": evidence, "assets": assets,
              "shared_executor_admission": admission, "packet": str(packet),
              "source_revision": subprocess.check_output(["git", "-C", str(ROOT / "third_party/dreamdojo"),
                                                           "rev-parse", "HEAD"], text=True).strip()}
    report["interrupted_transfer_files_preserved"] = [str(path.relative_to(ROOT)) for path in
        (ROOT / "checkpoints/world_models/dreamdojo").rglob("*preserved") if path.is_file()]
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def record_compatibility_evidence(output: Path, report):
    """Keep scientific caveats next to the unmodified shared-executor summary."""
    from adapters.world_models.dreamdojo import COMPATIBILITY_PROFILE
    if report.get("status") != "completed" or report.get("world_model_calls") != 1:
        raise RuntimeError("DreamDojo native replay did not complete exactly one validated chunk")
    steps = []
    for line in (output / "worker/console.log").read_text().splitlines():
        if line.startswith("DREAMDOJO_STEP_EVIDENCE="):
            steps.append({"metadata": json.loads(line.split("=", 1)[1]),
                          "evidence_source": "worker/console.log"})
    if len(steps) != 1 or steps[0]["metadata"].get("compatibility_profile") != COMPATIBILITY_PROFILE:
        raise RuntimeError("Missing actual worker compatibility/tensor-coverage evidence")
    evidence = {"compatibility_profile": COMPATIBILITY_PROFILE, "exact_paper_reproduction": False,
                "asset_fidelity_status": steps[0]["metadata"]["asset_fidelity_status"],
                "model_inference_performed": True, "shared_summary": "summary.json",
                "state_feedback_vla": False, "max_chunks_per_reset": 1, "steps": steps}
    (output / "compatibility_profile.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--admit-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Record environment/assets/admission without inference")
    parser.add_argument("--compare-encoding", action="store_true", help="CPU comparison with pinned official dataset transform")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/dreamdojo/official/In-lab_Eval/gr1_unified.color_puzzles_robot")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/replay/dreamdojo_native_v1.json")
    parser.add_argument("--packet", type=Path, default=ROOT / "datasets/dreamdojo/prepared/native_v2.packet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/integration/open_world_models_20261018/dreamdojo/native")
    args = parser.parse_args(argv)
    if args.chunks != 1:
        parser.error("multi-chunk replay is unsupported; exactly one chunk is validated")
    if args.compare_encoding:
        from evaluation.datasets.dreamdojo import compare_official_encoding
        print(json.dumps(compare_official_encoding(args.dataset, args.packet), indent=2))
        return
    if args.audit_only:
        print(json.dumps(audit_environment(args.output, args.config, args.packet), indent=2))
        return
    if args.prepare_only or not args.packet.exists():
        from evaluation.datasets.dreamdojo import prepare_packet
        print(json.dumps(prepare_packet(args.dataset, args.config, args.packet,
                                        chunks=args.chunks, start=args.start), indent=2))
    if not args.prepare_only:
        from evaluation.datasets.dreamdojo import validate_prepared_packet
        from evaluation.execution.transport import read_packet
        validate_prepared_packet(read_packet(args.packet))
        from evaluation.execution.native_replay import run_native_replay
        report = run_native_replay(args.config, args.packet, args.output, admit_only=args.admit_only)
        if not args.admit_only:
            record_compatibility_evidence(args.output, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
