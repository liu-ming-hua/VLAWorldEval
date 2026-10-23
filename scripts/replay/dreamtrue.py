#!/usr/bin/env python3
"""DreamTrue official input preparation and one-call shared persistent-worker smoke.

This is native-only: shared native_replay v1 cannot represent geometry mappings.
It deliberately has no private rollout loop and produces no video on blocked runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.world_models.dreamtrue import CHECKPOINT_SHA256, PROTOCOL, SOURCE_REVISION, VIEW_NAMES  # noqa: E402
from evaluation.execution.client import ModelWorker  # noqa: E402
from evaluation.execution.reliability import WorkerOptions  # noqa: E402
from evaluation.execution.transport import read_packet, write_packet  # noqa: E402
from vlaworldeval.assets import file_record  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results/integration/open_world_models_20261018/dreamtrue"


def gpu_status(gpu):
    row = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    name, driver, total, free = [v.strip() for v in row.split(",")]
    # Hard lower bound for 14B BF16 DiT alone; VACE/T5/VAE/activations add more.
    minimum_mib = 14_000_000_000 * 2 / 2**20
    return {"gpu": gpu, "name": name, "driver": driver, "total_mib": int(total),
            "free_mib": int(free), "dit_only_lower_bound_mib": minimum_mib,
            "dit_alone_fits": int(free) >= minimum_mib}


def camera_probe(gpu, memory_gib):
    """Actual trained native Plücker encoder kernel, not full-model inference."""
    from evaluation.execution.admission import resource_admission
    admission = resource_admission({"world_model_gpus": [gpu], "max_process_memory_gib_per_gpu": memory_gib}, False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from safetensors import safe_open
    from adapters.world_models.dreamtrue import wrap_camera_encoder
    sys.path.insert(0, str(ROOT / "third_party/dreamtrue/wmvideo"))
    from diffsynth.models.wan_video_vace import VaceWanModel
    from diffsynth.core.vram import AutoWrappedModule, enable_vram_management
    checkpoint = ROOT / "checkpoints/world_models/dreamtrue/wmvideo/checkpoint-wmvideo.safetensors"
    record = file_record(checkpoint)
    if record["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Camera probe requires verified official dual-LoRA")
    torch.cuda.set_per_process_memory_fraction(memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    # Invoke just the native encoder factory. No DiT/VACE blocks are constructed
    # with random weights and no full video-model result is claimed.
    owner = VaceWanModel.__new__(VaceWanModel)
    torch.nn.Module.__init__(owner)
    owner.vace_in_dim = 96
    owner.mask_cam_embedding = None
    owner.mask_cam_padding = (0, 0, 0, 0, 3, 0)
    owner.enable_plucker()
    prefix = "pipe.vace.mask_cam_embedding."
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        state = {key[len(prefix):]: weights.get_tensor(key) for key in weights.keys() if key.startswith(prefix)}
    owner.mask_cam_embedding.load_state_dict(state, strict=True)
    owner.mask_cam_embedding.to(device="cuda", dtype=torch.bfloat16).eval()
    mask = torch.ones((1, 1, 5, 16, 16), dtype=torch.bfloat16, device="cuda")
    plucker = torch.zeros((1, 6, 5, 16, 16), dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        reference = owner.encode_mask_cam(mask, plucker)
        owner.mask_cam_embedding = wrap_camera_encoder(owner.mask_cam_embedding, device="cuda", torch=torch,
            enable_vram_management=enable_vram_management, wrapped_module=AutoWrappedModule)
        for module in owner.mask_cam_embedding.modules():
            if isinstance(module, AutoWrappedModule):
                module.offload()
                module.onload()
        actual = owner.encode_mask_cam(mask, plucker)
        torch.cuda.synchronize()
    error = float((reference - actual).abs().max().cpu())
    if error != 0 or actual.dtype != torch.bfloat16 or actual.device.type != "cuda":
        raise RuntimeError("Native camera wrapper changes dtype/device/output")
    return {"resource_admission": admission, "trained_checkpoint": record, "strict_keys_loaded": sorted(state),
        "wrapped_layers": {name: type(module).__name__ for name, module in owner.mask_cam_embedding.named_children()},
        "parameter_devices": sorted({str(p.device) for p in owner.mask_cam_embedding.parameters()}),
        "parameter_dtypes": sorted({str(p.dtype) for p in owner.mask_cam_embedding.parameters()}),
        "output_shape": list(actual.shape), "output_device": str(actual.device), "output_dtype": str(actual.dtype),
        "max_abs_error_vs_native_unwrapped": error, "full_video_model_loaded": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["check", "prepare", "smoke", "camera-probe"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--source", choices=["droid", "agibot", "robomind"], default="droid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--cpu-offload", action="store_true", help="Use official layer offload wrappers, retaining BF16")
    parser.add_argument("--vram-limit", type=float, help="Official VRAM management GiB threshold; does not change precision")
    parser.add_argument("--memory-gib", type=float, default=16,
                        help="Explicit worker allocator budget, plus shared admission's 3.5 GiB CUDA margin")
    args = parser.parse_args(argv)
    if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error("--memory-gib must be finite and positive")
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error("--output must be a new project-local result directory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "protocol": PROTOCOL, "operation": args.operation,
              "source_revision": SOURCE_REVISION, "model_inference_performed": False,
              "task_success_evaluated": False, "supports_native_replay_v1": False,
              "worker_backend": "evaluation.execution.client.ModelWorker", "blockers": [], "videos": []}
    report["worker_memory_gib"] = args.memory_gib
    report["cpu_thread_environment"] = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    report["code_sha256"] = {name: file_record(ROOT / name)["sha256"] for name in (
        "adapters/world_models/dreamtrue.py", "evaluation/datasets/dreamtrue.py", "scripts/replay/dreamtrue.py",
        "evaluation/execution/client.py", "evaluation/execution/worker.py", "evaluation/execution/admission.py")}
    worker = None
    try:
        report["gpu"] = None
        if args.operation == "check":
            try:
                report["gpu"] = gpu_status(args.gpu)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                report["gpu_diagnostic"] = {"available": False, "error_type": type(exc).__name__, "error": str(exc)}
        elif args.operation != "prepare":
            report["gpu"] = gpu_status(args.gpu)
        report["cpu_offload"] = args.cpu_offload
        report["vram_limit"] = args.vram_limit
        if args.operation == "smoke" and not args.cpu_offload and not report["gpu"]["dit_alone_fits"]:
            report["blockers"].append("GPU free memory is smaller than the official 14B BF16 DiT alone")
        if args.operation == "camera-probe":
            report["worker_backend"] = "native_camera_kernel_only"
            report["camera_kernel_probe"] = camera_probe(args.gpu, args.memory_gib)
            report["status"] = "camera_probe_passed"
        elif args.operation == "prepare":
            from evaluation.datasets.dreamtrue import prepare_official
            packet = prepare_official(source=args.source, index=args.index, raw_root=args.raw_root)
            write_packet(output / "native_input.npz", packet)
            report.update(status="prepared", packet=str(output / "native_input.npz"),
                          packet_digest=file_record(output / "native_input.npz"),
                          source=packet["condition"]["source"], n_real_frames=packet["condition"]["n_real_frames"])
        elif args.operation == "check":
            source = ROOT / "third_party/dreamtrue/wmvideo"
            env = os.environ.copy()
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            env.update(PYTHONPATH=f"{ROOT}:{source}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                V2V_INF_DRY_RUN="1", NUM_PROCESSES="1", CUDA_VISIBLE_DEVICES=str(args.gpu),
                DREAMTRUE_RELEASE=str(ROOT / "datasets/dreamtrue/release"),
                MODEL_PATH=str(ROOT / "checkpoints/world_models/dreamtrue"),
                DUAL_LORA_CHECKPOINT=str(ROOT / "checkpoints/world_models/dreamtrue/wmvideo/checkpoint-wmvideo.safetensors"))
            if args.vram_limit is not None:
                env["VRAM_LIMIT"] = str(args.vram_limit)
            dry = subprocess.run(["bash", str(source / "Inference/V2V_VACE/inference_threeviews_mixed_20260623.sh")],
                                 env=env, capture_output=True, text=True, timeout=60)
            (output / "official_dry_run.log").write_text(dry.stdout + dry.stderr)
            report["official_dry_run_returncode"] = dry.returncode
            module_import = subprocess.run([sys.executable, "-c",
                "from Script.Demo.inference_demo_v2v_vace import WanV2VVaceInferenceModule; print('official import OK')"],
                env=env, capture_output=True, text=True, timeout=60)
            (output / "official_import.log").write_text(module_import.stdout + module_import.stderr)
            report["official_import_returncode"] = module_import.returncode
            runtime_probe = subprocess.run([sys.executable, "-c",
                "import json,torch; available=torch.cuda.is_available(); "
                "print(json.dumps({'torch':torch.__version__,'cuda_build':torch.version.cuda,"
                "'cuda_available':available,'bf16_supported':torch.cuda.is_bf16_supported() if available else None,"
                "'device':torch.cuda.get_device_name(0) if available else None,"
                "'memory_free_total':torch.cuda.mem_get_info(0) if available else None}))"],
                env=env, capture_output=True, text=True, timeout=60)
            (output / "cuda_runtime_probe.log").write_text(runtime_probe.stdout + runtime_probe.stderr)
            report["cuda_runtime_probe_returncode"] = runtime_probe.returncode
            uv = os.environ.get("UV_BIN") or __import__("shutil").which("uv")
            if uv:
                checked = subprocess.run([uv, "pip", "check", "--python", sys.executable],
                    env=env, capture_output=True, text=True, timeout=60)
                (output / "uv_pip_check.log").write_text(checked.stdout + checked.stderr)
                report["uv_pip_check_returncode"] = checked.returncode
            checkpoint = ROOT / "checkpoints/world_models/dreamtrue/wmvideo/checkpoint-wmvideo.safetensors"
            report["checkpoint"] = file_record(checkpoint) if checkpoint.is_file() else None
            if not report["checkpoint"] or report["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
                report["blockers"].append("Official dual-LoRA checkpoint missing or not verified")
            base = ROOT / "checkpoints/world_models/dreamtrue/Wan-AI/Wan2.1-VACE-14B"
            report["base_available"] = (base / ".vlaworldeval-download.json").is_file()
            if not report["base_available"]:
                report["blockers"].append("Exact Wan2.1-VACE-14B base snapshot is not installed")
            worker = ModelWorker(Path(sys.executable), output / "worker", module="adapters.world_models.dreamtrue",
                adapter="DreamTrueAdapter", kwargs={}, gpus=[], memory_gib=1,
                options=WorkerOptions(startup_timeout_seconds=60, request_timeout_seconds=60))
            report["worker_ready"] = worker.ready
            worker.call("seed", seed=0)
            report["worker_seed_call_completed"] = True
            report["status"] = "checks_passed_with_blockers" if report["blockers"] else "checks_passed"
            if dry.returncode or module_import.returncode or runtime_probe.returncode or report.get("uv_pip_check_returncode", 0):
                report["status"] = "failed"
        else:
            if not args.packet:
                raise ValueError("smoke requires --packet from the prepare operation")
            packet = read_packet(args.packet)
            if (not isinstance(packet, dict) or set(packet) != {"schema_version", "protocol", "reset", "condition"}
                    or packet["schema_version"] != 1 or packet["protocol"] != PROTOCOL):
                raise ValueError("Invalid DreamTrue native packet")
            report["source"] = packet["condition"]["source"]
            report["input_packet"] = {"path": str(args.packet), **file_record(args.packet)}
            base_marker = ROOT / "checkpoints/world_models/dreamtrue/Wan-AI/Wan2.1-VACE-14B/.vlaworldeval-download.json"
            if base_marker.is_file():
                base_record = json.loads(base_marker.read_text())
                report["base_snapshot"] = {"repository": base_record["repository"], "revision": base_record["revision"],
                    "selected_files": len(base_record["files"]),
                    "selected_snapshot_bytes": sum(item["size"] for item in base_record["files"].values()),
                    "diffusion_shard_bytes": sum(item["size"] for name, item in base_record["files"].items()
                        if name.startswith("diffusion_pytorch_model-") and name.endswith(".safetensors")),
                    "manifest_sha256": file_record(base_marker)["sha256"]}
            if report["blockers"]:
                report["status"] = "resource_blocked"
            else:
                import mediapy
                from evaluation.execution.admission import resource_admission
                report["resource_admission"] = resource_admission({"world_model_gpus": [args.gpu],
                    "max_process_memory_gib_per_gpu": args.memory_gib}, False)
                worker = ModelWorker(Path(sys.executable), output / "worker", module="adapters.world_models.dreamtrue",
                    adapter="DreamTrueAdapter", kwargs={"cpu_offload": args.cpu_offload, "vram_limit": args.vram_limit,
                        "audit_path": str(output / "checkpoint_audit.json")}, gpus=[args.gpu],
                    memory_gib=args.memory_gib,
                    options=WorkerOptions(startup_timeout_seconds=120, request_timeout_seconds=3600,
                                          shutdown_grace_seconds=120))
                report["worker_ready"] = worker.ready
                report["native_options"] = {"height": 240, "width": 320, "num_frames": 101,
                    "num_inference_steps": 50, "seed": 0, "cfg_scale": 1.0, "vace_scale": 1.0,
                    "use_plucker": True, "tiled": False, "dtype": "torch.bfloat16"}
                worker.call("seed", seed=0)
                worker.call("reset", value=packet["reset"])
                started = time.perf_counter()
                result = worker.call("step", action=packet["condition"], output_views=list(VIEW_NAMES), state_schema=None)
                report["model_step_seconds"] = time.perf_counter() - started
                report["worker_memory"] = worker.memory
                report["output_shapes"] = {name: list(result["views"][name].shape) for name in VIEW_NAMES}
                for name in VIEW_NAMES:
                    video_path = output / f"{name}.mp4"
                    mediapy.write_video(video_path, result["views"][name], fps=5)
                    report["videos"].append({"path": str(video_path), **file_record(video_path)})
                report.update(status="completed", model_inference_performed=True, metadata=result["metadata"])
    except Exception as exc:
        report.update(status="blocked", error_type=type(exc).__name__, error=str(exc))
        report["blockers"].append(str(exc))
    finally:
        if worker is not None:
            report["worker_cleanup"] = worker.close()
            from evaluation.execution.reliability import clean_worker_exit
            if report.get("status") == "completed" and not clean_worker_exit(report["worker_cleanup"]):
                report["status"] = "inference_completed_cleanup_failed"
                report["blockers"].append("Generated videos preserved, but worker did not exit cleanly")
        if (output / "checkpoint_audit.json").is_file():
            report["checkpoint_audit_file"] = {"path": str(output / "checkpoint_audit.json"),
                                               **file_record(output / "checkpoint_audit.json")}
        (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] in {"prepared", "completed", "checks_passed", "checks_passed_with_blockers", "camera_probe_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
