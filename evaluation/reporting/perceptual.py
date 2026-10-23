"""Offline FP32 CPU/explicit-GPU LPIPS; never a VLA/WM loop.

This supplemental metric explicitly uses decoded generated MP4s. The executor's
primary PSNR/SSIM use raw predictions and must not be relabelled as this domain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_lpips(root, device="cpu"):
    import torch
    import lpips
    from safetensors.torch import load_file
    from vlaworldeval.assets import verify_snapshot, file_record
    directory = root / "checkpoints/evaluators/lpips_vgg"
    verified = verify_snapshot(directory)
    if (verified["repository"] != "timm/vgg16.tv_in1k"
            or verified["revision"] != "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f"):
        raise ValueError("LPIPS requires the pinned verified official ImageNet VGG features")
    torch.set_num_threads(4)
    # pnet_rand prevents TorchVision's implicit direct Internet download. Every
    # feature tensor is then strictly replaced by the verified ImageNet payload.
    model = lpips.LPIPS(net="vgg", pnet_rand=True, pretrained=True, verbose=False).cpu().eval()
    original = load_file(str(directory / "model.safetensors"), device="cpu")
    state = {}
    for key, value in model.net.state_dict().items():
        source = "features." + key.split(".", 1)[1]
        if source not in original or original[source].shape != value.shape:
            raise ValueError("VGG feature shape/key differs from the LPIPS evaluator")
        state[key] = original[source]
    model.net.load_state_dict(state, strict=True)
    if not all(torch.equal(value, state[key]) for key, value in model.net.state_dict().items()):
        raise ValueError("LPIPS feature weights were not fully replaced")
    calibration = Path(lpips.__file__).parent / "weights/v0.1/vgg.pth"
    trained = torch.load(calibration, map_location="cpu", weights_only=True)
    linear = {key: value for key, value in model.named_parameters() if not key.startswith("net.")}
    if set(linear) != set(trained) or not all(torch.equal(value, trained[key]) for key, value in linear.items()):
        raise ValueError("Not every learned LPIPS calibration parameter matches the packaged release")
    model = model.to(device)
    return model, {"backbone_snapshot": verified, "feature_tensors_verified": len(state),
                   "all_feature_tensors_loaded_strictly": True, "random_features_used": False,
                   "calibration_tensors_verified": len(linear),
                   "linear_calibration": file_record(calibration), "device": device, "dtype": "float32"}


def score_replay(summary, reference, output, *, device="cpu", frame_batch_size=1):
    import torch
    from evaluation.execution.transport import read_packet
    from evaluation.execution.reliability import clean_worker_exit
    from evaluation.reporting.task_scores import _digest
    from evaluation.datasets.native_reference import indexed_video
    from vlaworldeval.paths import project_root
    root = project_root()
    if device not in {"cpu", "cuda"} or type(frame_batch_size) is not int or not 1 <= frame_batch_size <= 32:
        raise ValueError("Use cpu/cuda and a frame batch size in1..32")
    if device == "cuda":
        import os
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
            raise ValueError("CUDA metric requires exactly one explicitly assigned visible GPU")
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 8*2**30/total), 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    summary, reference, output = map(lambda p: Path(p).resolve(), (summary, reference, output))
    if output.exists() or not output.is_relative_to(root):
        raise ValueError("Perceptual report must be a new project-local file")
    report = json.loads(summary.read_text())
    ref = read_packet(reference)
    if (report["status"] != "completed" or not clean_worker_exit(report["cleanup"])
            or report["input_sha256"] != ref["input_sha256"]
            or report["reference_evaluation"]["reference_sha256"] != _digest(reference)):
        raise ValueError("Only completed, reference-bound clean replays may receive perceptual scores")
    model, audit = load_lpips(root, device)
    values = {}
    def tensor(value):
        return (torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float()/255).to(device)
    with torch.inference_mode():
        for name, gt in ref["views"].items():
            path = summary.parent / f"{name}.mp4"
            if _digest(path) != report["files"][path.name]:
                raise ValueError("Generated video changed after inference")
            pred = indexed_video(path, range(len(gt)))
            scores, baseline = [], []
            for i in range(1, len(gt), frame_batch_size):
                target = tensor(gt[i:i+frame_batch_size])
                scores.extend(model(tensor(pred[i:i+frame_batch_size]), target, normalize=True).flatten().tolist())
                frozen = tensor(np.repeat(gt[:1], len(target), axis=0))
                baseline.extend(model(frozen, target, normalize=True).flatten().tolist())
            values[name] = {"lpips_vgg": float(np.mean(scores)),
                            "frozen_initial_lpips_vgg": float(np.mean(baseline)),
                            "per_frame": scores, "initial_frame_scored": False}
    result = {"schema_version": 1, "status": "completed", "summary_sha256": _digest(summary),
              "reference_sha256": _digest(reference), "evaluator": audit, "views": values,
              "prediction_domain": "decoded_generated_mp4_vs_raw_prepared_reference",
              "physical_action_tracking_verified": False, "task_success_evaluated": False,
              "cuda_initialized": torch.cuda.is_initialized(), "frame_batch_size": frame_batch_size}
    from evaluation.reporting.checkpoint_metrics import metric_source_fingerprints
    result["implementation_sha256"] = metric_source_fingerprints("lpips")
    output.parent.mkdir(parents=True, exist_ok=True)
    from vlaworldeval.atomic_json import publish_json_exclusive
    publish_json_exclusive(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--frame-batch-size", type=int, default=1)
    args = parser.parse_args()
    from vlaworldeval.paths import project_root
    from vlaworldeval.cache import runtime_cache_environment
    import os
    os.environ.update(runtime_cache_environment(project_root()), HF_HUB_OFFLINE="1")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = score_replay(args.summary, args.reference, args.output,
                          device=args.device, frame_batch_size=args.frame_batch_size)
    print(json.dumps({"status": result["status"], "views": result["views"], "cuda_initialized": result["cuda_initialized"]}))
