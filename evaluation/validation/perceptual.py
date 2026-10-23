"""Qualify FP32 GPU/frame batching against serial CPU LPIPS on real replay RGB."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from evaluation.execution.admission import native_gpu_lease
from evaluation.execution.transport import read_packet
from evaluation.datasets.native_reference import indexed_video
from evaluation.reporting.perceptual import load_lpips
from evaluation.reporting.checkpoint_metrics import metric_source_fingerprints
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive
from vlaworldeval.paths import project_root


def verify(summary, reference, output, *, gpu=0, frames=8, tolerance=5e-5):
    if type(gpu) is not int or gpu not in {0, 1, 2, 3} or type(frames) is not int or not 1 <= frames <= 16:
        raise ValueError("Choose an authorized GPU0..3 and1..16 actual future frames")
    root = project_root()
    summary, reference, output = (Path(p).resolve() for p in (summary, reference, output))
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Qualification needs a fresh project-local artifact")
    report = json.loads(summary.read_text())
    if report['status'] != 'completed' or report['reference_evaluation']['reference_sha256'] != _digest(reference):
        raise ValueError("Qualification requires a completed, reference-bound replay")
    ref = read_packet(reference)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    results = {}
    with native_gpu_lease(root, [gpu]), torch.inference_mode():
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1, 8*2**30/torch.cuda.get_device_properties(0).total_memory), 0)
        model, audit = load_lpips(root)
        for name, gt in ref['views'].items():
            path = summary.parent/f'{name}.mp4'
            if _digest(path) != report['files'][path.name]:
                raise ValueError("Replay video changed")
            count = min(frames, len(gt)-1)
            pred = indexed_video(path, range(1, count+1))
            def tensor(array):
                return torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float()/255
            prediction, target = tensor(pred), tensor(gt[1:count+1])
            frozen = tensor(np.repeat(gt[:1], count, axis=0))
            model = model.cpu()
            serial = torch.cat([model(prediction[i:i+1], target[i:i+1], normalize=True).flatten()
                                for i in range(count)])
            serial_frozen = torch.cat([model(frozen[i:i+1], target[i:i+1], normalize=True).flatten()
                                       for i in range(count)])
            cpu_batch = model(prediction, target, normalize=True).flatten()
            model = model.cuda()
            gpu_batch = model(prediction.cuda(), target.cuda(), normalize=True).flatten().cpu()
            gpu_frozen = model(frozen.cuda(), target.cuda(), normalize=True).flatten().cpu()
            errors = {'cpu_batch_vs_serial_max_abs': float((cpu_batch-serial).abs().max()),
                      'gpu_batch_vs_cpu_serial_max_abs': float((gpu_batch-serial).abs().max()),
                      'gpu_frozen_vs_cpu_serial_max_abs': float((gpu_frozen-serial_frozen).abs().max())}
            results[name] = {'actual_frames': count, 'image_hwc': list(pred.shape[1:]), **errors,
                'passed': all(np.isfinite(v) and v <= tolerance for v in errors.values())}
    result = {'schema_version': 1, 'status': 'passed' if all(r['passed'] for r in results.values()) else 'failed',
        'summary_sha256': _digest(summary), 'reference_sha256': _digest(reference), 'tolerance_absolute': tolerance,
        'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(0), 'gpu': gpu,
        'dtype': 'float32', 'tf32': False, 'views': results, 'evaluator': audit,
        'implementation_sha256': metric_source_fingerprints('lpips'), 'validation_source_sha256': _digest(Path(__file__)),
        'physical_action_tracking_verified': False, 'real_video_resized': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--frames', type=int, default=8)
    args = parser.parse_args()
    result = verify(args.summary, args.reference, args.output, gpu=args.gpu, frames=args.frames)
    print(json.dumps({'status': result['status'], 'views': result['views']}))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
