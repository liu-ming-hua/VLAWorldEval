"""Real-weight OpenVLA/WorldGym native-vs-adapter traces; no task scoring."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.reporting.support import digest  # noqa: E402
from evaluation.execution.transport import write_packet  # noqa: E402
from evaluation.validation.parity import compare_traces  # noqa: E402
from vlaworldeval.worldgym_runtime import REVISION, verify_runtime, runtime_directory  # noqa: E402


def capture(profile, route, output, seed):
    import numpy as np
    import torch
    from PIL import Image
    from evaluation.datasets.native_reference import indexed_video
    policy = json.loads((ROOT / 'configs/native_parity_v1.json').read_text())[profile]
    for relative, expected in policy['native_source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Pinned native implementation changed')
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_per_process_memory_fraction(min(1., 20*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    data = ROOT / 'datasets/worldgym/official/bridge/test'
    images = indexed_video(data / '000000002.mp4', [0, 5, 10] if profile == 'openvla_bridge' else [0])
    instruction = 'put the eggplant in the pot'
    inputs = {'images': images, 'instruction': instruction}
    if profile == 'openvla_bridge':
        from transformers import AutoModelForVision2Seq, AutoProcessor
        path = ROOT / 'checkpoints/vla/openvla-7b'
        if route == 'adapter':
            from adapters.vla.openvla import OpenVLAAdapter
            adapter = OpenVLAAdapter(str(path))
            model = adapter.model
        else:
            model = AutoModelForVision2Seq.from_pretrained(path, torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True, trust_remote_code=True).cuda().eval()
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        native_actions, model_inputs, converted = [], [], []
        original = model.predict_action
        def observed_predict(**kwargs):
            model_inputs.append({k: {'value': v.detach().float().cpu().numpy(), 'torch_dtype': str(v.dtype)} if isinstance(v, torch.Tensor) else v
                                 for k, v in kwargs.items()})
            result = original(**kwargs)
            native_actions.append(np.asarray(result, dtype=np.float32))
            return result
        model.predict_action = observed_predict
        if route == 'adapter':
            for frame in images:
                converted.append(adapter.predict(frame, instruction).detach().cpu().numpy()[0])
        else:
            sys.path.insert(0, str(ROOT / 'third_party/world-model-eval/src'))
            from world_model_eval import run_openvla as native
            # The upstream loop is executed unchanged; only video dynamics and cloud
            # task scoring are replaced by observation hooks for this VLA-only test.
            class CaptureActions:
                def reset(self, frame):
                    self.frame = frame
                def generate_chunk(self, action):
                    converted.append(action.detach().float().cpu().numpy().copy())
                    yield 0, self.frame[None, None]
            trials = []
            for index, frame in enumerate(images):
                path = output / f'input_{index}.png'
                Image.fromarray(frame).save(path)
                trials.append({'trial_png': str(path), 'instruction': instruction, 'task_key': str(index), 'task_display': str(index)})
            native.predict = lambda *args, **kwargs: 0.0  # ignored placeholder, never reported as task success
            native.evaluate_openvla(CaptureActions(), model, processor, trials, rollout_length=1)
        inputs['model_inputs'] = model_inputs
        outputs = {'native_actions': np.stack(native_actions).astype(np.float32),
                   'wm_actions': np.stack(converted).astype(np.float32)}
    else:
        verify_runtime(ROOT)
        os.environ['WORLDGYM_VAE_PATH'] = str(ROOT / 'checkpoints/world_models/worldgym/vae/sd3_vae')
        checkpoint = ROOT / 'checkpoints/world_models/worldgym/checkpoint/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt'
        with np.load(data / '000000002.npz', allow_pickle=False) as file:
            actions = file['arr_0'][1:4].astype(np.float32)[:, None, :]
        inputs['actions'] = actions
        if route == 'adapter':
            from adapters.world_models.worldgym import WorldGymAdapter
            model = WorldGymAdapter(checkpoint, num_inference_steps=10, default_cfg=3., use_pixel_rope=False)
            model.reset(images[0])
            frames = np.concatenate([model.rollout(chunk) for chunk in actions])
        else:
            sys.path.insert(0, str(runtime_directory(ROOT)))
            from worldgym_runtime.world_model import WorldModel
            model = WorldModel(str(checkpoint), default_cfg=3., use_pixel_rope=False)
            model.reset(torch.from_numpy(images[0]).cuda().float()/255)
            frames = []
            for action in actions:
                generated = {int(i): np.clip(x[0, -1].detach().cpu().numpy()*255, 0, 255).astype(np.uint8)
                             for i, x in model.generate_chunk(torch.as_tensor(action, device='cuda'))}
                frames.extend(generated[i] for i in sorted(generated))
            frames = np.stack(frames)
        outputs = {'frames': frames}
    trace = {'schema_version': 1, 'profile': profile, 'route': route, 'seed': seed,
             'protocol': policy['protocol'], 'native_revision': REVISION,
             'asset_lock_sha256': digest(ROOT / 'configs/first_rollout_assets_v1.json'),
             'inputs': inputs, 'outputs': outputs,
             'provenance': {'native_source_sha256': policy['native_source_sha256'],
                 'adapter_sha256': digest(ROOT / policy['adapter_source']),
                 'runtime': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'diffusers']},
                 'elapsed_seconds': time.monotonic()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'scope': policy['scope']}}
    write_packet(output / 'trace.npz', trace)


def run(args):
    from scripts.acceptance.first_rollout import verify_assets, clean_environment
    from evaluation.execution.admission import resource_admission
    output = args.output.absolute()
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError('Output must stay inside project')
    if args.seed < 0 or args.seed >= 2**32 or args.gpu < 0:
        raise ValueError('Invalid seed/GPU')
    output.mkdir(parents=True, exist_ok=False)
    verify_assets(ROOT, json.loads((ROOT / 'configs/first_rollout_assets_v1.json').read_text()))
    resource_admission({'world_model_gpus': [args.gpu], 'max_process_memory_gib_per_gpu': 20}, False)
    policy = json.loads((ROOT / 'configs/native_parity_v1.json').read_text())[args.profile]
    (output / 'policy.json').write_text(json.dumps(policy, indent=2))
    env = {**clean_environment(ROOT), 'CUDA_VISIBLE_DEVICES': str(args.gpu)}
    for route in ['native', 'adapter']:
        with (output / f'{route}.log').open('w') as log:
            subprocess.run([str(args.python.absolute()), str(Path(__file__).resolve()), '--capture', route,
                            '--profile', args.profile, '--output', str(output / route), '--seed', str(args.seed)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    return compare_traces(output / 'native/trace.npz', output / 'adapter/trace.npz', output / 'policy.json', output / 'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['openvla_bridge', 'worldgym_bridge'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--capture', choices=['native', 'adapter'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture:
        capture(args.profile, args.capture, args.output, args.seed)
    else:
        if args.python is None:
            parser.error('--python must name an explicit model runtime')
        result = run(args)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['status'] == 'passed' else 1)
