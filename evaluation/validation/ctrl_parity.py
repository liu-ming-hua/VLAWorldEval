"""Real Ctrl replay-method parity, with shared weights and controlled VAE/diffusion RNG."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.execution.transport import write_packet
from evaluation.reporting.support import digest
from evaluation.validation.parity import compare_traces


def capture_ctrl(root, output, *, episode=199, chunks=2, seed=0):
    import einops
    import torch
    from adapters.world_models.ctrl_world import CtrlWorldAdapter
    from evaluation.datasets.ctrl_world import REVISION, load_released_episode, verified_asset
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    repo = root / 'third_party/ctrl-world'
    source = repo / 'scripts/rollout_replay_traj.py'
    verified_asset(repo, str(source.relative_to(repo)))
    states, views, instruction, records = load_released_episode(repo, episode)
    if chunks < 1 or 4 * chunks + 1 > len(states):
        raise ValueError('Parity requires complete inclusive chunks')
    torch.manual_seed(seed)
    kwargs = dict(svd_model_path=root/'checkpoints/world_models/ctrl_world/svd',
                  clip_model_path=root/'checkpoints/world_models/ctrl_world/clip',
                  checkpoint_path=root/'checkpoints/world_models/ctrl_world/ctrl_world/checkpoint-10000.pt',
                  repo_path=repo, prompt=instruction, num_inference_steps=50)
    wm = CtrlWorldAdapter(**kwargs)
    # Compile the pinned class itself, not a reimplementation. Omit module-level
    # teleoperation/logging imports unrelated to replay. The model stays resident.
    from decord import VideoReader, cpu
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'agent')
    namespace = dict(np=np, torch=torch, json=json, VideoReader=VideoReader, cpu=cpu,
                     einops=einops, CtrlWorldDiffusionPipeline=wm._pipeline_cls)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    native = object.__new__(namespace['agent'])
    native.model, native.device, native.dtype = wm.model, wm.device, wm.dtype
    stat = json.loads(wm.data_stat_path.read_text())
    native.state_p01 = np.asarray(stat['state_01'])[None]
    native.state_p99 = np.asarray(stat['state_99'])[None]
    native.args = SimpleNamespace(val_dataset_dir=str(repo/'dataset_example/droid_subset'),
        skip_step=1, num_history=6, num_frames=5, action_dim=7, width=320, height=192,
        num_inference_steps=50, decode_chunk_size=7, guidance_scale=1., fps=7, motion_bucket_id=127)
    actions = np.stack([states[i*4:i*4+5].astype(np.float32) for i in range(chunks)])
    frames = [views[v][0] for v in ('exterior_1', 'exterior_2', 'wrist')]
    captures = []
    original = wm._pipeline_cls.__call__
    def hook(pipeline, *args, **kw):
        captures.append({k: kw[k].detach().float().cpu().numpy().copy() for k in ('image', 'text', 'history')})
        return original(pipeline, *args, **kw)
    wm._pipeline_cls.__call__ = hook
    lock = {'checkpoint_sha256': digest(kwargs['checkpoint_path']), 'source_sha256': digest(source),
            'data': records, 'statistics_sha256': digest(wm.data_stat_path)}
    lock_path = output/'assets.json'
    lock_path.write_text(json.dumps(lock, indent=2))
    try:
        for route in ('native', 'adapter'):
            captures.clear()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                if route == 'native':
                    # Only the initial RGB is encoded. Future truth is never a model input.
                    _, _, _, latents, _ = native.get_traj_info(episode, steps=1)
                    initial = torch.cat([v[0] for v in latents], dim=1).unsqueeze(0)
                    memory = [initial.clone() for _ in range(24)]
                    memory_actions = [actions[0, :1].copy() for _ in range(24)]
                else:
                    wm.reset_official_replay(frames, actions[0, 0])
                videos = []
                for chunk in actions:
                    if route == 'native':
                        indices = [0, 0, -8, -6, -4, -2]
                        history = torch.cat([memory[i] for i in indices]).unsqueeze(0)
                        cond = np.concatenate([np.concatenate([memory_actions[i] for i in indices]), chunk])
                        _, _, video, predicted = native.forward_wm(cond, [v.expand(5, -1, -1, -1) for v in latents], memory[-1], history, instruction)
                        videos.append(video.transpose(1, 0, 2, 3, 4))
                        memory.append(torch.cat([v[-1] for v in predicted], dim=1).unsqueeze(0))
                        memory_actions.append(chunk[-1:].copy())
                    else:
                        videos.append(wm.rollout(chunk))
                trace = {'schema_version': 1, 'profile': 'ctrl_released', 'protocol': 'ctrl_native_v2',
                    'route': route, 'seed': seed, 'asset_lock_sha256': digest(lock_path), 'native_revision': REVISION,
                    'inputs': {'images': np.stack(frames), 'actions': actions, 'instruction': instruction,
                               'model_conditions': list(captures)},
                    'outputs': {'frames': np.stack(videos)},
                    'provenance': {'source_sha256': digest(source), 'adapter_sha256': digest(root/'adapters/world_models/ctrl_world.py'),
                        'torch': torch.__version__, 'shared_weights': True,
                        'initial_only_encoding': True, 'future_reference_in_inference': False}}
                write_packet(output/f'{route}.npz', trace)
    finally:
        wm._pipeline_cls.__call__ = original
    policy = {'profile': 'ctrl_released', 'protocol': 'ctrl_native_v2', 'atol': 0, 'rtol': 0,
              'scope': 'Pinned native get_traj_info/forward_wm versus adapter, shared loaded weights; initial-only VAE encoding and identical RNG; excludes module CLI and full-trajectory VAE sampling.'}
    (output/'policy.json').write_text(json.dumps(policy, indent=2))
    return compare_traces(output/'native.npz', output/'adapter.npz', output/'policy.json', output/'report.json')
