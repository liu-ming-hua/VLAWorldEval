"""Native action-mode client versus adapter, with independently seeded servers."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile

import numpy as np

from evaluation.reporting.support import digest


def array_record(value):
    value = np.ascontiguousarray(value)
    if not value.size or not np.isfinite(value).all():
        raise ValueError('Expected a finite nonempty array')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def audit_future_conditions(episode, actions):
    """Change future recorded states/calibration, keeping frame zero and actions fixed."""
    from gesim.conditioning.policy_band import load_bundle_cache, render_policy_traj_from_bundle
    cache = load_bundle_cache(episode)
    changed = {key: value.copy() for key, value in cache.items()}
    for key in ('state_joints', 'state_eef'):
        changed[key][1:] += 123.0
    changed['c2w_episode'][:, 1:] += 123.0
    left = render_policy_traj_from_bundle(actions, episode, bundle_cache=cache)
    right = render_policy_traj_from_bundle(actions, episode, bundle_cache=changed)
    names = ('trajectory_band', 'camera_to_world', 'fk_eef')
    checks = {name: array_record(np.asarray(a)) == array_record(np.asarray(b))
              for name, a, b in zip(names, left, right)}
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'scope': 'Future recorded state/EEF/camera rows only; initial calibration and commanded actions retained. Not a physical counterfactual.'}


def capture_ge_sim(root, output, episode, *, chunks=2, tail_actions=6, gpu=0, memory_gib=28, seed=0):
    """Run in the GE-Sim environment; temporary raw arrays are deleted on exit."""
    from adapters.environment import get_environment
    from adapters.world_models.ge_sim_v2 import GESimV2Adapter
    from evaluation.datasets.ge_sim import prepare_official
    from evaluation.execution.admission import native_gpu_lease, resource_admission
    from evaluation.execution.local_server import local_server
    from evaluation.execution.reliability import run_signal_handlers
    from gesim import WorldModelEnv
    import yaml

    if (type(chunks) is not int or chunks < 1 or type(tail_actions) is not int
            or not 0 <= tail_actions < 25 or gpu < 0
            or not np.isfinite(memory_gib) or memory_gib <= 0):
        raise ValueError('Require positive chunks/memory, nonnegative GPU and tail_actions in 0..24')
    root, output, episode = Path(root).resolve(), Path(output).resolve(), Path(episode).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'conditioning': 'action', 'gpu': gpu, 'seed': seed,
              'physical_reliability_verified': False, 'reference_future_rgb_available': False,
              'scope': 'Official WorldModelEnv action mode vs adapter. Not parity with episode-mode examples/replay.py; JPEG HTTP transport included.',
              'chunks': [], 'servers': []}
    path = output / 'report.json'
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        config_path = root / 'configs/world_models/ge_sim_v2.yaml'
        server_config = yaml.safe_load(config_path.read_text())
        checkpoint = root / server_config['checkpoint']
        repo = root / 'third_party/ge-sim-v2'
        patch = root / 'scripts/setup/patches/ge_sim_v2_finite_tail.patch'
        subprocess.run(['git', '-C', str(repo), 'apply', '--reverse', '--check', str(patch)], check=True)
        report['assets'] = {'checkpoint_sha256': digest(checkpoint / 'model.safetensors'),
                            'server_config_sha256': digest(config_path), 'patch_sha256': digest(patch),
                            'upstream_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                            'adapter_sha256': digest(root / 'adapters/world_models/ge_sim_v2.py')}
        actions = np.load(episode / 'actions_0.npy', allow_pickle=False).astype(np.float32)
        count = chunks * 25 + tail_actions
        if count > len(actions):
            raise ValueError('Insufficient recorded actions; padding is not allowed')
        selected = [actions[i:min(i + 25, count)] for i in range(0, count, 25)]
        environment = dict(os.environ)
        for key in list(environment):
            if key.lower().endswith('_proxy'):
                environment.pop(key)
                os.environ.pop(key, None)
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', HF_ENDPOINT='https://hf-mirror.com')
        python = get_environment('world_models.ge_sim_v2').python
        bootstrap = ('import random,numpy as np,torch,runpy; '
                     f'random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed}); '
                     'torch.cuda.set_device(0); '
                     f'torch.cuda.set_per_process_memory_fraction(min(1,{memory_gib}*2**30/torch.cuda.get_device_properties(0).total_memory),0); '
                     'runpy.run_module("gesim.server",run_name="__main__")')
        with tempfile.TemporaryDirectory(prefix='ge-parity-', dir=output) as scratch:
            scratch = Path(scratch)
            prepared = prepare_official(root / 'configs/replay/ge_sim_v2_action_full_v1.json', episode,
                                        scratch / 'input.npz', chunks='all')
            report['data'] = prepared['source']
            report['future_condition_audit'] = audit_future_conditions(episode, selected[0])
            if report['future_condition_audit']['status'] != 'passed':
                raise ValueError('Future recorded state changes affect action-mode conditioning')
            with run_signal_handlers(), native_gpu_lease(root, [gpu]):
                for route in ('native', 'adapter'):
                    resource_admission({'world_model_gpus': [gpu], 'max_process_memory_gib_per_gpu': memory_gib}, False)
                    with socket.socket() as sock:
                        sock.bind(('127.0.0.1', 0))
                        port = sock.getsockname()[1]
                    url = f'http://127.0.0.1:{port}'
                    command = [str(python), '-c', bootstrap, '--model', 'gesim_v2', '--config', str(config_path),
                               '--host', '127.0.0.1', '--port', str(port)]
                    with local_server(command, cwd=root, environment=environment,
                                      log_path=output / f'{route}-server.log', health_url=url + '/healthz') as server:
                        report['servers'].append(server)
                        client = (WorldModelEnv(url, chunk_size=25, compress_actions=False, keep_frames=False, timeout=600)
                                  if route == 'native' else GESimV2Adapter(server_url=url, compress_actions=False, timeout=600))
                        try:
                            if route == 'native':
                                client.reset(episode, conditioning='action')
                            else:
                                client.reset(episode)
                            for index, action in enumerate(selected):
                                if route == 'native':
                                    _, _, state, info = client.step(action)
                                    # Independent expression of the documented RGB contract.
                                    video = np.rint(np.clip(info.frames[:, :, :3], 0, 1) * 255).astype(np.uint8).transpose(0, 2, 3, 4, 1)
                                    np.save(scratch / f'video-{index}.npy', video)
                                    np.save(scratch / f'state-{index}.npy', state)
                                    report['chunks'].append({'actions': array_record(action), 'frames': len(video),
                                                             'native_video': array_record(video), 'native_state': array_record(state)})
                                else:
                                    video = client.rollout(action)
                                    row = report['chunks'][index]
                                    row['adapter_video'] = array_record(video)
                                    row['adapter_state'] = array_record(client.last_state)
                                    row['exact_video_match'] = row['native_video'] == row['adapter_video']
                                    row['exact_state_match'] = row['native_state'] == row['adapter_state']
                                    expected = np.load(scratch / f'video-{index}.npy', mmap_mode='r')
                                    row['rgb_max_abs'] = int(np.abs(expected.astype(np.int16) - video.astype(np.int16)).max())
                                    row['frame_count_matches_actions'] = len(video) == len(action)
                                save()
                        finally:
                            client.close()
            report['status'] = ('passed' if all(x['exact_video_match'] and x['exact_state_match'] and x['frame_count_matches_actions']
                                               for x in report['chunks']) else 'failed')
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    return report
