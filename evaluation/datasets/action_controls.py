"""Protocol-aware zero-motion and temporal-order controls for recorded Ctrl replay."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from evaluation.execution.transport import read_packet, write_packet
from evaluation.manifests import VideoSpec
from evaluation.reporting.support import digest


def ctrl_control_windows(actions, video, *, seed=0):
    """Hold absolute pose, or permute future states while preserving shared anchors."""
    actions = np.asarray(actions)
    if (actions.ndim != 3 or actions.shape[1:] != (5, 7) or len(actions) < 1
            or not np.isfinite(actions).all()
            or not np.array_equal(actions[1:, 0], actions[:-1, -1])):
        raise ValueError('Expected finite overlapping Ctrl [chunks,5,7] conditions')
    spec = VideoSpec.from_dict(video)
    if spec.temporal_layout != 'inclusive' or spec.frames_per_chunk != 5:
        raise ValueError('Ctrl controls require native inclusive five-frame windows')
    count = spec.frame_count(len(actions))
    sequence = np.concatenate([actions[0, :1], actions[:, 1:].reshape(-1, 7)])[:count]
    order = np.random.default_rng(seed).permutation(np.arange(1, count))
    shuffled = np.concatenate([sequence[:1], sequence[order]])
    indices = np.minimum(np.arange(len(actions))[:, None] * 4 + np.arange(5), count - 1)
    return {'hold_initial_pose': np.broadcast_to(sequence[0], actions.shape).copy(),
            'shuffle_future': shuffled[indices].copy()}, order


def prepare_ctrl_controls(root, prepared, output, *, gpu=0, seed=0):
    from evaluation.execution.branches import validate_branches
    from evaluation.planning.registry import project_path
    root = Path(root).resolve()
    prepared, output = [project_path(root, str(p)) for p in (prepared, output)]
    if output.exists() or type(gpu) is not int or gpu < 0:
        raise ValueError('Use a new output directory and a nonnegative physical GPU')
    config = json.loads((prepared / 'config.json').read_text())
    if (config.get('adapter') != 'CtrlWorldAdapter'
            or config.get('reset_operation') != 'reset_official_replay'
            or config['action'].get('semantics') != 'absolute_eef_state'):
        raise ValueError('Use a prepared Ctrl native absolute-state replay case')
    packet = read_packet(prepared / 'input.npz')
    controls, order = ctrl_control_windows(packet['actions'], config['video'], seed=seed)
    if not np.array_equal(packet['reset_args']['first_action'], packet['actions'][0, 0]):
        raise ValueError('Recorded reset anchor differs from first condition')
    output.mkdir(parents=True, exist_ok=False)
    config['gpus'], config['seed'] = [gpu], seed
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    shutil.copyfile(prepared / 'input.npz', output / 'recorded.npz')
    shutil.copyfile(prepared / 'reference.npz', output / 'reference.npz')
    def asset(path):
        return {'path': str(path.relative_to(root)), 'sha256': digest(path)}
    branches = [{'id': 'recorded', 'input': asset(output / 'recorded.npz'),
                 'reference': asset(output / 'reference.npz')}]
    for name, actions in controls.items():
        changed = copy.deepcopy(packet)
        changed['actions'] = actions
        changed['source']['intervention'] = name
        changed['source']['intervention_truth_available'] = False
        path = output / f'{name}.npz'
        write_packet(path, changed)
        branches.append({'id': name, 'input': asset(path), 'reference': None})
    manifest = {'schema_version': 1, 'config': asset(config_path), 'seeds': [seed],
                'families': [{'id': 'ctrl_episode_' + str(packet['source']['episode_index']),
                              'initial_condition': {'kind': 'observation_only'}, 'branches': branches}]}
    path = output / 'branches.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    validate_branches(root, path)
    (output / 'preparation.json').write_text(json.dumps({
        'source_input_sha256': digest(prepared / 'input.npz'), 'seed': seed,
        'shuffle_future_frame_indices': order.tolist(), 'raw_zero_coordinates_used': False,
        'scope': 'Recorded measured absolute EEF/gripper states are native model conditions, not raw controller commands. Hold means zero motion. Shuffling is a synthetic temporal intervention, potentially nonphysical. Only recorded branch has executed future truth.'}, indent=2) + '\n')
    return str(path)
