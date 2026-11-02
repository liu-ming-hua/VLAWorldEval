"""Prepare GE-Sim's calibrated PNG episode bundle for action-driven replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.execution.transport import write_packet


VIEWS = {'head': 'cur_head', 'left_wrist': 'cur_left', 'right_wrist': 'cur_right'}


def prepare_official(config_path, dataset, output, *, chunks=2):
    config = json.loads(Path(config_path).read_text())
    if (config.get('adapter') != 'GESimV2Adapter'
            or config.get('kwargs', {}).get('conditioning') != 'action'
            or config['kwargs'].get('compress_actions') is not False
            or config['kwargs'].get('chunk_size') != 25
            or config.get('action', {}).get('dimension') != 16
            or config['action'].get('horizon') != 25
            or config.get('output_views') != list(VIEWS)):
        raise ValueError('Requires uncompressed GE-Sim action-conditioned 25x16 protocol')
    if chunks != 'all' and (type(chunks) is not int or chunks < 1):
        raise ValueError('chunks must be a positive integer or all')
    dataset, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    required = ['actions_0.npy', 'task.txt', 'intrinsic.npy', 'extrinsic_alignstate_0.npy',
                'state_joints_0.npy', 'state_eef_poses_0.npy',
                *[stem + '.png' for stem in VIEWS.values()]]
    for name in required:
        if not (dataset / name).is_file():
            raise FileNotFoundError(dataset / name)
    actions = np.load(dataset / 'actions_0.npy', allow_pickle=False).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise ValueError('GE-Sim actions must be finite [T,16] in native WM layout')
    shapes = {'intrinsic.npy': (3, 3, 3),
              'extrinsic_alignstate_0.npy': (3, len(actions), 4, 4),
              'state_joints_0.npy': (len(actions), 20),
              'state_eef_poses_0.npy': (len(actions), 14)}
    for name, shape in shapes.items():
        values = np.load(dataset / name, allow_pickle=False)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid calibrated bundle array {name}; expected finite {shape}')
    if chunks == 'all' and config.get('schema_version') != 3:
        raise ValueError('Full GE-Sim episode requires schema_version=3 for an unpadded final chunk')
    count = len(actions) if chunks == 'all' else chunks * 25
    if len(actions) < count:
        raise ValueError(f'Requested {count} actions but only {len(actions)} exist; no padding')
    views = {}
    for name, stem in VIEWS.items():
        with Image.open(dataset / f'{stem}.png') as image:
            views[name] = np.asarray(image.convert('RGB'))
    if any(image.shape != (384, 512, 3) for image in views.values()):
        raise ValueError('Native calibrated bundle requires 384x512 RGB views; do not resize calibration')
    source = {'conversion': 'Official WM-layout absolute joint commands unchanged; native RGB/calibration; action-mode FK trajectory bands',
              'dataset': str(dataset), 'conditioning': 'action',
              'instruction': (dataset / 'task.txt').read_text().strip(),
              'actions_available': len(actions), 'actions_selected': count,
              'unselected_actions': len(actions) - count, 'reference_available': False,
              'files_sha256': {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest() for name in required},
              'scope': 'Native action-mode replay; static torso/head assumption; not episode-mode parity or full test set'}
    selected = [actions[start:min(start+25, count)] for start in range(0, count, 25)]
    source['native_internal_tail_policy'] = 'Server repeats last action token to 25 internally, returns only requested frames; final call only'
    packet = {'protocol': config['action'], 'actions': selected if config.get('schema_version') == 3 else np.stack(selected),
              'reset_args': {'value': str(dataset)}, 'initial_views': views, 'source': source}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_packet(output, packet)
    return {'ok': True, 'model_inference_performed': False, 'output': str(output),
            'action_shape': [list(chunk.shape) for chunk in selected] if config.get('schema_version') == 3 else [len(selected), 25, 16], 'reference_available': False, 'source': source}
