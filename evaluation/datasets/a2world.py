"""Prepare the released pretrained DROID example without model dependencies."""
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2

from evaluation.execution.transport import write_packet


VIEWS = ('exterior_1', 'wrist', 'exterior_2')


def prepare_official(config_path, dataset, output):
    """One released 20-action chunk; no invented future RGB or action repetition."""
    config = json.loads(Path(config_path).read_text())
    if (config.get('adapter') != 'A2WorldAdapter'
            or config.get('kwargs', {}).get('variant') != 'pretrained'
            or config.get('action', {}).get('dimension') != 14
            or config['action'].get('horizon') != 20
            or config['action'].get('temporal_layout') != 'future'
            or config.get('output_views') != list(VIEWS)):
        raise ValueError('Requires the A2World pretrained 20x14 three-view future protocol')
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    action_file = dataset / 'actions.npz'
    with np.load(action_file, allow_pickle=False) as archive:
        if set(archive.files) != {'actions'}:
            raise ValueError('Expected the released actions-only archive; alternate state formats require their official converter')
        actions = np.asarray(archive['actions'], dtype=np.float32)
    if actions.shape != (20, 14) or not np.isfinite(actions).all():
        raise ValueError('Official example requires exactly 20x14 finite actions; no padding or truncation')
    views, hashes = {}, {}
    paths = [action_file]
    size = (config['kwargs'].get('width', 256), config['kwargs'].get('height', 256))
    for name in VIEWS:
        path = dataset / f'{name}.png'
        paths.append(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable official camera image: {path}')
        views[name] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
    for path in paths:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = {'protocol': config['action'], 'actions': actions[None],
              'reset_args': {'value': {'views': [views[name] for name in VIEWS], 'view_ids': [2, 0, 3]}},
              'initial_views': views,
              'source': {'conversion': 'Official pretrained DROID npz actions unchanged; OpenCV RGB INTER_AREA resize; view IDs 2/0/3',
                         'dataset': str(dataset), 'files_sha256': hashes,
                         'reference_available': False,
                         'scope': 'One official example chunk, not a long rollout or quality benchmark'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_packet(output, packet)
    return {'ok': True, 'model_inference_performed': False, 'output': str(output),
            'action_shape': [1, 20, 14], 'reference_available': False, 'source': packet['source']}
