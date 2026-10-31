"""Cosmos official JSON inference entry versus adapter; shared resident pipeline."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.support import digest
from evaluation.validation.parity import compare_traces


def capture_cosmos(root, output, packet_path, *, chunks=2, seed=0):
    import av
    from PIL import Image
    import torch.distributed as dist
    from adapters.world_models.cosmos3_droid_native import Cosmos3DroidFDNativeAdapter
    root, output = Path(root), Path(output).absolute()
    os.environ.update(COSMOS_TRAINING='0', COSMOS_INTERNAL='0')
    sys.path.insert(0, str(root/'third_party/cosmos-framework'))
    from cosmos_framework.inference.common.init import init_script
    init_script()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()
    wm = Cosmos3DroidFDNativeAdapter(
        model_path=root/'checkpoints/world_models/cosmos3_nano_droid_fd/community_jere_mybao_v1_20k_ema',
        runtime_dir=output/'shared_model', diffusion_cache=False, num_inference_steps=30,
        guidance=1., shift=10., seed=seed)
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.scripts.inference import inference
    packet = read_packet(packet_path)
    actions = packet['actions'][:chunks]
    if len(actions) != chunks:
        raise ValueError('Requested chunks exceed the frozen packet')
    image = packet['reset_args']['value']
    sources = ['cosmos_framework/scripts/inference.py', 'cosmos_framework/inference/args.py',
               'cosmos_framework/inference/defaults/forward_dynamics/sample_args.json']
    lock = {'source_sha256': {s: digest(wm.framework_root/s) for s in sources},
            'checkpoint': {p.name: digest(p) for p in wm.model_path.glob('*.json')},
            'input_sha256': digest(packet_path)}
    if rank == 0:
        (output/'assets.json').write_text(json.dumps(lock, indent=2))
    if dist.is_initialized():
        dist.barrier()
    # Full effective sample arguments are retained; compare operational paths
    # separately from conditioning pixels/actions and numerical settings.
    conditions, generated = [], []
    original = wm._pipe.generate
    def generate(samples, *args, **kwargs):
        for sample in samples:
            payload = sample.model_dump(mode='json')
            conditions.append(canonical_sample(payload))
        result = original(samples, *args, **kwargs)
        generated[:] = result
        return result
    wm._pipe.generate = generate
    try:
        for route in ('native', 'adapter'):
            conditions.clear()
            videos = []
            current = Image.fromarray(image)
            wm.runtime_dir = output/route
            wm.reset(image)
            for index, action in enumerate(actions):
                if route == 'adapter':
                    video = wm.rollout(action)
                else:
                    directory = output/route/f'step_{index:05d}'
                    if rank == 0:
                        directory.mkdir(parents=True)
                        current.save(directory/'condition.png')
                        (directory/'actions.json').write_text(json.dumps(action.tolist()))
                        sample = dict(name=f'step_{index:05d}', model_mode='forward_dynamics',
                            vision_path=str(directory/'condition.png'), action_path=str(directory/'actions.json'),
                            domain_name='droid_lerobot', action_chunk_size=16, image_size=480,
                            aspect_ratio='4,3', resolution='480', fps=15, num_frames=17,
                            num_steps=30, guidance=1., shift=10., prompt='', seed=seed+index)
                        (directory/'sample.json').write_text(json.dumps(sample))
                    if dist.is_initialized():
                        dist.barrier()
                    setup = OmniSetupOverrides(checkpoint_path=str(wm.model_path),
                        config_file=str(output/'shared_model/config.local.json'), output_dir=output/route/'outputs',
                        guardrails=False, use_torch_compile=False, diffusion_cache=False,
                        parallelism_preset='latency', dp_shard_size=int(os.environ.get('WORLD_SIZE','1')),
                        cp_size=wm.parallelism_contract['context_parallel_degree'], cfgp_size=1, max_num_seqs=1)
                    # Invoke the pinned public entry; replace only model creation.
                    # This excludes independent checkpoint-loading verification.
                    with patch.object(type(wm._pipe), 'create', return_value=wm._pipe):
                        inference(SimpleNamespace(setup=setup, input_files=[directory/'sample.json']))
                    paths = [None]
                    if rank == 0:
                        paths[0] = str(next(Path(p) for p in generated[0].outputs[0].files if Path(p).suffix == '.mp4'))
                    if dist.is_initialized():
                        dist.broadcast_object_list(paths, src=0)
                    path = Path(paths[0])
                    with av.open(str(path)) as container:
                        full = np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
                    if len(full) != 17:
                        raise ValueError('Expected official inclusive 17-frame output')
                    video = full[1:]
                    current = Image.fromarray(video[-1])
                videos.append(video)
            if rank == 0:
                trace = {'schema_version': 1, 'profile': 'cosmos_droid_community20k', 'protocol': 'cosmos_droid_native_v1',
                    'route': route, 'seed': seed, 'asset_lock_sha256': digest(output/'assets.json'),
                    'native_revision': lock['source_sha256'],
                    'inputs': {'image': image, 'actions': actions, 'effective_sample_args': list(conditions)},
                    'outputs': {'frames': np.stack(videos)}, 'provenance': {'shared_weights': True,
                        'source_sha256': lock['source_sha256'], 'parallelism': wm.parallelism_contract,
                        'adapter_sha256': digest(root/'adapters/world_models/cosmos3_nano_droid_fd.py'),
                        'scope': 'Official JSON entry with local compatibility config and shared pipeline; not independent model loading or official NVIDIA DROID weights'}}
                write_packet(output/f'{route}.npz', trace)
    finally:
        wm._pipe.generate = original
    result = None
    if rank == 0:
        policy = {'profile': 'cosmos_droid_community20k', 'protocol': 'cosmos_droid_native_v1',
                  'atol': 0, 'rtol': 0, 'scope': 'Official JSON entry versus adapter, shared weights and local runtime config; operational path fields excluded, full effective numerical settings compared.'}
        (output/'policy.json').write_text(json.dumps(policy, indent=2))
        result = compare_traces(output/'native.npz', output/'adapter.npz', output/'policy.json', output/'report.json')
    if dist.is_initialized():
        values = [result]
        dist.broadcast_object_list(values, src=0)
        result = values[0]
        dist.destroy_process_group()
    return result


def canonical_sample(value, *, _top=True):
    """Exclude only operational names/paths; pixels/actions are compared separately."""
    if isinstance(value, dict):
        return {k: canonical_sample(v, _top=False) for k, v in value.items()
                if not _top or k not in {'name', 'output_dir', 'vision_path', 'action_path'}}
    if isinstance(value, list):
        return [canonical_sample(v, _top=False) for v in value]
    return value
