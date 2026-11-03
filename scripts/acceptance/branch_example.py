"""Prepare a recorded-vs-zero WorldGym branch example; no generation or invented truth."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.datasets.native_reference import prepare_case  # noqa: E402
from evaluation.execution.transport import read_packet, write_packet  # noqa: E402
from evaluation.planning.registry import project_path  # noqa: E402
from evaluation.reporting.support import digest  # noqa: E402


def prepare(output, gpu=0, chunks=3):
    output = project_path(ROOT, str(output))
    if gpu < 0 or chunks < 1:
        raise ValueError('Nonnegative GPU and positive chunk count required')
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'configs/replay/worldgym_bridge_single_step_v1.json').read_text())
    config['gpus'] = [gpu]
    cfg = output / 'config.json'
    cfg.write_text(json.dumps(config, indent=2))
    prepared = prepare_case('worldgym', ROOT / 'datasets/worldgym/official/bridge/test', cfg,
                            output / 'recorded', chunks=chunks, episode=2)
    packet = copy.deepcopy(read_packet(prepared['input']))
    packet['actions'][:] = 0
    zero = output / 'zero.npz'
    write_packet(zero, packet)
    def asset(path):
        path = Path(path)
        return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
    manifest = {'schema_version': 1, 'config': asset(cfg), 'seeds': [0, 1], 'families': [
        {'id': 'bridge_episode2', 'initial_condition': {'kind': 'observation_only'}, 'branches': [
            {'id': 'recorded', 'input': asset(prepared['input']), 'reference': asset(prepared['reference'])},
            {'id': 'zero', 'input': asset(zero), 'reference': None}]}]}
    path = output / 'branches.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return str(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--chunks', type=int, default=3)
    parser.add_argument('--model', choices=['worldgym', 'ctrl_world'], default='worldgym')
    parser.add_argument('--prepared', type=Path, help='Ctrl native case containing config/input/reference')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.model == 'ctrl_world':
        if args.prepared is None:
            parser.error('--prepared is required for Ctrl native controls')
        from evaluation.datasets.action_controls import prepare_ctrl_controls
        print(prepare_ctrl_controls(ROOT, args.prepared, args.output, gpu=args.gpu, seed=args.seed))
    else:
        print(prepare(args.output, args.gpu, args.chunks))
