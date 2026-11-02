#!/usr/bin/env python3
"""Prepare the A2World official example and run it through shared replay workers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from evaluation.datasets.a2world import prepare_official
    from evaluation.execution.native_replay import run_native_replay
    from evaluation.execution.reliability import run_signal_handlers
    from vlaworldeval.paths import fresh_output_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, default=ROOT / 'datasets/a2world/official/droid')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/world_models/a2world/a2world-pretrained.pt')
    parser.add_argument('--base-checkpoints', type=Path, default=ROOT / 'checkpoints/world_models/a2world/base')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sampling-steps', type=int, default=35)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--memory-gib', type=float, default=40)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--admit-only', action='store_true')
    args = parser.parse_args()
    if args.gpu < 0 or args.sampling_steps < 1:
        parser.error('GPU must be nonnegative; sampling steps must be positive')
    output = fresh_output_path(ROOT, args.output)
    config = json.loads((ROOT / 'configs/replay/a2world_pretrained_native_v1.json').read_text())
    for key, path in (('checkpoint', args.checkpoint), ('base_checkpoints', args.base_checkpoints),
                      ('tokenizer_path', args.base_checkpoints / 'nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth')):
        config['kwargs'][key] = {'project_path': str(path.resolve().relative_to(ROOT))}
    config.update(gpus=[args.gpu], memory_gib=args.memory_gib, seed=args.seed)
    config['kwargs'].update(num_sampling_steps=args.sampling_steps, seed=args.seed)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    prepared = prepare_official(config_path, args.asset_dir, output / 'input.npz')
    (output / 'prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps(prepared, indent=2))
        return
    with run_signal_handlers():
        result = run_native_replay(config_path, output / 'input.npz', output / 'replay', admit_only=args.admit_only)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
