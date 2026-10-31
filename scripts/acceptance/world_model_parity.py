"""Real-weight native world-model checks; run in the explicitly selected model environment."""
import argparse
import json
import os
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ctrl_world', 'cosmos_droid'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=199)
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    for name in list(os.environ):
        if name.lower().endswith('_proxy'):
            os.environ.pop(name)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_ENDPOINT='https://hf-mirror.com', HF_HOME=str(ROOT/'.cache/huggingface'))
    from evaluation.planning.registry import project_path
    args.output = project_path(ROOT, str(args.output))
    if args.input is not None:
        args.input = project_path(ROOT, str(args.input))
    if args.chunks < 1:
        parser.error('--chunks must be positive')
    if args.model == 'ctrl_world':
        from evaluation.validation.ctrl_parity import capture_ctrl
        result = capture_ctrl(ROOT, args.output, episode=args.episode, chunks=args.chunks)
    else:
        if args.input is None:
            parser.error('--input is required for cosmos_droid')
        from evaluation.validation.cosmos_parity import capture_cosmos
        try:
            result = capture_cosmos(ROOT, args.output, args.input, chunks=args.chunks)
        except Exception:
            # Exit the failed rank promptly so torchrun terminates peers rather
            # than leaving them blocked in a collective/NCCL destructor.
            traceback.print_exc()
            if int(os.environ.get('WORLD_SIZE', '1')) > 1:
                sys.stderr.flush()
                os._exit(1)
            raise
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
