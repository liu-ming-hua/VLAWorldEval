#!/usr/bin/env python3
"""Managed GE-Sim action replay: prepare, start owned server, replay, report, stop."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from adapters.environment import get_environment
    from evaluation.datasets.ge_sim import prepare_official
    from evaluation.execution.admission import native_gpu_lease, resource_admission
    from evaluation.execution.local_server import local_server
    from evaluation.execution.native_replay import run_native_replay
    from evaluation.execution.reliability import run_signal_handlers
    from vlaworldeval.paths import fresh_output_path
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=ROOT / 'datasets/ge_sim_v2/official/demo_000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--memory-gib', type=float, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunks', default='all', help='all, or a positive number of complete chunks')
    parser.add_argument('--server-config', type=Path, default=ROOT / 'configs/world_models/ge_sim_v2.yaml')
    args = parser.parse_args()
    if args.gpu < 0 or not 0 <= args.seed < 2**32:
        parser.error('GPU must be nonnegative; seed must be uint32')
    if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error('memory-gib must be finite and positive')
    chunks = 'all' if args.chunks == 'all' else int(args.chunks)
    output = fresh_output_path(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'preparing', 'gpu': args.gpu, 'seed': args.seed,
              'memory_gib': args.memory_gib, 'conditioning': 'action',
              'scientific_quality_verified': False, 'downloads_performed': False}
    def save():
        (output / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config = json.loads((ROOT / 'configs/replay/ge_sim_v2_action_full_v1.json').read_text())
        config['kwargs']['server_url'] = f'http://127.0.0.1:{port}'
        config['seed'] = args.seed
        config_path = output / 'config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        prepared = prepare_official(config_path, args.episode, output / 'input.npz', chunks=chunks)
        (output / 'prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
        server_config = output / 'server.yaml'
        server_config.write_bytes(args.server_config.read_bytes())
        python = get_environment('world_models.ge_sim_v2').python
        # Seed the independently owned server, not merely the CPU client worker.
        bootstrap = ('import random,numpy as np,torch,runpy; '
                     f'random.seed({args.seed}); np.random.seed({args.seed}); torch.manual_seed({args.seed}); '
                     'torch.cuda.set_device(0); '
                     f'torch.cuda.set_per_process_memory_fraction(min(1,{args.memory_gib}*2**30/torch.cuda.get_device_properties(0).total_memory),0); '
                     'runpy.run_module("gesim.server",run_name="__main__")')
        command = [str(python), '-c', bootstrap, '--model', 'gesim_v2', '--config', str(server_config),
                   '--host', '127.0.0.1', '--port', str(port)]
        environment = dict(os.environ)
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            environment.pop(key, None)
            os.environ.pop(key, None)
        environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
                           HF_ENDPOINT='https://hf-mirror.com', MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'))
        patch = ROOT / 'scripts/setup/patches/ge_sim_v2_finite_tail.patch'
        subprocess.run(['git', '-C', str(ROOT / 'third_party/ge-sim-v2'),
                        'apply', '--reverse', '--check', str(patch)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record['compatibility_patch_sha256'] = digest(patch)
        record['upstream_commit'] = subprocess.check_output(
            ['git', '-C', str(ROOT / 'third_party/ge-sim-v2'), 'rev-parse', 'HEAD'], text=True).strip()
        record['server_config_sha256'] = digest(server_config)
        # Resolve the checkpoint with the server's YAML parser; no heavy model load.
        checkpoint = subprocess.check_output([str(python), '-c',
            'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["checkpoint"])', str(server_config)],
            cwd=ROOT, env=environment, text=True).strip()
        weight_root = Path(checkpoint) if Path(checkpoint).is_absolute() else ROOT / checkpoint
        if not (weight_root / 'model.safetensors').is_file():
            raise FileNotFoundError(weight_root / 'model.safetensors')
        record['server_assets_sha256'] = {str(path.relative_to(weight_root)): digest(path)
                                         for path in sorted(weight_root.rglob('*')) if path.is_file()}
        record['upstream_sources_sha256'] = {str(path.relative_to(ROOT)): digest(path)
            for path in sorted((ROOT / 'third_party/ge-sim-v2/src/gesim').rglob('*.py'))}
        with run_signal_handlers(), native_gpu_lease(ROOT, [args.gpu]):
            resource_admission({'world_model_gpus': [args.gpu],
                                'max_process_memory_gib_per_gpu': args.memory_gib}, False)
            with local_server(command, cwd=ROOT, environment=environment, log_path=output / 'server.log',
                              health_url=config['kwargs']['server_url'] + '/healthz') as server:
                record.update(status='running', server=server)
                save()
                result = run_native_replay(config_path, output / 'input.npz', output / 'replay')
                record.update(status=result['status'], world_model_calls=result['world_model_calls'],
                              frames_per_view=result['frames_per_view'], actions_consumed=result['actions_consumed'])
        from evaluation.reporting.chunk_boundaries import inspect_replay
        (output / 'boundary_diagnostics.json').write_text(json.dumps(inspect_replay(output / 'replay'), indent=2) + '\n')
        subprocess.run([sys.executable, '-m', 'cli', 'report', '--source', str(output / 'replay/summary.json'),
                        '--output', str(output / 'report')], cwd=ROOT, env=environment, check=True)
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
