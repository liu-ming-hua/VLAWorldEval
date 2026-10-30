"""Fresh wheel/core/model installation and a fixed real OpenVLA × WorldGym loop.

Run after explicit asset preparation. Never installs into an existing environment,
never imports a model in the parent, never downloads during inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import signal
import time
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.reporting.support import digest  # noqa: E402
from vlaworldeval.worldgym_runtime import verify_runtime  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def verify_assets(root, manifest):
    if not manifest.get('files'):
        raise ValueError('Empty asset lock')
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Asset leaves workspace: {relative}')
        if (not path.is_file() or path.stat().st_size != expected['size']
                or digest(path) != expected['sha256']):
            raise ValueError(f'Missing or changed asset: {relative}; see docs/en/first-rollout.md')
    return {'status': 'passed', 'files': len(manifest['files']),
            'total_bytes': sum(f['size'] for f in manifest['files'].values())}


def clean_environment(root):
    env = dict(os.environ)
    for name in list(env):
        if name.lower().endswith('_proxy') or name in {
                'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'VLAWORLDEVAL_ENVIRONMENTS',
                'VLAWORLDEVAL_ROOT', 'CUDA_VISIBLE_DEVICES'}:
            env.pop(name, None)
    env.update(UV_CACHE_DIR=str(root / '.uv-cache'), UV_PYTHON_DOWNLOADS='never',
               UV_HTTP_TIMEOUT='600', HF_ENDPOINT='https://hf-mirror.com',
               UV_INDEX_URL=os.environ.get('PYPI_MIRROR', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
               HF_HOME=str(root / '.cache/huggingface'), PYTHONNOUSERSITE='1',
               WORLDGYM_VAE_PATH=str(root / 'checkpoints/world_models/worldgym/vae/sd3_vae'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               GIT_CONFIG_COUNT='2', GIT_CONFIG_KEY_0='http.proxy', GIT_CONFIG_VALUE_0='',
               GIT_CONFIG_KEY_1='https.proxy', GIT_CONFIG_VALUE_1='')
    return env


def run(output, *, vla_gpu, wm_gpu, offline=False):
    if vla_gpu < 0 or wm_gpu < 0 or vla_gpu == wm_gpu:
        raise ValueError('Select two distinct nonnegative GPU indices (18 + 20 GiB budgets)')
    output = output.absolute()
    if not output.resolve().is_relative_to(ROOT) or output.is_symlink():
        raise ValueError('Use a new output directory inside this project')
    uv = shutil.which('uv')
    if not uv or not shutil.which('ffmpeg'):
        raise ValueError('uv and FFmpeg must be installed before running this recipe')
    output.mkdir(parents=True, exist_ok=False)
    runtime_dir = ROOT / 'environment/verification' / ('first_rollout_' + hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:12])
    if not runtime_dir.resolve().is_relative_to(ROOT) or runtime_dir.exists():
        raise ValueError('Fresh runtime directory must be new and inside the project')
    env = clean_environment(ROOT)
    flags = ['--offline'] if offline else []
    report = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'fresh_wheel_core_and_model_envs_with_explicit_content_verified_assets',
              'assets_reused': True, 'cold_asset_download_verified': False,
              'package_cache_allowed': True, 'offline_install': offline,
              'runtime_directory': str(runtime_dir.relative_to(ROOT)),
              'proxy_disabled': True, 'index_url': env['UV_INDEX_URL'],
              'host_library_override': env.get('LD_PRELOAD'),
              'scientific_quality_verified': False, 'checks': []}
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    def execute(name, command, *, execution_env=None):
        print(f'[{name}]', flush=True)
        started = time.monotonic()
        process = None
        try:
            with (output / f'{name}.stdout').open('w') as stdout, (output / f'{name}.stderr').open('w') as stderr:
                process = subprocess.Popen(list(map(str, command)), cwd=output, env=execution_env or env,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                process.wait(timeout=3600)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            report['checks'].append({'name': name, 'returncode': process.returncode if process else None,
                                     'seconds': time.monotonic() - started,
                                     'command': list(map(str, command))})
            save()
        if process.returncode:
            raise ValueError(f'{name} failed; inspect {output / (name + ".stderr")}')
    save()
    try:
        manifest = ROOT / 'configs/first_rollout_assets_v1.json'
        report['assets'] = verify_assets(ROOT, json.loads(manifest.read_text()))
        report['asset_lock_sha256'] = digest(manifest)
        verify_runtime(ROOT)
        execute('build', [uv, 'build', ROOT, '--wheel', '--no-build-isolation', '--python', sys.executable,
                          '--out-dir', output / 'distributions', *flags])
        wheel, = (output / 'distributions').glob('*.whl')
        report['wheel_sha256'] = digest(wheel)
        runtime_dir.mkdir(parents=True, exist_ok=False)
        for name, lock in [('core', 'core-py311.txt'), ('models', 'worldgym-openvla-py311.txt')]:
            python = runtime_dir / name / 'bin/python'
            execute(f'{name}_venv', [uv, 'venv', runtime_dir / name, '--python', sys.executable, *flags])
            execute(f'{name}_dependencies', [uv, 'pip', 'install', '--python', python,
                    '--index-url', env['UV_INDEX_URL'], '--link-mode', 'hardlink', *flags,
                    '-r', ROOT / 'environment/locks' / lock])
            execute(f'{name}_wheel', [uv, 'pip', 'install', '--python', python, '--no-deps', *flags, wheel])
            execute(f'{name}_check', [uv, 'pip', 'check', '--python', python])
            execute(f'{name}_freeze', [uv, 'pip', 'freeze', '--python', python])
        core = runtime_dir / 'core/bin/python'
        execute('installed_import', [core, '-c',
                'import cli,importlib.util; assert "/core/" in cli.__file__; '
                'assert importlib.util.find_spec("torch") is None; print(cli.__file__)'])
        execute('model_import', [runtime_dir / 'models/bin/python', '-c',
                'import cli,torch,transformers,diffusers; assert "/models/" in cli.__file__; '
                'print(cli.__file__,torch.__version__,transformers.__version__,diffusers.__version__)'])
        suite = json.loads((ROOT / 'configs/experiments/openvla_worldgym_v1.json').read_text())
        suite['suite_id'] = 'first_real_rollout_v1'
        suite['runtime'].update(vla_gpus=[vla_gpu], world_model_gpus=[wm_gpu])
        suite['output_root'] = str((output / 'rollouts').relative_to(ROOT))
        suite['experiments'] = [e for e in suite['experiments'] if e['id'] == 'loop_long']
        suite['experiments'][0]['chunks'] = 20
        config = output / 'suite.json'
        config.write_text(json.dumps(suite, indent=2) + '\n')
        mapping = output / 'environments.json'
        mapping.write_text(json.dumps({'schema_version': 1, 'paths': {
            'vla.openvla': str(runtime_dir / 'models'), 'world_models.worldgym': str(runtime_dir / 'models')}}))
        inference_env = {**env, 'VLAWORLDEVAL_ROOT': str(ROOT), 'VLAWORLDEVAL_ENVIRONMENTS': str(mapping)}
        command = [core, '-m', 'cli', 'evaluate', '--config', config, '--experiment', 'loop_long']
        execute('preflight', [*command, '--check-only'], execution_env=inference_env)
        execute('evaluate', [*command, '--execute'], execution_env=inference_env)
        result = json.loads((output / 'evaluate.stdout').read_text())
        if result.get('ok') is not True or not result.get('execution_performed'):
            raise ValueError('Real execution and artifact audit required')
        jobs = result['execution']['jobs']
        if len(jobs) != 1 or any(j.get(k) != 20 for j in jobs for k in ['completed_chunks', 'vla_calls', 'world_model_calls']):
            raise ValueError('Expected exactly 20 real policy/WM calls')
        artifact = Path(jobs[0]['output'])
        import numpy as np
        metadata = json.loads((artifact / 'metadata.json').read_text())
        if metadata['num_video_frames'] != 21 or metadata['video_fps'] != 20:
            raise ValueError('Expected 21 frames at declared 20 FPS')
        with np.load(artifact / 'actions.npz', allow_pickle=False) as actions:
            for key in ('native_actions', 'bridged_actions'):
                if actions[key].shape != (20, 1, 10) or not np.isfinite(actions[key]).all():
                    raise ValueError('Expected finite [20,1,10] action arrays')
        report['expected_artifacts_verified'] = {'frames': 21, 'fps': 20, 'actions_shape': [20, 1, 10]}
        execute('inspect', [core, '-m', 'cli', 'inspect', '--artifact', artifact], execution_env=inference_env)
        report.update(status='passed', execution_report='evaluate.stdout',
                      artifact=str(artifact.relative_to(ROOT)), real_weight_calls={'vla':20,'world_model':20},
                      fresh_core_and_model_install_verified=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vla-gpu', type=int, required=True)
    parser.add_argument('--wm-gpu', type=int, required=True)
    parser.add_argument('--offline', action='store_true', help='Require cached package archives; still create fresh environments')
    args = parser.parse_args()
    print(json.dumps(run(args.output, vla_gpu=args.vla_gpu, wm_gpu=args.wm_gpu, offline=args.offline), indent=2))
