"""Import audited real runs, or render/check the public support tables."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.reporting.artifact_checks import audit_attempt  # noqa: E402
from evaluation.reporting.support import digest, render_table, support_report  # noqa: E402
from evaluation.planning.registry import project_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
START, END = '<!-- support-evidence:start -->', '<!-- support-evidence:end -->'


def import_run(root, path):
    path = project_path(root, str(path))
    value = json.loads(path.read_text())
    if (value.get('ok') is not True or value.get('execution_performed') is not True
            or value.get('status') != 'completed'):
        raise ValueError('Only completed executed evaluate reports are evidence')
    execution = value['execution']
    for job in execution['jobs']:
        project_path(root, job['output'])
    audit = audit_attempt(execution, expected_jobs=value['preflight']['selection'])
    if not audit['ok']:
        raise ValueError(f'Artifact re-audit failed: {path}: {audit}')
    jobs = [j for j in execution['jobs'] if j.get('vla_calls', 0) > 0]
    if not jobs:
        raise ValueError('Replay-only evidence does not validate a policy loop')
    contracts = [json.loads((Path(j['output']) / 'metadata.json').read_text())['pair_manifest'] for j in jobs]
    if any(p != contracts[0] for p in contracts):
        raise ValueError('One evidence record must describe one pair contract')
    expected_adapters = {contracts[0]['vla']['adapter'], contracts[0]['world_model']['adapter']}
    loaded_adapters = {w.get('adapter') for w in execution.get('worker_loads', []) if w.get('ok') is True}
    if not expected_adapters.issubset(loaded_adapters):
        raise ValueError('Missing successful real-adapter load receipts')
    attempt = project_path(root, execution['attempt'])
    workers = json.loads((attempt / 'resolved_workers.json').read_text())
    def portable(value):
        if isinstance(value, str):
            return value.replace(str(root.resolve()) + '/', '')
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value
    return contracts[0]['pair_id'], {
        'status': 'historical_pass',
        'attempt': attempt.name,
        'report': str(path.relative_to(root)), 'report_sha256': digest(path),
        'artifact_rechecked_at': datetime.now(timezone.utc).isoformat(),
        'pair_contract': contracts[0],
        'max_chunks': max(j['completed_chunks'] for j in jobs),
        'jobs': [{k: j[k] for k in ('id', 'episode', 'seed', 'completed_chunks', 'vla_calls', 'world_model_calls')} for j in jobs],
        'artifact_hashes': [{
            'directory': str(Path(j['output']).relative_to(root)),
            'metadata_sha256': digest(Path(j['output']) / 'metadata.json'),
            'actions_sha256': digest(Path(j['output']) / 'actions.npz'),
        } for j in jobs],
        'loaded_models': portable({k: workers[k] for k in ('vla', 'world_model')}),
        'package_versions': [{
            'adapter': w['adapter'], 'python': w.get('python'),
            'packages': {k: v for k, v in w.get('packages', {}).items()
                         if k in {'torch', 'transformers', 'diffusers', 'jax', 'tensorflow', 'vlaworldeval'}}
        } for w in execution.get('worker_loads', [])],
        'source_version': 'Historical execution; report hash identifies evidence, not current code validation',
        'checkpoint_identity': 'Recorded paths; full checkpoint digests not asserted by this historical receipt',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', default=[])
    parser.add_argument('--fresh-report', type=Path, help='Successful first_rollout.py report.json')
    parser.add_argument('--check', action='store_true', help='Check generated README tables without modifying')
    args = parser.parse_args()
    catalog = ROOT / 'configs/support_evidence_v1.json'
    value = json.loads(catalog.read_text()) if catalog.exists() else {'schema_version': 1, 'pairs': {}}
    if args.check and (args.report or args.fresh_report):
        parser.error('--check cannot import reports')
    for path in args.report:
        pair, record = import_run(ROOT, path)
        value['pairs'].setdefault(pair, {})['real_weight_loop'] = record
    if args.fresh_report:
        fresh_path = project_path(ROOT, str(args.fresh_report))
        fresh = json.loads(fresh_path.read_text())
        required = {'core_venv', 'models_venv', 'core_check', 'models_check',
                    'installed_import', 'model_import', 'evaluate', 'inspect'}
        passed = {c['name'] for c in fresh.get('checks', []) if c.get('returncode') == 0}
        if (fresh.get('status') != 'passed' or not fresh.get('fresh_core_and_model_install_verified')
                or not required.issubset(passed)):
            raise ValueError('Incomplete fresh-install evidence')
        pair, record = import_run(ROOT, fresh_path.parent / fresh['execution_report'])
        wheel, = (fresh_path.parent / 'distributions').glob('*.whl')
        if digest(wheel) != fresh['wheel_sha256']:
            raise ValueError('Installed wheel evidence changed')
        with zipfile.ZipFile(wheel) as archive:
            names = sorted(n for n in archive.namelist() if n.endswith('.py')
                           and n.split('/')[0] in {'adapters', 'cli', 'evaluation', 'vlaworldeval'})
            fingerprint = hashlib.sha256()
            for name in names:
                fingerprint.update(name.encode() + b'\0' + hashlib.sha256(archive.read(name)).digest())
        value['pairs'].setdefault(pair, {})['fresh_install'] = {
            'status': 'historical_pass', 'started_at': fresh['started_at'],
            'report': str(fresh_path.relative_to(ROOT)), 'report_sha256': digest(fresh_path),
            'wheel_sha256': fresh['wheel_sha256'], 'asset_lock_sha256': fresh['asset_lock_sha256'],
            'framework_source': {'files': names, 'sha256': fingerprint.hexdigest()},
            'assets_reused': fresh['assets_reused'], 'package_cache_allowed': fresh['package_cache_allowed'],
            'cold_asset_download_verified': False,
        }
        value['pairs'][pair]['latest_fresh_loop'] = record
    if args.report or args.fresh_report:
        catalog.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    report = support_report(ROOT)
    for name, chinese in [('README.md', False), ('README_CN.md', True)]:
        path = ROOT / name
        original = path.read_text()
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError(f'Missing unique support table markers: {name}')
        before, rest = original.split(START)
        _, after = rest.split(END)
        updated = before + START + '\n' + render_table(report, chinese=chinese) + '\n' + END + after
        if args.check:
            if updated != original:
                raise ValueError(f'Stale table: {name}; run scripts/acceptance/support_evidence.py')
        else:
            path.write_text(updated)
    print(json.dumps({'pairs': len(report['pairs']), 'checked': args.check}))


if __name__ == '__main__':
    main()
