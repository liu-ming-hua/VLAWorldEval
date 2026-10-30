"""Portable, conservative support evidence; registration never implies validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.manifests import PairManifest, load_pair_manifest
from evaluation.planning.registry import load_runtime_registry


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_matches(root, record):
    root = Path(root).resolve()
    frozen = record.get('fresh_install', {}).get('framework_source')
    if not frozen or not frozen.get('files'):
        return None
    fingerprint = hashlib.sha256()
    for relative in frozen['files']:
        path = root / relative
        if (path.is_absolute() and not path.resolve().is_relative_to(root.resolve())) or not path.is_file():
            return False
        fingerprint.update(relative.encode() + b'\0' + bytes.fromhex(digest(path)))
    return fingerprint.hexdigest() == frozen['sha256']


def support_report(root):
    root = Path(root)
    catalog = root / 'configs/support_evidence_v1.json'
    evidence = json.loads(catalog.read_text()) if catalog.exists() else {'pairs': {}}
    rows = []
    registry = load_runtime_registry(root)
    for path in sorted((root / 'configs/pairs').glob('*.json')):
        pair = load_pair_manifest(path)
        registry.pair_profile(pair)
        record = evidence['pairs'].get(pair.pair_id, {})
        historical = record.get('real_weight_loop', {})
        frozen = historical.get('pair_contract')
        # Compare semantic contracts, not JSON whitespace or default field spelling.
        current = json.loads(json.dumps(pair.to_dict()))
        if frozen:
            normalized = json.loads(json.dumps(PairManifest.from_dict(frozen).to_dict()))
            normalized.pop('metadata', None)
            current.pop('metadata', None)
            matches = normalized == current
        else:
            matches = None
        rows.append({
            'pair_id': pair.pair_id,
            'manifest': str(path.relative_to(root)),
            'registered': True,
            'vla': pair.vla.adapter, 'world_model': pair.world_model.adapter,
            'state_source': pair.state_provider.type,
            'latest_fresh_loop': record.get('latest_fresh_loop'),
            'fresh_install': record.get('fresh_install', {'status': 'not_verified'}),
            'real_weight_loop': historical or {'status': 'not_verified'},
            'current_contract_matches_evidence': matches,
            'official_input_alignment': record.get('official_input_alignment', 'declared_only_not_certified'),
            'last_tested_attempt': record.get('latest_fresh_loop', historical).get('attempt'),
            'official_numerical_parity': record.get('official_numerical_parity', 'not_verified'),
            'current_source_inference': ('matches_tested_wheel' if source_matches(root, record) is True
                                         else 'not_verified'),
            'other_contract_evidence': [key for key, item in evidence['pairs'].items()
                if key != pair.pair_id
                and item.get('real_weight_loop', {}).get('pair_contract', {}).get('vla', {}).get('adapter') == pair.vla.adapter
                and item.get('real_weight_loop', {}).get('pair_contract', {}).get('world_model', {}).get('adapter') == pair.world_model.adapter],
            'physical_reliability': 'not_verified',
        })
    return {'schema_version': 1, 'pairs': rows, 'execution_performed': False,
            'scope': 'Published historical evidence, not current machine readiness. '
                     'Use evaluate --check-only for local preflight; execute for a new run.'}


def render_table(report, *, chinese=False):
    headers = ('组合', '干净安装', '真实闭环', '历史 chunk 数', '状态来源') if chinese else (
        'Pair', 'Fresh install', 'Real-weight loop', 'Historical chunks', 'State source')
    lines = ['| ' + ' | '.join(headers) + ' |', '| --- | --- | --- | ---: | --- |']
    for row in report['pairs']:
        run = row['real_weight_loop']
        label = run.get('status', 'not_verified')
        if row['current_contract_matches_evidence'] is False:
            label += ' (contract changed)'
        lines.append('| ' + ' | '.join([row['pair_id'], row['fresh_install']['status'],
                     label, str(run.get('max_chunks', '—')), row['state_source']]) + ' |')
    return '\n'.join(lines)
