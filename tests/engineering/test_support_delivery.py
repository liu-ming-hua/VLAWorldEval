"""Public evidence must not promote old contracts or CPU checks to model passes."""
import json
from pathlib import Path
import shutil

import pytest

from evaluation.manifests import load_pair_manifest
from evaluation.reporting.support import support_report
from scripts.acceptance.first_rollout import clean_environment, verify_assets
from scripts.acceptance.support_evidence import import_run

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def fixture_root(tmp_path):
    (tmp_path / 'configs/pairs').mkdir(parents=True)
    shutil.copy(ROOT / 'configs/runtime_registry.json', tmp_path / 'configs/runtime_registry.json')
    shutil.copy(ROOT / 'configs/pairs/openvla__worldgym.json', tmp_path / 'configs/pairs/openvla__worldgym.json')
    return tmp_path


def test_registration_does_not_imply_any_verification(tmp_path):
    report = support_report(fixture_root(tmp_path))
    row, = report['pairs']
    assert row['registered']
    assert row['real_weight_loop']['status'] == 'not_verified'
    assert row['fresh_install']['status'] == 'not_verified'
    assert row['official_numerical_parity'] == 'not_verified'
    assert not report['execution_performed']


def test_old_contract_is_not_promoted_to_new_contract(tmp_path):
    root = fixture_root(tmp_path)
    contract = load_pair_manifest(root / 'configs/pairs/openvla__worldgym.json').to_dict()
    pair_id = contract['pair_id']
    old = json.loads(json.dumps(contract))
    old['vla']['action']['frame'] = 'different_frame'
    old['action_bridge']['source']['frame'] = 'different_frame'
    path = root / 'configs/support_evidence_v1.json'
    path.write_text(json.dumps({'pairs': {pair_id: {'real_weight_loop': {
        'status': 'historical_pass', 'pair_contract': old}}}}))
    row, = support_report(root)['pairs']
    assert row['current_contract_matches_evidence'] is False
    assert row['current_source_inference'] == 'not_verified'
    old['pair_id'] += '_previous'
    path.write_text(json.dumps({'pairs': {old['pair_id']: {'real_weight_loop': {
        'status': 'historical_pass', 'pair_contract': old}}}}))
    row, = support_report(root)['pairs']
    assert row['real_weight_loop']['status'] == 'not_verified'
    assert row['other_contract_evidence'] == [old['pair_id']]


def test_default_fields_and_descriptive_metadata_are_not_contract_changes(tmp_path):
    root = fixture_root(tmp_path)
    contract = json.loads((root / 'configs/pairs/openvla__worldgym.json').read_text())
    contract['metadata'] = {'note': 'historical description'}
    (root / 'configs/support_evidence_v1.json').write_text(json.dumps({'pairs': {
        contract['pair_id']: {'real_weight_loop': {'status': 'historical_pass', 'pair_contract': contract}}
    }}))
    assert support_report(root)['pairs'][0]['current_contract_matches_evidence'] is True


def test_preflight_is_rejected_as_run_evidence(tmp_path):
    report = tmp_path / 'preflight.json'
    report.write_text(json.dumps({'ok': True, 'status': 'checked_not_executed', 'execution_performed': False}))
    with pytest.raises(ValueError, match='completed executed'):
        import_run(tmp_path, report)


def test_clean_recipe_drops_ambient_environment(monkeypatch, tmp_path):
    for name in ['HTTP_PROXY', 'https_proxy', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'VLAWORLDEVAL_ENVIRONMENTS', 'WORLDGYM_VAE_PATH']:
        monkeypatch.setenv(name, 'unwanted')
    env = clean_environment(tmp_path)
    assert not any(k.lower().endswith('_proxy') for k in env)
    assert 'VLAWORLDEVAL_ENVIRONMENTS' not in env and 'PYTHONPATH' not in env
    assert env['HF_HUB_OFFLINE'] == '1'
    assert env['HF_ENDPOINT'] == 'https://hf-mirror.com'
    assert env['PYTHONNOUSERSITE'] == '1'
    assert env['WORLDGYM_VAE_PATH'] == str(tmp_path / 'checkpoints/world_models/worldgym/vae/sd3_vae')


def test_fixed_asset_hash_catches_same_size_replacement(tmp_path):
    import hashlib
    (tmp_path / 'model').write_bytes(b'abc')
    manifest = {'files': {'model': {'size': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}}}
    assert verify_assets(tmp_path, manifest)['status'] == 'passed'
    (tmp_path / 'model').write_bytes(b'xyz')
    with pytest.raises(ValueError, match='changed asset'):
        verify_assets(tmp_path, manifest)
    with pytest.raises(ValueError, match='Empty asset lock'):
        verify_assets(tmp_path, {'files': {}})


def test_fresh_source_identity_is_invalidated_by_code_changes(tmp_path):
    import hashlib
    from evaluation.reporting.support import source_matches
    (tmp_path / 'module.py').write_bytes(b'original')
    fingerprint = hashlib.sha256(b'module.py\0' + hashlib.sha256(b'original').digest()).hexdigest()
    record = {'fresh_install': {'framework_source': {'files': ['module.py'], 'sha256': fingerprint}}}
    assert source_matches(tmp_path, record) is True
    (tmp_path / 'module.py').write_bytes(b'changed')
    assert source_matches(tmp_path, record) is False
    (tmp_path / 'module.py').unlink()
    assert source_matches(tmp_path, record) is False


def test_runtime_acceptance_receipt_does_not_upgrade_pair_or_changed_sources(tmp_path):
    from evaluation.reporting.support import digest
    root = fixture_root(tmp_path)
    adapter = root / 'adapter.py'
    adapter.write_text('version1')
    path = root / 'configs/world_model_acceptance_v1.json'
    path.write_text(json.dumps({'records': [{'model': 'fixture', 'status': 'historical_pass',
                                           'adapter_sources': {'adapter.py': digest(adapter)}}]}))
    result = support_report(root)
    assert result['runtime_acceptance'][0]['current_adapter_sources_match']
    assert result['pairs'][0]['official_numerical_parity'] == 'not_verified'
    adapter.write_text('version2')
    assert not support_report(root)['runtime_acceptance'][0]['current_adapter_sources_match']
