import json

import numpy as np
import pytest

from evaluation.execution.transport import write_packet
from evaluation.validation.parity import compare_traces, compare_values

pytestmark = pytest.mark.unit


def traces(tmp_path):
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'profile': 'fixture', 'protocol': 'v1', 'atol': .1, 'rtol': 0}))
    base = {'schema_version': 1, 'profile': 'fixture', 'protocol': 'v1', 'seed': 0,
            'asset_lock_sha256': 'fixture', 'native_revision': 'fixture', 'provenance': {'synthetic': True},
            'inputs': {'history': np.ones((2, 3), np.float32)}, 'outputs': {'actions': np.ones((2, 3), np.float32)}}
    paths = []
    for route in ['native', 'adapter']:
        path = tmp_path / f'{route}.npz'
        write_packet(path, {**base, 'route': route})
        paths.append(path)
    return policy, base, paths


def test_exact_capture_is_scoped_not_physical_validation(tmp_path):
    policy, _, paths = traces(tmp_path)
    r = compare_traces(*paths, policy, tmp_path / 'report.json')
    assert r['status'] == 'passed' and not r['physical_reliability_verified']
    assert r['output_metrics']['outputs.actions']['max_abs'] == 0
    with pytest.raises(FileExistsError):
        compare_traces(*paths, policy, tmp_path / 'report.json')


@pytest.mark.parametrize('change', ['input', 'seed', 'assets', 'dtype', 'output', 'nan'])
def test_protocol_inputs_and_outputs_cannot_silently_drift(tmp_path, change):
    policy, base, paths = traces(tmp_path)
    if change == 'input':
        base['inputs']['history'][0, 0] += .01
    if change == 'seed':
        base['seed'] = 1
    if change == 'assets':
        base['asset_lock_sha256'] = 'other'
    if change == 'dtype':
        base['outputs']['actions'] = base['outputs']['actions'].astype(np.float64)
    if change == 'output':
        base['outputs']['actions'][0, 0] += 1
    if change == 'nan':
        base['outputs']['actions'][0, 0] = np.nan
    write_packet(paths[1], {**base, 'route': 'adapter'})
    assert compare_traces(*paths, policy, tmp_path / 'r.json')['status'] == 'failed'


def test_declared_tolerance_does_not_apply_to_inputs():
    assert compare_values(np.array([1.]), np.array([1.01]), atol=.1)[0] == []
    assert compare_values({'a': np.array([1.])}, {'b': np.array([1.])})[0]
    assert compare_values(np.zeros(0), np.zeros(0))[0]
