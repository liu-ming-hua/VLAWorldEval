import copy
import json

import pytest

from evaluation.execution.branches import run_branches, validate_branches
from evaluation.execution.transport import write_packet
from evaluation.reporting.support import digest
from tests.support.builders import replay_fixture

pytestmark = pytest.mark.process


def branch_fixture(tmp_path, monkeypatch, *, fail=False):
    _, config, first, packet = replay_fixture(tmp_path, monkeypatch)
    packet['actions'] = packet['actions'][:2]
    write_packet(first, packet)
    second = tmp_path / 'second.npz'
    other = copy.deepcopy(packet)
    other['actions'] *= -1 if fail else 2
    write_packet(second, other)
    def ref(p):
        return {'path': p.name, 'sha256': digest(p)}
    manifest = tmp_path / 'branches.json'
    value = {'schema_version': 1, 'config': ref(config), 'seeds': [0, 1], 'families': [
        {'id': 'scene', 'initial_condition': {'kind': 'observation_only'}, 'branches': [
            {'id': 'a', 'input': ref(first), 'reference': None}, {'id': 'b', 'input': ref(second), 'reference': None}]}]}
    manifest.write_text(json.dumps(value))
    return manifest, value, second, other


def test_shared_real_worker_path_preserves_all_slots_and_unknown_truth(tmp_path, monkeypatch):
    manifest, *_ = branch_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'result'
    assert run_branches(tmp_path, manifest, output)['status'] == 'validated_not_executed'
    assert not output.exists()
    result = run_branches(tmp_path, manifest, output, execute=True)
    assert result['status'] == 'completed' and result['counts']['completed'] == 4
    assert result['counts']['reference_unknown'] == 4 and result['independent_families'] == 1
    assert len(result['branch_contrasts']) == 2
    assert all(r['status'] == 'unknown' for r in result['branch_contrasts'])
    assert all(r['status'] == 'scored' for r in result['prediction_sensitivity'])
    assert all(not r['physical_action_tracking_verified'] for r in result['prediction_sensitivity'])
    assert not result['physical_reliability_verified']


def test_absolute_state_controls_keep_anchor_overlap_and_native_tail():
    import numpy as np
    from evaluation.datasets.action_controls import ctrl_control_windows
    from evaluation.datasets.ctrl_world import inclusive_action_windows

    states = np.arange(10 * 7, dtype=np.float32).reshape(10, 7) + 1
    windows, _, tail = inclusive_action_windows(states)
    video = {'fps': 5, 'frames_per_chunk': 5, 'temporal_layout': 'inclusive',
             'final_chunk_valid_frames': tail}
    controls, order = ctrl_control_windows(windows, video, seed=0)
    hold, shuffle = controls['hold_initial_pose'], controls['shuffle_future']
    np.testing.assert_array_equal(hold, np.broadcast_to(states[0], windows.shape))
    assert np.all(hold != 0)  # Zero motion is not zero absolute pose.
    np.testing.assert_array_equal(shuffle[1:, 0], shuffle[:-1, -1])
    np.testing.assert_array_equal(shuffle[0, 0], states[0])
    unique = np.concatenate([shuffle[0, :1], shuffle[:, 1:].reshape(-1, 7)])[:len(states)]
    np.testing.assert_array_equal(unique[1:], states[order])
    assert set(order) == set(range(1, len(states)))
    np.testing.assert_array_equal(shuffle[-1, 1:], np.broadcast_to(unique[-1], (4, 7)))
    broken = windows.copy()
    broken[1, 0] += 1
    with pytest.raises(ValueError, match='overlapping'):
        ctrl_control_windows(broken, video)


def test_failed_branch_is_retained_and_other_slots_execute(tmp_path, monkeypatch):
    manifest, *_ = branch_fixture(tmp_path, monkeypatch, fail=True)
    result = run_branches(tmp_path, manifest, tmp_path / 'result', execute=True)
    assert result['planned'] == 4 and result['counts']['failed'] == 2
    assert result['counts']['completed'] == 2
    assert result['status'] == 'completed_with_failures'


@pytest.mark.parametrize('change', ['history', 'duplicate_action', 'checksum', 'seed', 'path'])
def test_branch_admission_rejects_bad_matching_and_mutation(tmp_path, monkeypatch, change):
    manifest, value, path, other = branch_fixture(tmp_path, monkeypatch)
    if change == 'history':
        other['reset_args']['value'][0, 0] = 1
    if change == 'duplicate_action':
        other['actions'][:] = 1
    if change == 'checksum':
        path.write_bytes(b'changed')
    else:
        write_packet(path, other)
        value['families'][0]['branches'][1]['input']['sha256'] = digest(path)
    if change == 'seed':
        value['seeds'] = [0, 0]
    if change == 'path':
        value['families'][0]['branches'][1]['input']['path'] = '../outside'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_branches(tmp_path, manifest)


@pytest.mark.parametrize('misaligned', [False, True])
def test_paired_truth_scoring_requires_aligned_references(tmp_path, monkeypatch, misaligned):
    import numpy as np
    from evaluation.datasets.native_reference import save_reference
    manifest, value, *_ = branch_fixture(tmp_path, monkeypatch)
    for i, branch in enumerate(value['families'][0]['branches']):
        frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        frames[1:] = 10 * (i + 1)
        path = tmp_path / f'truth{i}.npz'
        save_reference(tmp_path / branch['input']['path'], path,
                       {v: frames for v in ('front', 'wrist')},
                       {v: np.arange(5) * (2 if misaligned and i else 1) for v in ('front', 'wrist')},
                       domain='synthetic_test_fixture', split='test', episode=0,
                       conversion='synthetic fixture only', condition_kind='recorded_command')
        branch['reference'] = {'path': path.name, 'sha256': digest(path)}
    manifest.write_text(json.dumps(value))
    if misaligned:
        with pytest.raises(ValueError, match='timelines differ'):
            validate_branches(tmp_path, manifest)
    else:
        report = run_branches(tmp_path, manifest, tmp_path / 'result', execute=True)
        assert report['status'] == 'completed'
        assert all(row['status'] == 'scored' for row in report['branch_contrasts'])
        assert all(row['views']['front']['frames'] == 4 for row in report['branch_contrasts'])
