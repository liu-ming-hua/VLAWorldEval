"""Recorded joint commands must not be confused with episode trajectory conditioning."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from evaluation.datasets.ge_sim import prepare_official, VIEWS
from evaluation.execution.transport import read_packet

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle(tmp_path):
    data = tmp_path / 'bundle'
    data.mkdir()
    actions = np.arange(51 * 16, dtype=np.float32).reshape(51, 16)
    np.save(data / 'actions_0.npy', actions)
    for name, shape in {'intrinsic': (3, 3, 3), 'extrinsic_alignstate_0': (3, 51, 4, 4),
                        'state_joints_0': (51, 20), 'state_eef_poses_0': (51, 14)}.items():
        np.save(data / f'{name}.npy', np.zeros(shape, np.float32))
    (data / 'task.txt').write_text('Pick up the object')
    for i, stem in enumerate(VIEWS.values()):
        Image.fromarray(np.full((384, 512, 3), 20 * i, np.uint8)).save(data / f'{stem}.png')
    config = ROOT / 'configs/replay/ge_sim_v2_action_native_v1.json'
    return data, actions, config


def test_commands_preserved_no_reference_no_padding(bundle, tmp_path):
    data, actions, config = bundle
    output = tmp_path / 'input.npz'
    result = prepare_official(config, data, output, chunks=2)
    packet = read_packet(output)
    np.testing.assert_array_equal(packet['actions'].reshape(-1, 16), actions[:50])
    assert result['source']['unselected_actions'] == 1
    assert packet['source']['conditioning'] == 'action'
    assert not result['reference_available']
    with pytest.raises(FileExistsError):
        prepare_official(config, data, output)
    with pytest.raises(ValueError, match='no padding'):
        prepare_official(config, data, tmp_path / 'too_long.npz', chunks=3)


def test_rejects_episode_conditioning_and_missing_calibration(bundle, tmp_path):
    data, _, config = bundle
    value = json.loads(config.read_text())
    value['kwargs']['conditioning'] = 'episode'
    wrong = tmp_path / 'wrong.json'
    wrong.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='action-conditioned'):
        prepare_official(wrong, data, tmp_path / 'bad.npz')
    (data / 'intrinsic.npy').unlink()
    with pytest.raises(FileNotFoundError):
        prepare_official(config, data, tmp_path / 'bad.npz')


def test_video_only_output_keeps_native_state_without_claiming_policy_contract():
    from adapters.world_models.ge_sim_v2 import GESimV2Adapter

    model = GESimV2Adapter.__new__(GESimV2Adapter)
    model.conditioning = 'action'
    model.last_video = np.zeros((25, 3, 8, 8, 3), np.uint8)
    model.last_state = np.zeros((25, 16), np.float32)
    model.last_policy_state = np.ones(16, np.float32)
    model.step = lambda action: model.last_video[-1, 0]
    result = model.step_output(np.zeros((25, 16), np.float32))
    assert result.robot_state is None
    assert result.metadata['model_state_returned']
    assert tuple(result.views) == model.view_names
    assert model.last_policy_state.shape == (16,)
    with pytest.raises(ValueError, match='view order'):
        model.step_output(None, output_views=('right_wrist', 'head', 'left_wrist'))


def test_full_episode_keeps_tail_without_fabricated_actions(bundle, tmp_path):
    data, actions, _ = bundle
    config = ROOT / 'configs/replay/ge_sim_v2_action_full_v1.json'
    result = prepare_official(config, data, tmp_path / 'all.npz', chunks='all')
    packet = read_packet(tmp_path / 'all.npz')
    assert [len(chunk) for chunk in packet['actions']] == [25, 25, 1]
    np.testing.assert_array_equal(np.concatenate(packet['actions']), actions)
    assert result['source']['unselected_actions'] == 0
