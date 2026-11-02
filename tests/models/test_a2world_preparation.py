"""Official example conversion must preserve actions and reject invented futures."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from evaluation.datasets.a2world import prepare_official, VIEWS
from evaluation.execution.transport import read_packet

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def example(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    actions = np.arange(280, dtype=np.float32).reshape(20, 14) / 100
    np.savez(dataset / 'actions.npz', actions=actions)
    for index, view in enumerate(VIEWS):
        Image.fromarray(np.full((30, 40, 3), index * 50, np.uint8)).save(dataset / f'{view}.png')
    config = ROOT / 'configs/replay/a2world_pretrained_native_v1.json'
    return dataset, actions, config


def test_official_packet_preserves_commands_camera_order_and_no_future(example, tmp_path):
    dataset, actions, config = example
    output = tmp_path / 'input.npz'
    report = prepare_official(config, dataset, output)
    packet = read_packet(output)
    np.testing.assert_array_equal(packet['actions'][0], actions)
    assert packet['reset_args']['value']['view_ids'] == [2, 0, 3]
    assert packet['initial_views']['wrist'].shape == (256, 256, 3)
    assert list(packet['initial_views']) == list(VIEWS)
    assert not report['reference_available'] and not report['model_inference_performed']
    assert len(packet['source']['files_sha256']) == 4
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_official(config, dataset, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize('bad', [np.zeros((19, 14)), np.zeros((21, 14)),
                                  np.zeros((20, 7)), np.full((20, 14), np.nan)])
def test_rejects_padding_truncation_mapping_and_nonfinite(example, tmp_path, bad):
    dataset, _, config = example
    np.savez(dataset / 'actions.npz', actions=bad)
    output = tmp_path / 'input.npz'
    with pytest.raises(ValueError, match='exactly 20x14 finite'):
        prepare_official(config, dataset, output)
    assert not output.exists()


def test_rejects_libero_and_camera_permutation(example, tmp_path):
    dataset, _, original = example
    config = json.loads(original.read_text())
    config['output_views'].reverse()
    path = tmp_path / 'wrong.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='pretrained'):
        prepare_official(path, dataset, tmp_path / 'input.npz')


def test_official_opencv_area_resize_and_ambiguous_archive(example, tmp_path):
    import cv2

    dataset, actions, config = example
    image = np.random.default_rng(2).integers(0, 256, (271, 301, 3), dtype=np.uint8)
    Image.fromarray(image).save(dataset / 'wrist.png')
    output = tmp_path / 'input.npz'
    prepare_official(config, dataset, output)
    np.testing.assert_array_equal(read_packet(output)['initial_views']['wrist'],
                                  cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA))
    np.savez(dataset / 'actions.npz', actions=actions, obs_state=np.zeros((21, 7)))
    with pytest.raises(ValueError, match='actions-only'):
        prepare_official(config, dataset, tmp_path / 'ambiguous.npz')
