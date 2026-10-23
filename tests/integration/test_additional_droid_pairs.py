"""Optional native-domain parity; requires separately acquired assets or runtimes."""

from pathlib import Path

import pytest


def test_actual_additional_dataset_packets_when_available():
    root = Path("datasets/cosmos3_droid/success")
    if not root.is_dir():
        pytest.skip("Official DROID assets absent")
    pytest.importorskip("pyarrow")
    from evaluation.datasets.droid import load_groot_n17_droid_episode, load_openpi_ctrl_episode

    ctrl = load_openpi_ctrl_episode(root, 0, 30, 2)
    assert ctrl.trial["episode_bundle"]["first_action"].shape == (7,)
    assert ctrl.replay_actions[0].shape == (5, 7)
    n17 = load_groot_n17_droid_episode(root, 0, 30, 2)
    assert n17.trial["observation_bundle"]["views"]["droid.wrist"].shape[0] == 16


def test_n17_ctrl_packet_on_official_dataset():
    if not Path("datasets/cosmos3_droid/success").is_dir():
        pytest.skip("Official DROID assets absent")
    pytest.importorskip("pyarrow")
    from evaluation.datasets.droid import load_groot_n17_ctrl_episode

    packet = load_groot_n17_ctrl_episode(Path("datasets/cosmos3_droid/success"), 0, 30, 2)
    assert packet.trial["observation_bundle"]["views"]["droid.wrist"].shape[0] == 6
    assert packet.trial["episode_bundle"]["views"][-1].ndim == 3
