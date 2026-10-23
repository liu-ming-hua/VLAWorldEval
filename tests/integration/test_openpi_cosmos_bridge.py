"""Optional native-domain parity; requires separately acquired assets or runtimes."""

from pathlib import Path

import pytest


def test_official_episode_fk_admission_when_assets_exist():
    root = Path("datasets/cosmos3_droid/success")
    if not root.is_dir():
        pytest.skip("Official DROID episode assets not installed")
    pytest.importorskip("pyarrow")
    from evaluation.datasets.droid import load_openpi_droid_episode

    packet = load_openpi_droid_episode(root, 0, 30, 2)
    evidence = packet.provenance["kinematics_validation"]
    assert evidence["rows_checked"] > 32
    assert evidence["max_position_error_m"] < 0.002
    state = packet.trial["initial_state"]
    assert state.source == "dataset_ground_truth"
    assert state.values["observation/joint_position"].shape == (7,)
