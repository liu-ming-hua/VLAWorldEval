"""Optional native-domain parity; requires separately acquired assets or runtimes."""

from pathlib import Path

import pytest

from evaluation.datasets.worldgym import load_worldgym_episode


def test_official_worldgym_data_rejects_incomplete_replay_but_allows_imagination():
    root = Path("datasets/worldgym/official/bridge/test")
    if not root.is_dir():
        pytest.skip("Official fixture not present")
    sample = load_worldgym_episode(root, 2, 0, 3, 4, 20, instruction="move", replay=True)
    assert len(sample.replay_actions) == 3
    assert sample.replay_actions[0].shape == (4, 10)
    with pytest.raises(ValueError, match="Incomplete"):
        load_worldgym_episode(root, 2, 0, 20, 4, 20, instruction="move", replay=True)
    imagined = load_worldgym_episode(root, 2, 0, 20, 4, 20, instruction="move", replay=False)
    assert imagined.provenance["imagined_beyond_sample"]
    assert imagined.trial["observation_bundle"].get("robot_state") is None
