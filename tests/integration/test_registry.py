"""Optional native-domain parity; requires separately acquired assets or runtimes."""

from pathlib import Path

import pytest

from evaluation.manifests import load_pair_manifest
from evaluation.planning.registry import load_runtime_registry

ROOT = Path(__file__).resolve().parents[2]


def test_registered_loader_preserves_official_worldgym_packet():
    root = ROOT / "datasets/worldgym/official/bridge/test"
    if not root.is_dir():
        pytest.skip("Local official sample not installed")
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    registry = load_runtime_registry(ROOT)
    packet = registry.load_episode(
        registry.pair_profile(pair),
        root,
        {"start_frame": 0, "fps": 20, "instruction": "move"},
        {"episode_index": 2, "chunks": 3, "mode": "wm_replay"},
        1,
    )
    assert len(packet.replay_actions) == 3
    assert packet.replay_actions[0].shape == (1, 10)
    assert packet.trial["observation_bundle"].get("robot_state") is None
