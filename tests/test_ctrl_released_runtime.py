"""Optional real media/condition checks, no model construction or CUDA."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("decord")
torch = pytest.importorskip("torch")


def test_released_cameras_and_raw_states_match_native_timeline_and_bfloat16_conditions():
    from evaluation.datasets.ctrl_world import load_released_episode
    from adapters.world_models.ctrl_world import CtrlWorldAdapter
    repo = Path(__file__).resolve().parents[1]/"third_party/ctrl-world"
    if not (repo/".git").exists():
        pytest.skip("Optional pinned native media absent")
    stat = json.loads((repo/"dataset_meta_info/droid/stat.json").read_text())
    adapter = CtrlWorldAdapter.__new__(CtrlWorldAdapter)
    adapter.state_p01 = np.asarray(stat["state_01"], np.float32)
    adapter.state_p99 = np.asarray(stat["state_99"], np.float32)
    count = 0
    for episode, length in ((199, 33), (18599, 72)):
        states, views, instruction, assets = load_released_episode(repo, episode)
        assert states.shape == (length, 7)
        assert all(value.shape == (length, 192, 320, 3) for value in views.values())
        assert len(assets) == 4 and instruction.strip()
        official = np.clip(2*(states-np.asarray(stat["state_01"])) /
                           (np.asarray(stat["state_99"])-np.asarray(stat["state_01"])+1e-8)-1, -1, 1)
        actual = np.stack([adapter._normalize_action(row) for row in states])
        torch.testing.assert_close(torch.tensor(actual).to(torch.bfloat16),
                                   torch.tensor(official).to(torch.bfloat16), rtol=0, atol=0)
        count += states.size
    assert count == 735 and not torch.cuda.is_initialized()
