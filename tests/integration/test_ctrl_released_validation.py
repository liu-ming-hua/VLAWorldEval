"""Optional native-domain parity; requires separately acquired assets or runtimes."""

from pathlib import Path

import pytest

from evaluation.datasets.ctrl_world import released_selection


def test_actual_pinned_release_has_all97_distinct_windows_but_only_two_episodes():
    root = Path(__file__).resolve().parents[2] / "third_party/ctrl-world"
    if not (root / ".git").exists():
        pytest.skip("Optional pinned Ctrl checkout absent")
    samples, files = released_selection(root)
    assert len(samples) == 97
    assert {s["episode_id"] for s in samples} == {199, 18599}
    assert len(files) == 1
    assert (
        next(iter(files.values()))["sha256"]
        == "a58c908fc9d787eb52b750e7004967acd1f40c8a72f581dc1852300a9383a100"
    )
