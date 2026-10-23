import pytest

from evaluation.datasets.success_samples import candidate_suite


def template():
    return {
        "suite_id": "test",
        "dataset": {},
        "experiments": [{"state_provider": "integrated_action_target"}],
    }


def test_fixed_window_uses_complete_chunks_without_mutating_template():
    original = template()
    result = candidate_suite(original, episode=350, length=150)
    assert result["experiments"][0]["chunks"] == 8
    assert result["metadata"]["source_end_frame"] == 143
    assert result["metadata"]["discarded_tail_frames"] == 6
    assert result["metadata"]["task_success"] is None
    assert original["dataset"] == {}
    assert result["experiments"][1]["state_provider"] == "integrated_action_target"


@pytest.mark.parametrize(
    "overrides",
    [{"length": 16}, {"start": 0}, {"source_horizon": 17}, {"episode": -1}, {"length": True}],
)
def test_invalid_window_is_rejected(overrides):
    values = dict(episode=0, length=150)
    values.update(overrides)
    with pytest.raises(ValueError):
        candidate_suite(template(), **values)
