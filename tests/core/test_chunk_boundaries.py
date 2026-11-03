import pytest

from evaluation.reporting.chunk_boundaries import summarize_transitions, replay_chunk_lengths


def test_boundary_diagnostics_include_tail_and_exclude_initial_boundary():
    result = summarize_transitions([1, 1, 10, 1, 0], [2, 2, 1])
    assert [b['first_frame_next_chunk'] for b in result['boundaries']] == [3, 5]
    assert result['boundaries'][0]['ratio_to_nonboundary_median'] == 10
    assert result['longest_near_static_transition_run'] == 1
    assert result['near_static_transition_fraction'] == 0.2
    with pytest.raises(ValueError, match='Transition count'):
        summarize_transitions([1, 2], [2, 1])


def test_inclusive_cropped_tail_uses_retained_frame_count():
    lengths = replay_chunk_lengths({'world_model_calls': 18, 'advance_frames_per_chunk': 4,
                                    'frames_per_view': 72})
    assert lengths == [4] * 17 + [3]
    with pytest.raises(ValueError):
        replay_chunk_lengths({'world_model_calls': 2, 'advance_frames_per_chunk': 4,
                              'frames_per_view': 10})
