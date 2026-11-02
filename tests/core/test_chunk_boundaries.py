import pytest

from evaluation.reporting.chunk_boundaries import summarize_transitions


def test_boundary_diagnostics_include_tail_and_exclude_initial_boundary():
    result = summarize_transitions([1, 1, 10, 1, 0], [2, 2, 1])
    assert [b['first_frame_next_chunk'] for b in result['boundaries']] == [3, 5]
    assert result['boundaries'][0]['ratio_to_nonboundary_median'] == 10
    assert result['longest_near_static_transition_run'] == 1
    assert result['near_static_transition_fraction'] == 0.2
    with pytest.raises(ValueError, match='Transition count'):
        summarize_transitions([1, 2], [2, 1])
