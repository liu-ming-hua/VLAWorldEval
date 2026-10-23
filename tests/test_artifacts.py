import numpy as np

from evaluation.reporting.artifacts import pack_action_chunks


def test_pack_action_chunks_preserves_lengths_and_masks():
    actions, lengths, mask = pack_action_chunks(
        [np.ones((2, 3), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)]
    )

    assert actions.shape == (2, 2, 3)
    assert lengths.tolist() == [2, 1]
    assert mask.tolist() == [[True, True], [True, False]]
    assert np.isnan(actions[1, 0, 2])
    assert np.isnan(actions[1, 1]).all()


def test_pack_action_chunks_accepts_single_action_vectors():
    actions, lengths, mask = pack_action_chunks([np.array([1.0, 2.0])])

    assert actions.shape == (1, 1, 2)
    assert lengths.tolist() == [1]
    assert mask.tolist() == [[True]]
