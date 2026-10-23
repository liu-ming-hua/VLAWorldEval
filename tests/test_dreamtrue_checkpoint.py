"""Weight-free CPU rejection tests for complete checkpoint tensor accounting."""
import pytest

from adapters.world_models.dreamtrue import classify_checkpoint_shapes


def shapes():
    return {"pipe.dit.block.q.lora_A.default.weight": (2, 3),
            "pipe.dit.block.q.lora_B.default.weight": (4, 2),
            "pipe.vace.mask_cam_embedding.1.weight": (4,)}


def test_every_tensor_is_accounted_for_with_pairs_and_full_weights():
    pairs, full, counts = classify_checkpoint_shapes(shapes())
    assert counts == {"dit": 2, "vace": 1}
    assert pairs == {("dit", "block.q"): {"A": (2, 3), "B": (4, 2)}}
    assert full == [("vace", "mask_cam_embedding.1.weight", "pipe.vace.mask_cam_embedding.1.weight")]
    assert 2 * len(pairs) + len(full) == len(shapes())


@pytest.mark.parametrize("fault", ["prefix", "orphan", "rank", "suffix", "dimension"])
def test_header_faults_never_reach_permissive_native_loader(fault):
    value = shapes()
    if fault == "prefix":
        value["unknown.extra"] = (1,)
    elif fault == "orphan":
        value.pop("pipe.dit.block.q.lora_B.default.weight")
    elif fault == "rank":
        value["pipe.dit.block.q.lora_B.default.weight"] = (4, 3)
    elif fault == "suffix":
        value["pipe.dit.block.q.lora_A.other.weight"] = (2, 3)
    elif fault == "dimension":
        value["pipe.dit.block.q.lora_A.default.weight"] = (2, 3, 1, 1)
    with pytest.raises(ValueError):
        classify_checkpoint_shapes(value)
