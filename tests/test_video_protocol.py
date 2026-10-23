from dataclasses import replace

import numpy as np
import pytest

from evaluation.contracts import WorldModelOutput
from evaluation.manifests import VideoSpec, load_pair_manifest


@pytest.mark.parametrize("fps,frames", [(0, 1), (float("inf"), 1), (True, 1), (20, True), (20, 0), (20, 1.5)])
def test_bad_video_spec(fps, frames):
    with pytest.raises(ValueError):
        VideoSpec(fps, frames)


def test_video_rate_and_frames_are_independent_of_action():
    pair = load_pair_manifest("configs/pairs/openvla__worldgym.json")
    wm = replace(pair.world_model, video=VideoSpec(29.97, 3))
    assert wm.action.horizon == 1 and wm.action.rate_hz == 20
    output = WorldModelOutput(views={"bridge.primary": np.zeros((3, 8, 8, 3), dtype=np.uint8)})
    wm.video.validate_output(output, wm.output_views)
    with pytest.raises(ValueError, match="frames; expected"):
        VideoSpec(29.97, 2).validate_output(output, wm.output_views)
    with pytest.raises(ValueError, match="views"):
        wm.video.validate_output(output, ["missing"])


def test_all_executable_pairs_explicit_video_contract():
    from pathlib import Path
    for path in Path("configs/pairs").glob("*.json"):
        pair = load_pair_manifest(path)
        assert pair.world_model.video is not None
        assert pair.to_dict()["world_model"]["video"]["fps"] > 0


def test_nonfinite_images_are_not_silently_cast_to_black():
    frames = np.zeros((1, 8, 8, 3), dtype=np.float32)
    frames[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        WorldModelOutput(views={"front": frames})


def test_composite_and_decoded_views_have_separate_contracts():
    from adapters.observations.base import build_view_bridge
    pair = load_pair_manifest("configs/pairs/groot_n15__cosmos3_droid.json")
    assert pair.world_model.output_views == ("droid.cosmos_composite",)
    assert len(pair.video_output_views) == 4
    output = WorldModelOutput(views={"droid.cosmos_composite": np.zeros((16, 48, 32, 3), dtype=np.uint8)})
    pair.world_model.video.validate_output(output, pair.world_model.output_views)
    decoded = build_view_bridge(pair.view_bridge).process_world_output(output)
    pair.world_model.video.validate_output(decoded, pair.video_output_views)


@pytest.mark.parametrize("value,dtype", [(-1, np.float32), (256, np.uint16), (-1, np.int16), (True, np.bool_)])
def test_invalid_rgb_does_not_wrap_or_hide_normalization_errors(value, dtype):
    with pytest.raises((ValueError, TypeError)):
        WorldModelOutput(views={"front": np.full((1, 8, 8, 3), value, dtype=dtype)})
