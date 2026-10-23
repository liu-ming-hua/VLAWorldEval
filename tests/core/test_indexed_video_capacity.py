"""Codec infrastructure tests are synthetic and never count as model results."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from evaluation.datasets.video import decode_indexed_video, selection_filter

pytestmark = pytest.mark.process


def movie(path, count, codec="ffv1"):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Native video tools unavailable")
    raw = np.empty((count, 16, 16, 3), np.uint8)
    raw[..., 0] = (np.arange(count) % 251)[:, None, None]
    raw[..., 1] = (np.arange(count) // 251 % 251)[:, None, None]
    raw[..., 2] = 17
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "16x16",
            "-framerate",
            "15",
            "-i",
            "pipe:0",
            "-c:v",
            codec,
            str(path),
        ],
        input=raw.tobytes(),
        check=True,
        timeout=60,
    )
    return raw


@pytest.mark.parametrize("indices", [[], [2, 1], [1, 1], [-1], [True], [0.0, 1.0]])
def test_selector_rejects_ambiguous_timelines(indices):
    with pytest.raises(ValueError):
        selection_filter(indices)


@pytest.mark.parametrize("indices", [range(12737), range(1, 2_000_000, 3)])
def test_native_dense_and_strided_selectors_do_not_grow_with_length(indices):
    values, expression = selection_filter(indices)
    assert len(values) == len(indices)
    assert len(expression) < 100


def test_actual_longest_planned_cosmos_length_decodes_every_synthetic_frame(tmp_path):
    from evaluation.datasets.native_reference import indexed_video

    raw = movie(tmp_path / "fixture.mkv", 12737)
    actual = indexed_video(tmp_path / "fixture.mkv", range(12737), timeout=120)
    np.testing.assert_array_equal(actual, raw)


@pytest.mark.parametrize(
    "indices", [list(range(3, 48)), list(range(4, 59, 3)), [2, 3, 7, 9, 20, 40]]
)
def test_seek_stride_and_irregular_pixels_match_existing_decoder(tmp_path, indices):
    from evaluation.datasets.native_reference import indexed_video

    raw = movie(tmp_path / "fixture.mkv", 60)
    actual = decode_indexed_video(tmp_path / "fixture.mkv", indices)
    np.testing.assert_array_equal(actual, raw[indices])
    np.testing.assert_array_equal(actual, indexed_video(tmp_path / "fixture.mkv", indices))


def test_large_filter_uses_owned_script_and_cleans_up_on_success_or_failure(tmp_path, monkeypatch):
    from evaluation.datasets import video as module
    from vlaworldeval import paths

    monkeypatch.setattr(module, "INLINE_FILTER_BYTES", 1)
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    path = tmp_path / "fixture.mkv"
    raw = movie(path, 60)
    indices = [1, 3, 4, 8, 27]
    np.testing.assert_array_equal(decode_indexed_video(path, indices), raw[indices])
    cache = tmp_path / ".cache/vlaworldeval/video_filters"
    assert not list(cache.iterdir())
    with pytest.raises(ValueError, match="every requested real frame"):
        decode_indexed_video(path, [1, 3, 90])
    assert not list(cache.iterdir())


def test_transforms_receive_independent_writeable_frames(tmp_path):
    path = tmp_path / "fixture.mkv"
    raw = movie(path, 3)

    def change(value):
        value[0, 0] = 255
        return value[:8, :8]

    actual = decode_indexed_video(path, range(3), change)
    expected = raw[:, :8, :8].copy()
    expected[:, 0, 0] = 255
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "indices", [list(range(3, 48)), list(range(4, 59, 3)), [2, 3, 7, 9, 20, 40]]
)
def test_h264_pixels_match_independent_legacy_seek_select_command(tmp_path, indices):
    """Keep the old short-window algorithm independent of the new wrapper."""
    from evaluation.datasets.native_reference import indexed_video

    path = tmp_path / "fixture.mp4"
    movie(path, 60, "libx264")
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    expression = "+".join(f"eq(n\\,{i - indices[0]})" for i in indices)
    pixels = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(indices[0] / (numerator / denominator)),
            "-i",
            str(path),
            "-vf",
            "select=" + expression,
            "-vsync",
            "0",
            "-frames:v",
            str(len(indices)),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        timeout=60,
    )
    old = np.frombuffer(pixels, np.uint8).reshape(-1, stream["height"], stream["width"], 3)
    np.testing.assert_array_equal(indexed_video(path, indices), old)
