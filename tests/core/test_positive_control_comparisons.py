"""Review display aligns control time without padding references or launching models."""

import json
import subprocess

import mediapy as media
import numpy as np
import pytest

from evaluation.reporting.positive_controls import (
    VIEWS,
    aligned_indices,
    render_droid_comparison,
    write_comparison_gallery,
)

pytestmark = pytest.mark.process


def test_sparse_native_video_is_held_only_for_display_time():
    ticks, indices = aligned_indices(3, 3)
    assert ticks.tolist() == list(range(7))
    assert indices.tolist() == [0, 0, 0, 1, 1, 1, 2]


@pytest.mark.parametrize("frames,stride", [(1, 3), (3, 0), (3, 1.5), (True, 1)])
def test_invalid_timebase_rejected(frames, stride):
    with pytest.raises(ValueError):
        aligned_indices(frames, stride)


@pytest.fixture
def inputs(tmp_path):
    refs, generated = {}, {}
    for name in VIEWS:
        refs[name] = tmp_path / f"real_{name}.mp4"
        generated[name] = tmp_path / f"gen_{name}.mp4"
        media.write_video(refs[name], np.full((7, 16, 16, 3), 80, np.uint8), fps=15)
        values = np.repeat(np.arange(3, dtype=np.uint8)[:, None, None, None], 16, axis=1)
        values = np.broadcast_to(values, (3, 16, 16, 3)).copy() * 70
        media.write_video(generated[name], values, fps=7)
    return refs, generated


def test_real_ffmpeg_comparison_is_viewable_and_not_quality_admission(
    tmp_path, inputs, monkeypatch
):
    refs, generated = inputs

    def no_model(*a, **k):
        pytest.fail("A post-hoc comparison must not load a model")

    monkeypatch.setattr("evaluation.execution.client.ModelWorker", no_model)
    report = render_droid_comparison(
        tmp_path,
        references=refs,
        generated=generated,
        output="review",
        label="fixture",
        source_ticks_per_frame=3,
        native_fps=7,
    )
    assert report["comparison_frames"] == 7
    assert report["source_control_seconds"] == 0.4
    assert report["quality_admitted"] is None and report["human_review_performed"] is False
    assert report["model_inference_performed"] is False
    assert len(report["inputs"]) == 6
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(tmp_path / "review/comparison.mp4"),
            ],
            text=True,
        )
    )["streams"][0]
    assert info["nb_frames"] == "7" and info["avg_frame_rate"] == "15/1"
    assert (tmp_path / "review/contact_sheet.jpg").is_file()


def test_short_reference_is_not_padded(tmp_path, inputs):
    refs, generated = inputs
    media.write_video(refs["wrist"], np.zeros((6, 16, 16, 3), np.uint8), fps=15)
    with pytest.raises(ValueError, match="too short"):
        render_droid_comparison(
            tmp_path,
            references=refs,
            generated=generated,
            output="short",
            label="fixture",
            source_ticks_per_frame=3,
            native_fps=7,
        )
    assert not (tmp_path / "short").exists()


def test_duplicate_cameras_are_rejected(tmp_path, inputs):
    refs, generated = inputs
    refs["wrist"] = refs["exterior_1"]
    with pytest.raises(ValueError, match="distinct"):
        render_droid_comparison(
            tmp_path,
            references=refs,
            generated=generated,
            output="duplicate",
            label="fixture",
            source_ticks_per_frame=3,
            native_fps=7,
        )


def test_existing_outputs_never_overwritten(tmp_path, inputs):
    refs, generated = inputs
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        render_droid_comparison(
            tmp_path,
            references=refs,
            generated=generated,
            output=output,
            label="fixture",
            source_ticks_per_frame=3,
            native_fps=7,
        )
    assert marker.read_text() == "keep"


def test_gallery_escapes_labels_and_cannot_overwrite(tmp_path):
    write_comparison_gallery(tmp_path, [{"label": '<unsafe "fixture">'}])
    text = (tmp_path / "index.html").read_text()
    assert "<unsafe" not in text and "&lt;unsafe" in text
    assert 'preload="none"' in text
    with pytest.raises(FileExistsError):
        write_comparison_gallery(tmp_path, [])
