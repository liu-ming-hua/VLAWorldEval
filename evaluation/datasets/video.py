"""Accurate recorded RGB boundaries shared by diagnostic preparation tools."""
import json
from contextlib import ExitStack
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

INLINE_FILTER_BYTES = 32*1024


def selection_filter(indices):
    """Relative-frame selector: arithmetic progressions have constant text size."""
    values = np.asarray(list(indices))
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in "iu"
            or values[0] < 0 or np.any(values[1:] <= values[:-1])):
        raise ValueError("Frame indices must be nonempty, strictly increasing nonnegative integers")
    relative = values-values[0]
    if len(relative) == 1:
        return values, "eq(n\\,0)"
    steps = np.diff(relative)
    if np.all(steps == steps[0]):
        expression = f"between(n\\,0\\,{int(relative[-1])})"
        if steps[0] != 1:
            expression += f"*not(mod(n\\,{int(steps[0])}))"
        return values, expression
    return values, "+".join(f"eq(n\\,{int(i)})" for i in relative)


def decode_indexed_video(path, indices, transform=None, *, timeout=180):
    """Exact old seek/select semantics with bounded argv, no reduced frame count.

    Contiguous/strided windows need a tiny selector. Irregular selectors that
    exceed the inline limit use an owned temporary filter script, not argv.
    No caches or synthesized frames substitute for missing recorded pixels.
    """
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Decoder timeout must be finite and positive")
    indices, expression = selection_filter(indices)
    stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
        timeout=min(timeout, 60)))["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Indexed video requires a positive recorded frame rate")
    command = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(indices[0]/(numerator/denominator)),
               "-i", str(path), "-map", "0:v:0", "-filter_threads", "1"]
    with ExitStack() as resources:
        if len(expression.encode()) <= INLINE_FILTER_BYTES:
            command += ["-vf", "select="+expression]
        else:
            from vlaworldeval.paths import project_root
            cache = project_root()/".cache/vlaworldeval/video_filters"
            if not cache.resolve().is_relative_to(project_root().resolve()):
                raise ValueError("Temporary video filters must stay inside the project")
            cache.mkdir(parents=True, exist_ok=True)
            directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="selection_", dir=cache))
            script = Path(directory)/"filter.txt"
            script.write_text("select="+expression)
            command += ["-filter_script:v", str(script)]
        command += ["-vsync", "0", "-frames:v", str(len(indices)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        pixels = subprocess.check_output(command, timeout=timeout)
    size = int(stream["height"])*int(stream["width"])*3
    if len(pixels) != len(indices)*size:
        raise ValueError("Decoder did not return every requested real frame; no padding/substitution")
    frames = np.frombuffer(pixels, np.uint8).reshape(len(indices), stream["height"], stream["width"], 3)
    return frames.copy() if transform is None else np.stack([transform(frame.copy()) for frame in frames])


def decode_boundaries(source, chunks, stride):
    """Decode each original camera independently; no GT proprioception."""
    if type(chunks) is not int or chunks < 1 or type(stride) is not int or stride < 1:
        raise ValueError("Boundary counts and strides must be positive integers")
    frames = decode_source_views(source, chunks * stride + 1)
    return {name: value[:chunks * stride:stride].copy() for name, value in frames.items()}


def decode_source_views(source, count):
    """Decode an exact inclusive source prefix, without repeating the endpoint."""
    if type(count) is not int or count < 1:
        raise ValueError("Source frame count must be a positive integer")
    result = {}
    for name, info in source["videos"].items():
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", info["path"]], text=True))["streams"][0]
        data = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "2", "-i", info["path"],
            "-ss", str(info["initial_frame_time"]), "-frames:v", str(count),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        frames = np.frombuffer(data, np.uint8).reshape(-1, probe["height"], probe["width"], 3)
        if len(frames) != count:
            raise ValueError("Real camera does not cover the complete diagnostic window")
        result[name] = frames.copy()
    return result
