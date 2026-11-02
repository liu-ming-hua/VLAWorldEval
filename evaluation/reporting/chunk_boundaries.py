"""Decoded-video transition diagnostics; not a task-success or physics score."""
from pathlib import Path
import json

import cv2
import numpy as np


def summarize_transitions(motion, chunk_lengths):
    values = np.asarray(motion, dtype=np.float64)
    if (not chunk_lengths or any(type(n) is not int or n < 1 for n in chunk_lengths)
            or len(values) != sum(chunk_lengths) or not np.isfinite(values).all()):
        raise ValueError('Transition count must equal the number of executed actions')
    # motion[i] compares decoded frames i and i+1; initial RGB occupies frame 0.
    indices = np.cumsum(chunk_lengths)[:-1]
    ordinary = np.delete(values, indices)
    baseline = float(np.median(ordinary)) if len(ordinary) else 0.0
    streak = maximum = 0
    for value in values:
        streak = streak + 1 if value <= 0.1 else 0
        maximum = max(maximum, streak)
    return {'mean_abs_change_0_255': float(values.mean()),
            'median_nonboundary_change': baseline,
            'near_static_threshold_0_255': 0.1,
            'near_static_transition_fraction': float(np.mean(values <= 0.1)),
            'longest_near_static_transition_run': maximum,
            'boundaries': [{'first_frame_next_chunk': int(i + 1),
                            'mean_abs_change_0_255': float(values[i]),
                            'ratio_to_nonboundary_median': float(values[i] / baseline) if baseline else None}
                           for i in indices]}


def inspect_replay(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    lengths = summary['action_chunk_lengths']
    report = {'scope': 'Decoded H264 transition diagnostics; motion/jumps do not prove action or task correctness',
              'views': {}, 'action_chunk_lengths': lengths}
    for filename in summary['files']:
        if not filename.endswith('.mp4') or '_prediction_vs_reference' in filename:
            continue
        capture = cv2.VideoCapture(str(directory / filename))
        transitions = []
        previous = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if previous is not None:
                    transitions.append(float(np.abs(frame.astype(np.int16) - previous).mean()))
                previous = frame.astype(np.int16)
        finally:
            capture.release()
        report['views'][filename] = summarize_transitions(transitions, lengths)
    if len(report['views']) != len(summary['video_diagnostics']):
        raise ValueError('Missing expected replay videos')
    return report
