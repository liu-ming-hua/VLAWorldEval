"""Optional VLM scoring and result aggregation."""

from __future__ import annotations

import base64
import re

import cv2
import numpy as np
from openai import OpenAI


def encode_video(video: np.ndarray, stride: int = 20) -> list[str]:
    """Sample RGB frames and encode them as JPEG data strings."""

    frames = []
    for index, frame in enumerate(video):
        if index % stride:
            continue
        if (frame == 0).all():
            break
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", frame_bgr)
        if ok:
            frames.append(base64.b64encode(buffer).decode())
    return frames


def score_video_with_vlm(video: np.ndarray, trial: dict, n: int = 5) -> float:
    """Score a generated rollout with an OpenAI-compatible vision model."""

    frames = encode_video(video)
    instruction = trial["instruction"]
    has_partial = bool(trial.get("partial_criteria"))
    partial_text = trial.get("partial_criteria", "")
    rubric = (
        f'0 = Failure: little or no progress toward: "{instruction}"\n'
        f'0.5 = Partial: "{partial_text}" achieved but the instruction is incomplete\n'
        f"1 = Success: instruction fully completed"
        if has_partial
        else f'0 = Failure: instruction "{instruction}" not completed.\n1 = Success: instruction completed.'
    )
    allowed = r"(0\.5|0(?:\.0)?|1(?:\.0)?)" if has_partial else r"(0(?:\.0)?|1(?:\.0)?)"
    prompt = f"""Here is a sequence of frames from a robot policy rolled out in an action-conditioned video world model.
Determine how successfully the policy completes the task.

Instruction: {instruction}
{rubric}

Give brief reasoning, then output exactly one final line: Final Score: X"""
    response = OpenAI().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    prompt,
                    *[{"image": frame, "resize": 256} for frame in frames],
                ],
            }
        ],
        n=n,
    )
    counts = {"0": 0, "0.5": 0, "1": 0}
    pattern = re.compile(rf"Final\s*Score:\s*{allowed}", re.IGNORECASE)
    for choice in response.choices:
        match = pattern.search((choice.message.content or "").strip())
        if not match:
            continue
        value = match.group(1)
        counts["0" if value == "0.0" else "1" if value == "1.0" else value] += 1
    ordered = ["1", "0.5", "0"] if has_partial else ["1", "0"]
    selected = max(ordered, key=lambda key: (counts[key], -ordered.index(key)))
    return {"0": 0.0, "0.5": 0.5, "1": 1.0}[selected]


# Backward-compatible name used by the original CLI entry points.
predict = score_video_with_vlm


def aggregate_model_results(results: list[dict]) -> dict:
    unscored_trials = sum(result.get("score") is None for result in results)
    tasks = {}
    for result in results:
        if result.get("score") is None:
            continue
        key = result["task_key"]
        tasks.setdefault(
            key,
            {"Task": result["task_display"], "# Trials": 0, "# Successes": 0.0},
        )
        tasks[key]["# Trials"] += 1
        tasks[key]["# Successes"] += float(result["score"])
    task_list = sorted(tasks.values(), key=lambda item: item["Task"])
    per_trial = []
    for task in task_list:
        if task["# Trials"]:
            per_trial.extend([task["# Successes"] / task["# Trials"]] * task["# Trials"])
    scores = np.asarray(per_trial, dtype=np.float32)
    return {
        "tasks": task_list,
        "mean_success_rate": 100.0 * float(scores.mean()) if len(scores) else 0.0,
        "ste": 100.0 * float(scores.std() / np.sqrt(len(scores))) if len(scores) else 0.0,
        "unscored_trials": unscored_trials,
    }


def print_results_table(aggregate: dict) -> None:
    tasks = aggregate["tasks"]
    if not tasks:
        print(
            f"No scored trials. Completed {aggregate.get('unscored_trials', 0)} "
            "rollout(s); pass --score-vlm to enable scoring."
        )
        return
    header = ["Task", "# Trials", "# Successes"]
    widths = [len(value) for value in header]
    for task in tasks:
        widths[0] = max(widths[0], len(task["Task"]))
        widths[1] = max(widths[1], len(str(task["# Trials"])))
        widths[2] = max(widths[2], len(str(task["# Successes"])))
    def format_row(row):
        return " | ".join(
            str(value).ljust(widths[i]) for i, value in enumerate(row)
        )
    print(format_row(header))
    print("-" * (sum(widths) + 3 * (len(header) - 1)))
    for task in tasks:
        print(format_row([task["Task"], task["# Trials"], task["# Successes"]]))
    print(f"Mean Success Rate: {aggregate['mean_success_rate']:.1f}±{aggregate['ste']:.1f}%")


__all__ = [
    "aggregate_model_results",
    "encode_video",
    "predict",
    "print_results_table",
    "score_video_with_vlm",
]
