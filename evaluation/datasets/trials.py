"""Trial discovery and task metadata utilities."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def load_tasks(root: str | Path):
    """Yield image stems for the legacy task-list interface."""

    for file in os.listdir(root):
        if file.endswith(".png"):
            yield str(Path(root) / Path(file).stem)


def _titleize(name: str) -> str:
    name = name.replace("--", " ")
    return " ".join(word.capitalize() for word in re.split(r"[_\\-]+", name))


def discover_trials(root_dir: str | Path) -> list[dict]:
    """Discover local ``trial.png`` + ``trial.json`` pairs."""

    root_path = Path(root_dir).resolve()
    trials = []
    for image_path in root_path.rglob("*.png"):
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists():
            print(f"[WARN] No JSON for {image_path}")
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Bad JSON {metadata_path}: {exc}")
            continue
        instruction = metadata.get("instruction")
        if not instruction:
            print(f"[WARN] No instruction in {metadata_path}")
            continue
        task_dir = image_path.parent
        trials.append(
            {
                "trial_png": str(image_path),
                "instruction": instruction,
                "partial_criteria": metadata.get("partial_credit_criteria"),
                "task_key": str(task_dir.relative_to(root_path)),
                "task_display": _titleize(task_dir.name),
            }
        )
    return trials


__all__ = ["discover_trials", "load_tasks"]
