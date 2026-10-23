"""Read-only inventory of suite attempts; no model imports or GPU use."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.execution.transport import read_packet


def collect_results(root: Path) -> dict:
    attempts, errors = [], []
    for path in sorted(root.rglob("summary.json")):
        if not path.parent.name.startswith("attempt_"):
            continue
        try:
            value = json.loads(path.read_text())
            loads = len(value.get("worker_loads", []))
            if "worker_loads" not in value:
                loads = sum(bool(read_packet(ready).get("ok"))
                            for ready in (path.parent / "workers").glob("*/ready.npz"))
            attempts.append({"directory": str(path.parent), "status": value["status"],
                             "jobs": value["jobs"],
                             "error_info": value.get("error_info"),
                             "worker_cleanup": value.get("worker_cleanup", []),
                             "model_loads": loads})
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"attempts": attempts, "errors": errors,
            "status_counts": {status: sum(item["status"] == status for item in attempts)
                              for status in ("running", "completed", "failed")}}
