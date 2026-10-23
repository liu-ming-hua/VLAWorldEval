#!/usr/bin/env python3
"""Launch all VLA official-data smoke tests across eight GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts/vla/smoke.py"
PYTHONS = {
    "openvla": "openvla",
    "spatialvla": "spatialvla",
    "octo": "octo",
    "rt1_x": "rt1",
    "groot_n15": "groot_n15",
    "groot_n17": "groot_n17",
    "pi0_droid": "openpi",
    "pi05_droid": "openpi",
    "smolvla": "smolvla",
}
INITIAL = [
    (0, "openvla"),
    (1, "spatialvla"),
    (2, "octo"),
    (3, "rt1_x"),
    (4, "groot_n15"),
    (5, "groot_n17"),
    (6, "pi0_droid"),
    (7, "smolvla"),
]


def process_memory_mib(pid: int) -> int:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    total = 0
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            total += int(fields[1])
    return total


def launch(gpu: int, model: str, output_dir: Path):
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLA_PHYSICAL_GPU=str(gpu),
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TFHUB_CACHE_DIR=str(ROOT / ".cache/tfhub"),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.23",
        PYTHONPATH=str(ROOT),
        TOKENIZERS_PARALLELISM="false",
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    python = ROOT / "environment/vla" / PYTHONS[model] / "bin/python"
    result = output_dir / f"{model}.json"
    log_path = output_dir / f"{model}.log"
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(python), str(WORKER), "--model", model, "--output", str(result)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return {
        "gpu": gpu,
        "model": model,
        "process": process,
        "log": log,
        "peak_memory_mib": 0,
        "result": result,
    }


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results/vla_official_smoke" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = ROOT / "results/vla_official_smoke/latest.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output_dir.relative_to(ROOT)) + "\n", encoding="utf-8")

    active = {gpu: launch(gpu, model, output_dir) for gpu, model in INITIAL}
    pending = deque(["pi05_droid"])
    finished = []
    print(f"results={output_dir}", flush=True)
    while active:
        for gpu, job in list(active.items()):
            job["peak_memory_mib"] = max(
                job["peak_memory_mib"], process_memory_mib(job["process"].pid)
            )
            code = job["process"].poll()
            if code is None:
                continue
            job["log"].close()
            job["exit_code"] = code
            finished.append(job)
            del active[gpu]
            print(
                f"finished gpu={gpu} model={job['model']} code={code} "
                f"peak={job['peak_memory_mib']}MiB",
                flush=True,
            )
            if pending:
                model = pending.popleft()
                active[gpu] = launch(gpu, model, output_dir)
                print(f"launched gpu={gpu} model={model}", flush=True)
        time.sleep(1)

    results = []
    for job in finished:
        if job["result"].is_file():
            result = json.loads(job["result"].read_text(encoding="utf-8"))
        else:
            result = {
                "model": job["model"],
                "status": "failed",
                "error": "worker exited without a result file",
            }
        result.update(
            assigned_gpu=job["gpu"],
            peak_process_gpu_memory_mib=job["peak_memory_mib"],
            exit_code=job["exit_code"],
            log=str((output_dir / f"{job['model']}.log").relative_to(ROOT)),
        )
        results.append(result)
    order = {name: index for index, name in enumerate(PYTHONS)}
    results.sort(key=lambda item: order[item["model"]])
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "memory_ceiling_gib": 0.23 * 80,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
