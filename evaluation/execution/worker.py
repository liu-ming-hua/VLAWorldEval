"""Persistent model worker, optionally under torchrun. Parent owns the loop."""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np

from evaluation.execution.transport import read_packet, write_packet
from evaluation.execution.reliability import remote_error


def serve(directory: Path) -> None:
    config = json.loads((directory / "worker.json").read_text())
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    torch = None
    started = time.perf_counter()
    try:
        # CUDA initialization failures must have the same startup diagnostics.
        if config.get("gpus"):
            import torch
            torch.cuda.set_device(local_rank)
            total = torch.cuda.get_device_properties(local_rank).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1, config["memory_gib"] * 2**30 / total), local_rank)
        adapter = getattr(importlib.import_module(config["module"]), config["adapter"])(**config["kwargs"])
        if distributed and not torch.distributed.is_initialized():
            raise RuntimeError("Distributed adapter did not initialize its process group")
        if rank == 0:
            write_packet(directory / "ready.npz", {"ok": True, "adapter": type(adapter).__name__,
                         "supports_episode_reuse": bool(getattr(adapter, "supports_episode_reuse", False)),
                         "load_seconds": time.perf_counter() - started, "pid": os.getpid(),
                         "python": sys.version, "executable": sys.executable,
                         "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
                                      if dist.metadata["Name"]}})
    except BaseException as exc:
        if rank == 0:
            write_packet(directory / "ready.npz", {"ok": False, "error": traceback.format_exc(),
                                                   "error_info": remote_error(exc)})
        raise
    while True:
        line = sys.stdin.readline().strip() if rank == 0 else None
        if distributed:
            message = [line]
            torch.distributed.broadcast_object_list(message, src=0)
            line = message[0]
        if not line or line == "close":
            if rank == 0:
                (directory / "shutdown.json").write_text(json.dumps({"status": "received", "reason": "close" if line else "stdin_eof"}))
            break
        request = directory / line
        response = directory / line.replace("request_", "response_")
        try:
            value = read_packet(request)
            operation, args = value["operation"], value["args"]
            if operation == "episode_context":
                if not getattr(adapter, "supports_episode_reuse", False):
                    raise ValueError("Adapter has no verified cross-episode reset contract")
                adapter.set_episode_context(args["runtime_dir"])
                result = None
            elif operation == "seed":
                seed = int(args["seed"])
                random.seed(seed)
                np.random.seed(seed)
                if torch is not None:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                if hasattr(adapter, "seed"):
                    adapter.seed = seed
                result = None
            elif operation == "reset":
                adapter.reset(args["value"])
                result = None
            elif operation == "reset_official_replay":
                # Explicit protocol operation, never arbitrary remote getattr.
                adapter.reset_official_replay(args["image"], args["first_action"])
                result = None
            elif operation == "reanchor_image":
                result = adapter.reanchor_image(args["image"])
            elif operation == "verify_ctrl_reference":
                from evaluation.validation.ctrl_reference import verify_ctrl_reference
                result = verify_ctrl_reference(adapter, args["action"], args["first_action"])
            elif operation == "predict":
                result = adapter.predict(args["image"], args["instruction"], args["state"])
            elif operation == "step":
                output = adapter.step_output(args["action"], output_views=args["output_views"],
                                             state_schema=args["state_schema"])
                result = {"views": output.views, "metadata": output.metadata,
                          "robot_state": None if output.robot_state is None else {
                              "values": output.robot_state.values, "schema": output.robot_state.schema,
                              "source": output.robot_state.source, "uncertainty": output.robot_state.uncertainty,
                          }}
            elif operation == "native_batch_step":
                if not getattr(adapter, "supports_native_batch", False):
                    raise ValueError("Adapter has no native batch contract")
                result = adapter.native_batch_step(args["items"])
            else:
                raise ValueError(f"Unknown worker operation {operation}")
            stats = None
            if torch is not None:
                stats = {"rank": rank, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                try:
                    output = subprocess.check_output([
                        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits",
                    ], text=True, timeout=5)
                    stats["process_memory_mib"] = sum(int(row.split(",")[1].strip()) for row in output.splitlines()
                                                      if row.split(",")[0].strip() == str(os.getpid()))
                except (subprocess.SubprocessError, ValueError):
                    stats["process_memory_mib"] = None
            if distributed:
                gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered, stats)
                stats = gathered
            if rank == 0:
                write_packet(response, {"ok": True, "result": result, "memory": stats})
        except BaseException as exc:
            if rank == 0:
                write_packet(response, {"ok": False, "error": traceback.format_exc(), "error_info": remote_error(exc)})
            # Collective failures cannot safely continue with a desynchronized rank.
            raise
    try:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        if distributed:
            torch.distributed.destroy_process_group()
        if rank == 0:
            (directory / "shutdown.json").write_text(json.dumps({"status": "adapter_closed"}))
    except BaseException as exc:
        if rank == 0:
            (directory / "shutdown.json").write_text(json.dumps({"status": "failed", "error_info": remote_error(exc)}))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    serve(parser.parse_args().directory.resolve())
