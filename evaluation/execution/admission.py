"""Shared, read-only job/data/GPU admission; never starts model workers."""
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path


class ResourceAdmissionError(RuntimeError):
    pass


@contextmanager
def native_gpu_lease(root, gpus):
    """Cooperating native queues share one resident model per physical GPU.

    This is not a device-wide reservation: other users' jobs are untouched and
    ordinary free-memory admission remains mandatory. Kernel locks release on
    crashes; lock files are never removed or replaced.
    """
    from vlaworldeval.asset_lock import asset_destination_lock
    if (not isinstance(gpus, list) or any(type(g) is not int or g < 0 for g in gpus)
            or len(gpus) != len(set(gpus))):
        raise ValueError("Native GPU leases require unique nonnegative GPU IDs")
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            destination = Path(root) / ".cache/vlaworldeval/native_gpu_leases" / f"gpu-{gpu}"
            try:
                stack.enter_context(asset_destination_lock(root, destination))
            except ValueError as exc:
                if "Asset operation already running" not in str(exc):
                    raise
                raise ResourceAdmissionError(f"GPU {gpu}: another exclusive native queue holds the lease") from exc
        yield


def select_jobs(plan, experiments=(), seeds=()):
    jobs = plan["jobs"]
    for requested, key, field in ((experiments, "id", "--experiment"), (seeds, "seed", "--seed")):
        known = {job[key] for job in jobs}
        unknown = set(requested) - known
        if unknown:
            raise ValueError(f"{field}: unknown selection {sorted(unknown)}")
        if requested:
            jobs = [job for job in jobs if job[key] in requested]
    if not jobs:
        raise ValueError("No matching jobs for the selected experiments/seeds")
    return jobs


def admit_episodes(registry, profile, root, dataset, jobs, horizon):
    episodes = {}
    for job in jobs:
        key = (job["episode_index"], job["chunks"], job["mode"])
        if key not in episodes:
            episodes[key] = registry.load_episode(profile, root, dataset, job, horizon)
    return episodes


def _admit_gpu_budgets(budgets):
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits",
        ], text=True, timeout=10)
        devices = {int(row.split(",")[0]): [int(x.strip()) for x in row.split(",")[1:]]
                   for row in output.strip().splitlines()}
        if not devices or any(len(values) != 2 for values in devices.values()):
            raise ValueError("Incomplete GPU memory response")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ResourceAdmissionError(f"GPU query failed: {exc}") from exc
    for gpu, required in budgets.items():
        if gpu not in devices or devices[gpu][0] < (required + 0.5) * 1024:
            raise ResourceAdmissionError(f"GPU {gpu}: need {required + 0.5} GiB free for resident workers; available {devices.get(gpu)} MiB")
    return {"observed_mib": devices, "budget_gib": budgets}


def policy_resource_admission(gpus, memory_gib):
    """One distributed policy only; reserve CUDA/NCCL space on every rank."""
    return {**_admit_gpu_budgets({gpu: memory_gib + 3 for gpu in gpus}),
            "policy_process_gib": memory_gib,
            "limit_scope": "PyTorch allocator; additionally reserves 3 GiB/rank plus 0.5 GiB margin"}


def resource_admission(runtime, needs_vla):
    wm_limit = runtime["max_process_memory_gib_per_gpu"]
    vla_limit = runtime.get("vla_memory_gib", min(8, wm_limit))
    budgets = {}
    for gpu in runtime["world_model_gpus"]:
        budgets[gpu] = budgets.get(gpu, 0) + wm_limit + 3
    if needs_vla:
        for gpu in runtime["vla_gpus"]:
            budgets[gpu] = budgets.get(gpu, 0) + vla_limit + 1
    return {**_admit_gpu_budgets(budgets), "vla_process_gib": vla_limit,
            "world_model_process_gib": wm_limit,
            "limit_scope": "PyTorch allocator; admission additionally reserves 3 GiB/WM rank and 1 GiB/VLA for CUDA/NCCL"}
