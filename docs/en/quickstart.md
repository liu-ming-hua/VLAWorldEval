# Your first rollout

[Manual](README.md) / Quick start · [简体中文](../zh/quickstart.md)

## 1 · Check the lightweight core

From the source root, with Python 3.11, uv and FFmpeg available:

```bash
bash scripts/setup/core.sh
environment/core/bin/python -m cli demo --output results/first_demo --chunks 30
environment/core/bin/python -m cli inspect --artifact results/first_demo
environment/core/bin/python -m cli capabilities
```

Expected: a synthetic continuous video, saved action arrays and an inspectable artifact.
This checks the framework—not pretrained models or robot task success.
Every output directory must be new.

## 2 · Prepare one real pair

For OpenVLA + WorldGym:

```bash
bash scripts/setup/vla_envs.sh openvla
bash scripts/download/vla_hf.sh openvla
bash scripts/setup/worldgym.sh
environment/core/bin/python scripts/download/worldgym_data.py
```

WorldGym setup also prepares its separately obtained runtime. Its released checkpoint is on
Google Drive; its VAE has upstream access terms. An HF mirror cannot replace these hosts or permissions.
See [installation](installation.md) before downloading.

## 3 · Select your devices

```bash
environment/core/bin/python -m cli configure \
  --config configs/experiments/openvla_worldgym_v1.json \
  --output configs/experiments/my_openvla.json \
  --vla-gpus 0 --wm-gpus 1 --budget-gib 20 --vla-budget-gib 18
```

Budgets are admission/allocator caps, not measured model requirements.
Only adapters declared distributed accept multiple GPU indices.
Do not select GPUs belonging to another workload without permission.

## 4 · Preflight, then execute

```bash
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/my_openvla.json \
  --experiment loop_smoke --check-only
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/my_openvla.json \
  --experiment loop_smoke --execute
```

Check-only validates configuration, local assets, runtimes and data windows without loading models.
Execute additionally performs resource admission. Inspect the returned artifact paths with
`cli inspect --artifact PATH`.

## Next

[Choose another pair](compatibility.md) · [Long rollouts and native replay](workflows.md) · [Understand outputs](artifacts.md)
