# A reproducible real-model rollout

[Manual](README.md) / First real rollout · [简体中文](../zh/first-rollout.md)

Run **OpenVLA-7B × WorldGym**, using the released Bridge test episode 2 and the
instruction “put the eggplant in the pot”. The fixed recipe performs **20 policy
calls and 20 world-model calls**, with seed 0. It tests integration, not task success.
At the declared 20 FPS, the expected 21-frame video is approximately 1.05 seconds;
this deliberately short acceptance test is not a complete-task demonstration.

## Requirements

| Resource | Requirement |
| --- | --- |
| System | Linux x86-64, Python 3.11, uv, FFmpeg, compatible NVIDIA driver |
| GPUs | Two distinct CUDA GPUs, at least 18 GiB available for VLA and 20 GiB for WM; admission rechecks availability |
| Assets | About 25.2 GB decimal, content-pinned in `configs/first_rollout_assets_v1.json` |
| Disk | Allow about 50 GB for assets, package downloads and a new runtime; more if the package cache cannot hard-link |
| Access | WorldGym checkpoint on Google Drive; SD3 VAE requires upstream access/terms |

## 1 · Prepare assets explicitly

From a source checkout, run:

```bash
bash scripts/setup/core.sh
bash scripts/setup/first_rollout_assets.sh
```

This preparation disables proxies and uses configurable PyPI/HF mirrors. GitHub and
Google Drive retain their actual hosting; the script does not route those through HF.
Set `HF_TOKEN` only when required by the VAE's access terms; never put it in a config.
Existing assets are reused only after content verification. A different checkpoint
must use a different recipe/version, not silently replace the asset lock.

## 2 · Run the complete acceptance recipe

Choose two available GPU indices and a **new** output directory:

```bash
environment/core/bin/python scripts/acceptance/first_rollout.py \
  --output results/delivery/my_first_rollout --vla-gpu 0 --wm-gpu 1
```

The command verifies all pinned assets, builds the current wheel, creates new core
and model environments, installs explicit dependency versions, checks their imports
from outside the source directory, and invokes the **installed CLI and shared executor**.
VLA and WM run in separate persistent processes; this pair shares one compatible
model dependency environment. It does not invent a second rollout implementation.

The package cache is allowed and disclosed. `--offline` requires all dependency
archives already cached; a cache miss fails rather than using another environment.
This validates fresh installation, **not** a cold download of every weight. Model
inference is offline. No hidden model environment or user site-packages is reused.

## 3 · Inspect the result

| File under your output directory | Meaning |
| --- | --- |
| `report.json` | Overall status, installation steps, asset-lock and wheel hashes, failure details |
| `evaluate.stdout` | Shared framework execution and artifact-audit report |
| `inspect.stdout` | Artifact inspection from the freshly installed core |
| `core_freeze.stdout`, `models_freeze.stdout` | Actually installed package versions |
| `suite.json`, `environments.json` | Exact inference configuration and explicit runtime mapping |
| `rollouts/attempt_*/loop_long/episode_000002/seed_000/` | Continuous video, actions, metadata and per-chunk records |

A pass requires 20 calls from **both real models**, completed artifacts and successful
inspection. Preflight, asset availability or a synthetic fixture cannot count as a pass.
A failure keeps logs and partial artifacts. Use a new output directory after correcting
it; do not overwrite the failed receipt. Installed environment paths are not relocatable.

## Troubleshooting

- **Asset hash mismatch:** stop; check the pinned content and download source. Do not regenerate the lock just to pass.
- **Gated VAE:** accept the upstream terms and provide authorized access, or place the exact permitted assets locally.
- **Offline package miss:** omit `--offline` to use the mirror, with proxies still disabled.
- **Driver/library mismatch:** fix the host driver installation; the recipe does not modify system libraries.
- **GPU admission failure:** select two GPUs with enough free memory. Budgets are caps, not a physical-quality guarantee.

[Support evidence](support.md) · [Longer experiments](workflows.md)

New environments live under `environment/verification/first_rollout_<id>/`; the report records the location. Results contain receipts and rollout artifacts rather than virtual environments. The recipe also checks finite `[20,1,10]` native/converted action arrays and the expected 21-frame output.
