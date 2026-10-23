# Installation & asset management

[Manual](README.md) / Installation · [简体中文](../zh/installation.md)

## Keep three layers separate

| Layer | Location | Responsibility |
| --- | --- | --- |
| Core | `environment/core` | CLI, contracts, scheduling and artifact inspection |
| Model runtimes | `environment/vla/`, `environment/world_models/` | Official inference dependencies; registry overrides can point elsewhere |
| Assets | `checkpoints/`, `datasets/`, `third_party/` | Weights, data and separately obtained upstream source |

Run from the source root. Do not install every optional extra into one environment:
OpenVLA and SpatialVLA require different Transformers versions.

```bash
bash scripts/setup/core.sh
# Optional: weight-free development tools / DROID Parquet support.
CORE_DEV=1 CORE_DROID=1 bash scripts/setup/core.sh
environment/core/bin/python -m cli environments
environment/core/bin/python -m cli models
```

Core setup requires a local Python 3.11; set `CORE_PYTHON=/absolute/path/to/python3.11`
if automatic selection is unsuitable. It does not silently download Python.

## Network policy

Setup helpers source [common.sh](../../scripts/setup/common.sh), remove proxy variables and keep
caches within this project. Defaults use a PyPI mirror and `HF_ENDPOINT=https://hf-mirror.com`.
Override `PYPI_MIRROR` / `HF_ENDPOINT` explicitly if needed. Authentication and gated-model
terms still apply. Source repositories, Drive and GCS are downloaded from their actual hosts.

Do not copy entire venvs between machines: interpreter symlinks, shebangs, compiled extensions
and CUDA versions can be host-specific. Rebuild from the checked-in scripts and requirements.
Reuse uv's cache rather than treating a copied environment as portable.

## Selective installation

| Family | Setup | Weight acquisition |
| --- | --- | --- |
| OpenVLA / SpatialVLA / Octo | `scripts/setup/vla_envs.sh MODEL` | `scripts/download/vla_hf.sh MODEL` |
| RT-1-X | `scripts/setup/vla_envs.sh rt1` | `scripts/download/rt1_x.sh` |
| GR00T N1.5 / N1.7 | `scripts/setup/groot_n15.sh` / `groot_n17.sh` | `scripts/download/vla_hf.sh`; inspect accepted model names first |
| OpenPI DROID family | `scripts/setup/openpi.sh` | `scripts/download/openpi.sh` / `droid_specialists.sh` |
| SmolVLA / DreamZero | `scripts/setup/smolvla.sh` / `dreamzero.sh` | Model-specific selectors; see scripts before downloading |
| WorldGym | `scripts/setup/worldgym.sh` | Setup's checkpoint helper; official Drive host |
| Cosmos DROID EEF FD | `scripts/setup/cosmos3_nano_droid_fd.sh` | `scripts/download/cosmos3_nano_droid_fd.sh` |
| Ctrl / A2World / GE-Sim / τ₀ | Corresponding `scripts/setup/*.sh` | Setup helper + declared upstream assets |
| IWS / IRASim / DreamDojo / DreamTrue / PlayWorld | Corresponding `scripts/setup/*.sh` | Corresponding `scripts/download/*.py --help` |

This table is an entry-point map, not an instruction to run every installer.
[Setup sources](../../scripts/setup/) · [Download sources](../../scripts/download/) · [Per-model protocols](models.md).
Inspect each script's options; downloads can be large.

## Paths and environment selection

[environment/registry.toml](../../environment/registry.toml) maps adapter environment keys to interpreters.
Reusable configurations use project-relative paths. For separately installed runtimes, set
`VLAWORLDEVAL_ENVIRONMENTS` to a private JSON file:

```json
{"schema_version": 1, "paths": {"vla.openvla": "/absolute/path/to/openvla_env"}}
```

Override directories must exist and be absolute. Do not commit private paths, keys or tokens.
A path override does not guarantee compatible package versions.

```bash
environment/core/bin/python -m cli env-audit --environment vla.openvla --dependencies
environment/core/bin/python -m cli verify-assets --directory checkpoints/vla/openvla-7b
```

## Wheel users

After installing a locally built wheel, `vlaworldeval init --directory /absolute/new_workspace`
creates an empty asset workspace; set `VLAWORLDEVAL_ROOT` to it.
Initialization downloads nothing and refuses nonempty destinations.
Use the source checkout for installation scripts and official-runtime patches.

[Quick start](quickstart.md) · [Troubleshooting](troubleshooting.md)
