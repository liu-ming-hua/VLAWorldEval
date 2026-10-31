# Project entry points

Scripts prepare environments/assets or invoke shared framework functionality.
They are not independent per-experiment rollout implementations.

| Folder | Responsibility |
| --- | --- |
| `setup/` | uv environments, pinned upstream source and runtime patches |
| `download/` | Checkpoints/data and integrity checks |
| `vla/` | Native policy input preparation and startup checks |
| `benchmarks/` | Native input/reference preparation |
| `replay/` | Model-specific native protocol probes |
| `checks/` | Offline checks and explicitly requested model diagnostics |
| `acceptance/` | Fresh installation, package and CPU worker verification |
| `analyze/` | Optional analysis tools |
| `ops/` | Recoverable transfers and operations |

`bash scripts/check_project.sh` is the weight-free quality gate.
For inference use the shared CLI, not a new experiment script.

[Installation](../docs/en/installation.md) · [Workflows](../docs/en/workflows.md) · [Release](../docs/en/release.md) · [中文手册](../docs/zh/README.md)

`acceptance/native_parity.py` captures real native/adapter traces with fixed tolerances.
`acceptance/branch_example.py` prepares a recorded-vs-zero WorldGym example without generation.
See the [evaluation tools guide](../docs/en/evaluation-tools.md) for scope and execution commands.

`acceptance/world_model_parity.py` compares pinned Ctrl/Cosmos inference paths with adapters using shared real weights.
`acceptance/prepare_world_model_cases.py` prepares fixed native long-replay plans; execute them with `native-study`, not another inference runner.
