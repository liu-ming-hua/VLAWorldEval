# Declarative configuration

| Location | What it declares |
| --- | --- |
| `model_catalog.json` | Public model input/output and training-domain descriptions |
| `runtime_registry.json` | Constructors, runtime keys, assets and admitted pair profiles |
| `pairs/` | Action/view/time protocol manifests |
| `experiments/` | Data windows, sampling, seeds and GPU budgets |
| `replay/` | Native WM-only replay templates |
| `benchmarks/` | Native-domain capability inventories |
| `calibration/`, `kinematics/` | Explicit conversion constants and robot geometry |
| `world_models/`, `vlas/` | Model-specific runtime defaults |

Reusable paths are project-relative. Use `cli configure` for machine-specific suites;
do not overwrite templates. Equal tensor widths do not imply protocol compatibility.

[Model reference](../docs/en/models.md) · [Compatibility](../docs/en/compatibility.md) · [中文手册](../docs/zh/README.md)
