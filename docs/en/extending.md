# Add a model or pair

[Manual](README.md) / Extending · [简体中文](../zh/extending.md)

## Integration checklist

1. **Specify the native contract first.** Pin the upstream source/checkpoint revision, robot,
   training domain, views, image layout, action semantics, history and timebase.
2. **Add one adapter** under `adapters/vla/` or `adapters/world_models/`.
   Keep official inference logic and lazy imports inside the model boundary.
3. **Register one environment key** in [registry.toml](../../environment/registry.toml).
   Add setup/download entry points reusing proxy removal, project-local caches and integrity helpers.
4. **Describe the model** in [model_catalog.json](../../configs/model_catalog.json).
   A catalog entry alone does not authorize a VLA × WM pair.
5. **Bind the constructor** in [runtime_registry.json](../../configs/runtime_registry.json).
   Declare executable assets, distributed support and explicit dataset loaders.
6. **For a pair**, add a manifest under [pairs/](../../configs/pairs/) and a suite under
   [experiments/](../../configs/experiments/). Implement necessary action/view bridges, with units,
   coordinates, rates, rotation and gripper conventions explicitly tested.
7. **Test without weights**, then run a native sample in the isolated runtime.
   Record shapes, official-source parity, lifecycle and asset identity separately from scientific quality.
8. **Update both manuals** and the compatibility table. Never promote startup-only integration to full pairing.

## Adapter interfaces

Start with [base.py](../../adapters/base.py), [contracts.py](../../evaluation/contracts.py)
and an adapter with a genuinely similar native protocol—not merely the same action dimension.
Use the shared evaluator and native replay; do not copy rollout orchestration into your model runner.

Reusable converters belong in `adapters/actions/` and `adapters/observations/`.
Plugins use explicit module/class references. They should fail on incompatible shapes rather than
padding an unknown robot's action vector.

## Validate in stages

```bash
CORE_DEV=1 bash scripts/setup/core.sh
bash scripts/check_project.sh
environment/core/bin/python -m cli models
environment/core/bin/python -m cli runtimes
environment/core/bin/python -m cli capabilities
```

The weight-free suite must not download or initialize CUDA.
Real model tests need their own environment, checkpoint and explicit GPU opt-in.
Inspect official native-domain replay against reference before drawing policy conclusions.

## Contribution boundaries

Keep weights, environments, upstream clones, caches and full results out of Git.
Use project-relative shared configuration and explicit private overrides.
Follow [CONTRIBUTING](../../CONTRIBUTING.md), [release rules](release.md)
and [reliability boundaries](reliability.md).
