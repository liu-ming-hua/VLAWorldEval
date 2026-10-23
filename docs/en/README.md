# User manual

[Project home](../../README.md) · [简体中文](../zh/README.md)

**From the first rollout to a new adapter.** This manual describes the reusable framework,
not a running log of experiments. Choose a task below.

## Get started

| Goal | Read |
| --- | --- |
| Run without weights or GPUs | [Quick start](quickstart.md) |
| Install one model and its assets | [Installation & mirrors](installation.md) |
| Select a valid policy / WM combination | [Compatibility](compatibility.md) |
| Check inputs, outputs and training domains | [Model reference](models.md) |

## How-to guides

- [Run a closed loop or recorded-action replay](workflows.md)
- [Open videos and inspect saved arrays](artifacts.md)
- [Resolve environment, protocol or runtime failures](troubleshooting.md)
- [Add a model without duplicating the evaluator](extending.md)

## Understand & maintain

- [Architecture and directory responsibilities](architecture.md)
- [Metric interpretation and reliability boundaries](reliability.md)
- [Build, review and release the framework](release.md)

### Reading conventions

A catalog entry is not a runnable pair. An available pair is not a physical-validity certificate.
`--check-only` and `--admit-only` do not generate video. Choose fresh output directories.
Commands assume the source root; model environments are installed only when explicitly requested.

Reference data: [model catalog](../../configs/model_catalog.json),
[runtime registry](../../configs/runtime_registry.json),
[environment registry](../../environment/registry.toml).
