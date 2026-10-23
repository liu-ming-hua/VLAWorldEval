# Shared evaluation core

The same contracts, isolated workers and artifact pipeline serve policy loops
and recorded-action native replay.

| Folder | Responsibility |
| --- | --- |
| `datasets/` | Native input/reference preparation |
| `planning/` | Schemas, pair admission and frozen plans |
| `execution/` | Resource admission, worker IPC and scheduling |
| `reporting/` | Artifacts, diagnostics and metrics |
| `validation/` | Offline, environment and delivery checks |

Public contracts are exported from `evaluation`; adapters own model inference.
Recorded future must remain outside the generation path.

[Architecture](../docs/en/architecture.md) · [Evaluation boundaries](../docs/en/reliability.md) · [中文手册](../docs/zh/README.md)
