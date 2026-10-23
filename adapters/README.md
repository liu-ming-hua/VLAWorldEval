# Model adapters

| Folder | Boundary |
| --- | --- |
| `vla/` | Native policy loading and inference |
| `world_models/` | Native world-model loading and generation |
| `actions/` | Explicit action conversions and kinematics |
| `observations/` | Camera layout and named-view bridges |
| `backends/` | Optional serving backends |

Adapters reuse official inference logic inside isolated runtimes.
The shared evaluator owns orchestration; catalog registration alone does not create a compatible pair.

[Model reference](../docs/en/models.md) · [Add an adapter](../docs/en/extending.md) · [中文手册](../docs/zh/README.md)
