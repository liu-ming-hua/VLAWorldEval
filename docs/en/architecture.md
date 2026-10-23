# Architecture

[Manual](README.md) / Architecture · [简体中文](../zh/architecture.md)

![Execution and evaluation boundaries](../assets/framework.svg)

## Directory map

| Directory | Responsibility |
| --- | --- |
| `adapters/vla/` | Policy loading, native input packing and action inference |
| `adapters/world_models/` | Native WM loading, reset, generation and history |
| `adapters/actions/` | Explicit action-space conversions and kinematics |
| `adapters/observations/` | Camera packing, splitting and declared view bridges |
| `adapters/backends/` | Optional model-serving integration |
| `evaluation/datasets/` | Episode discovery, native converters and reference isolation |
| `evaluation/planning/` | Pair admission, schema validation and frozen plans |
| `evaluation/execution/` | Worker IPC, resource admission, loop/replay scheduling |
| `evaluation/reporting/` | Artifact writing, visual diagnostics and metric reports |
| `evaluation/validation/` | Offline checks, dependency and distribution audits |
| `cli/` | Argument parsing, dispatch and structured output |
| `vlaworldeval/` | Workspace, asset, cache and package resources |
| `configs/` | Model catalog, executable registry, pair manifests and suite templates |
| `scripts/` | Setup, download, native preparation and maintenance entry points |
| `environment/` | Runtime registry, requirements, locks and local uv environments |
| `checkpoints/`, `datasets/` | Protected input assets; not code or cleanup targets |
| `third_party/` | Separately obtained upstream runtime source; not bundled |
| `results/`, `.archive/` | Generated artifacts and recoverable local archives; not bundled |

## Model boundaries

The parent process owns plans, shared evaluation and artifact lifecycle.
Each model runs in its registered interpreter. Optional Torch/JAX/TF libraries are imported lazily,
not during core CLI startup. Process isolation avoids conflicting model dependencies.

`client.py` controls the parent-side IPC connection; `worker.py` serves model requests.
Persistent sessions reuse weights only for adapters declaring reset capability.
Episode image/RNG context must reset; failures cannot be silently counted as clean completion.

The evaluator owns policy/WM handoff. Model-specific scripts prepare native inputs but must not
create a second evaluation loop. Multi-GPU visibility and process cleanup remain worker responsibilities.

## Protocol boundaries

Pair manifests declare action meaning, camera identities, normalization, FPS and history.
Bridges convert known semantics; they do not manufacture physical truth.
Video FPS is separate from action rate, diffusion iterations and future-frame count.

Recorded future belongs exclusively to post-generation evaluation.
Autoregressive generation must feed predicted images back to the model, not secretly reanchor to reference frames.
Any intentionally reanchored diagnostic must be named and reported separately.

## Public API

```python
from evaluation import ActionChunk, ObservationBundle, PairManifest, PolicyWorldModelEvaluator
```

Use documented exports and CLI commands; internal modules may evolve.
[Add an adapter](extending.md) · [Artifacts](artifacts.md)
