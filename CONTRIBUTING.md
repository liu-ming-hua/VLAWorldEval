# Contributing

Keep adapters small, contracts explicit and execution shared.

1. Add the model under `adapters/vla/` or `adapters/world_models/`.
2. Put reusable conversions in `adapters/actions/` or `adapters/observations/`.
3. Declare its environment, catalog contract, constructor and assets.
4. Add a pair manifest only when camera, action, time and robot-input semantics match.
5. Add weight-free contract tests, then separately verify native inference.
6. Update both [English](docs/en/README.md) and [Chinese](docs/zh/README.md) manuals.

Follow the [integration checklist](docs/en/extending.md).
Never create a parallel per-model evaluation loop.
Shared configuration must not contain personal server paths or credentials.

## Before submitting

```bash
CORE_DEV=1 bash scripts/setup/core.sh
bash scripts/check_project.sh
```

Routine checks must not download or load models. GPU tests require explicit opt-in and owned resources.
Do not equate engineering checks with [scientific reliability](docs/en/reliability.md).

## What belongs in Git

Commit original code, small declarations, tests and current user documentation.
Do not commit weights, datasets, venvs, upstream clones, caches or `results/`.
Only reviewed presentation assets explicitly listed in the source-export allowlist may be added under
`docs/assets/`; include provenance and review applicable media/data terms.
Do not force-add ignored files.

For the initial public repository, use the [reviewed source export](docs/en/release.md),
not inherited upstream history. Preserve attribution and [license boundaries](LICENSE_SCOPE.md).
