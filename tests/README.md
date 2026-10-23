# Test suite

Tests are source-level developer tools, not part of the installed inference wheel.
All commands below run from the repository root. No test command installs dependencies,
downloads assets, or certifies a world model's physical reliability.

## Find a test by responsibility

| Directory | Responsibility |
| --- | --- |
| `core/` | Contracts, pair admission, planning, closed loops, replay, IPC, timing, artifacts and metrics |
| `models/` | Model-specific input/output protocols, conversion, history and lifecycle using synthetic inputs/mocks |
| `engineering/` | Environments, setup/download safety, portability, docs, packages and source export |
| `integration/` | Separately acquired official data, native upstream code and optional model-runtime parity |
| `support/` | Shared builders and lightweight worker adapters; not collected as tests |
| `conftest.py` | Cost-tier selection, explicit integration opt-in and optional machine-readable reports |

## Select by cost, independently of directory

| Tier | What runs | Command |
| --- | --- | --- |
| `unit` | Weight-free CPU contracts, small arrays and synthetic files | `environment/core/bin/python -m pytest tests -m unit` |
| `process` | Weight-free CPU workers, subprocess cleanup, local HTTP, shell and FFmpeg | `environment/core/bin/python -m pytest tests -m process` |
| `integration` | Optional official assets or native runtime; never automatic | `environment/core/bin/python -m pytest tests --run-integration -m integration` |

`python -m pytest tests` runs **both unit and process tiers**. The `integration/` directory
is not even imported by default. `--run-integration` enables collection; `-m integration`
selects only that tier. Missing optional dependencies/assets remain explicit skips.
Run a model-specific integration module with its registered interpreter when it needs native
dependencies. These retained parity tests do not construct learned models or perform GPU inference;
real learned rollouts remain separately configured framework runs with resource admission.

Process is a conservative module-level classification: a mixed module containing IPC/video
checks stays in the process tier even if some individual cases are cheap. This is a dependency
boundary, not a promised wall-clock limit. Each collected case has exactly one cost tier.

## Run the engineering gate

```bash
CORE_DEV=1 bash scripts/setup/core.sh
bash scripts/check_project.sh                  # default: all weight-free checks
TEST_TIER=unit bash scripts/check_project.sh    # fast iteration
TEST_TIER=process bash scripts/check_project.sh # workers / video / packaging-related checks
```

GitHub Actions tests Python 3.10 and 3.11 independently in both weight-free tiers.
Wheel installation and source-export acceptance also run in each Python process-tier job.
CI does not silently enable integration just because an asset happens to be installed.

## Inspect ownership and observed cost

```bash
environment/core/bin/python scripts/checks/test_suite.py
environment/core/bin/python scripts/checks/test_suite.py --json
environment/core/bin/python -m pytest tests --durations=20 \
  --suite-report .cache/quality/tests.json
```

The inventory maps test modules to imported production modules, counts test functions, and flags
exact duplicate function bodies for review. It is **not executed-line or branch coverage**;
overlapping imports do not prove redundant tests. The optional suite report records the selected
parameterized cases, their tiers, outcomes and observed durations, including collection skips.
CI publishes these reports as `test-audit-*` artifacts; generated reports stay outside source control.

Before consolidating tests, compare names, parameterizations and assertions—not just file counts.
Keep boundary/failure cases. Put reusable builders in `support/`, not in another collected test file.
Changes to checkpoints, data splits or rollout settings require native evaluation evidence;
engineering tests and per-trajectory cherry-picked success are not that evidence.

[Contributing](../CONTRIBUTING.md) · [English manual](../docs/en/README.md) · [中文手册](../docs/zh/README.md)
