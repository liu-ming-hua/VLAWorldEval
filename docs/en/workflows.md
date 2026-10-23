# Run, replay & evaluate

[Manual](README.md) / Workflows · [简体中文](../zh/workflows.md)

## A · Policy × world-model loop

1. Select a [registered pair](compatibility.md) and prepare [assets](installation.md).
2. Use `configure` to create a new local suite with your GPUs, budgets and optional `--chunks N`.
3. Run `evaluate --check-only`; inspect errors before `evaluate --execute`.
4. Inspect per-job artifacts. Keep cameras separate and videos continuous.

```bash
environment/core/bin/python -m cli configure \
  --config configs/experiments/openvla_worldgym_v1.json \
  --output configs/experiments/my_long_loop.json \
  --vla-gpus 0 --wm-gpus 1 --budget-gib 20 --vla-budget-gib 18 --chunks 100
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/my_long_loop.json --experiment loop_smoke --check-only
```

Increasing chunks does not make a dataset/window infinite. Data admission can reject insufficient history
or unavailable actions. Do not silently repeat frames or ground-truth future to extend a run.

## B · Recorded actions, no VLA

Use the **same framework workers** to test a WM independently. Preparation converts a native episode into
an inference packet and a separate reference packet. Real future images must never enter inference.

```bash
environment/core/bin/python scripts/benchmarks/native_replay.py --help
environment/core/bin/python -m cli replay --help
```

Example after preparing native input/reference packets:

```bash
environment/core/bin/python -m cli replay \
  --config configs/replay/worldgym_bridge_native_v1.json \
  --input results/my_case/input.npz --reference results/my_case/reference.npz \
  --output results/my_case/replay --admit-only

# Remove --admit-only only when input/configuration and resources are ready.
```

Packet paths above are placeholders for preparation outputs; use paths actually returned by the helper.
Choose `--model` and dataset according to the [native contract](models.md), not one universal DROID conversion.
Use the preparation helper inside the required model/data runtime if a native converter imports
Torch, FK libraries or other optional dependencies.

## C · Frozen multi-case studies

`native-study` freezes case identity, config/input/reference hashes and seeds before inference.
Subset selection is outcome-independent; report planned, available, completed and scored counts separately.

```bash
environment/core/bin/python -m cli native-study prepare --help
environment/core/bin/python -m cli native-study run --help
environment/core/bin/python -m cli native-study score --help
environment/core/bin/python -m cli replay-report --help
```

For an already prepared study:

```bash
environment/core/bin/python -m cli native-study run \
  --directory results/my_study --exclusive-gpu
environment/core/bin/python -m cli native-study score \
  --directory results/my_study --output results/my_scoring --workers 2
```

Scoring writes a separate version directory; changing metrics does not require reloading the WM.
Use `replay-report --directory RUN --plan PLAN --metrics SCORING --output NEW_REPORT`
to select a completed metric version explicitly. Do not merge duplicate case outcomes by choosing the best.

## Resource and throughput options

To resume a full cohort across stopped runs and transferred results, use
\`native-study remainder --plan FULL_PLAN --directory NEW_REMAINDER --gpus 0 1 2 3\`
with repeated \`--source-directory RUN_ROOT\`. It validates completed artifact hashes and freezes
disjoint missing-case plans without loading models. Keep the full parent plan as the scoring denominator;
the remainder alone is not a complete test set.

- `--exclusive-gpu` provides cooperative queue locking, not eviction of other users.
- `--reuse-worker` keeps explicitly supported/reset-capable models resident across episodes.
- `--gpu` / `--gpus` relocate permitted jobs without changing scientific inputs.
- `score --metric-gpus ... --metric-frame-batch-size N` batches LPIPS evaluation, not inference.
- Cosmos native batching has a separate measured qualification path under
  [native_batch_benchmark](../../evaluation/execution/native_batch_benchmark.py)
  and [native_batch_stream](../../evaluation/execution/native_batch_stream.py).
  Do not enable arbitrary batch sizes on unsupported adapters.

GPU budgets do not guarantee throughput. Image rate, action rate, diffusion steps and AR chunks are different quantities.
For every optimization, keep checkpoint, action conversion, resolution, seeds and scoring horizon traceable.

[Outputs](artifacts.md) · [Metric interpretation](reliability.md)
