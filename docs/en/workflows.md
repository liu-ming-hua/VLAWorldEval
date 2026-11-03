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

## A2World: released example through shared replay

The pretrained DROID example contains three initial views and **one 20×14 action
chunk**. It supplies no recorded future RGB. This route verifies generation plumbing,
not long-horizon performance or prediction accuracy. It uses the same isolated worker,
action saving, per-camera continuous videos and summary format as other native replays.

```bash
environment/core/bin/python -m cli prepare-replay \
  --config configs/replay/a2world_pretrained_native_v1.json \
  --dataset datasets/a2world/official/droid \
  --output results/a2world_case/input.npz --chunks 1
environment/core/bin/python -m cli replay \
  --config configs/replay/a2world_pretrained_native_v1.json \
  --input results/a2world_case/input.npz \
  --output results/a2world_case/replay --admit-only
```

Remove `--admit-only` after admission succeeds. Do not pass a fabricated `--reference`.
Use a fresh result directory and select your GPU in a copied configuration. Alternatively,
`scripts/replay/a2world.py --output results/a2world_example --gpu 0 --prepare-only`
prepares an effective configuration; without `--prepare-only` it invokes shared replay.

Install with `scripts/setup/a2world.sh` after obtaining access to NVIDIA's required
base tokenizer. The explicit `tokenizer_path` must match
`base_checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth`.
An empty base directory is not a complete installation: admission rejects a missing
file before launching a model worker. A mirror returning 403 does not mean this
prerequisite can be skipped. The model has no state output.

## GE-Sim 2.0: calibrated action-mode replay

Start a dedicated GE-Sim server with `scripts/serve/ge_sim_v2.sh` after installation.
Choose its GPU through `CUDA_VISIBLE_DEVICES` and its port through `GE_SIM_PORT`;
set `kwargs.server_url` in a copied replay configuration to the same local address.
The shared worker is a CPU client (`gpus: []`), so its memory budget does **not**
limit the separately launched GPU server. Stop your own server after the run.

```bash
environment/core/bin/python -m cli prepare-replay \
  --config configs/replay/ge_sim_v2_action_native_v1.json \
  --dataset datasets/ge_sim_v2/official/demo_000 \
  --output results/gesim_case/input.npz --chunks 2
environment/core/bin/python -m cli replay \
  --config configs/replay/ge_sim_v2_action_native_v1.json \
  --input results/gesim_case/input.npz --output results/gesim_case/replay
```

This uses the first 50 of 456 official actions, uncompressed, in two consecutive
25-frame chunks. Three calibrated 384×512 views and 16-D absolute joint commands
retain the upstream order. The remaining actions are counted, not padded or repeated.
The example has no ground-truth future RGB reference. `action` conditioning derives
trajectory bands from commands via FK with a stationary torso/head assumption;
`episode` conditioning instead uploads recorded future end-effector trajectories.
They are different experiments. The shared configuration deliberately selects `action`.
Server weights/configuration and logs must be retained with the run: a CPU client
receipt alone does not certify which remote checkpoint was used.

### Complete GE-Sim episode with an owned server

```bash
environment/core/bin/python scripts/replay/ge_sim_v2.py \
  --output results/gesim_full --gpu 0 --memory-gib 28 --chunks all
```

The entry checks free GPU memory, acquires the framework's cooperative GPU lease,
starts a dedicated loopback server on an available port, seeds that server and caps
its PyTorch allocator. It prepares data and executes **shared native replay**,
builds the offline report, and stops its own server on success or error. Other users'
processes are untouched. Proxy variables are removed from the server environment;
model loading is offline. Dependencies, calibrated example and checkpoint must already exist.
`--server-config PATH` selects another explicitly recorded server configuration.

For the released 456-action example, `all` means 18 chunks of 25 and a final chunk of
6: three continuous 457-frame videos including initial RGB, about 28.6 s at 16 FPS.
The native server internally repeats the final action token to fill its fixed 25-token
network window, then returns only six frames. The framework sends and saves exactly
456 source actions. The short chunk is terminal: the server's internal padded history
must not be used as the start of another continuation.

The full configuration uses replay schema **v3**, a list of numeric future-action
chunks with only the last chunk allowed to be short and one output frame per action.
`actions.npy` is flat `[456,16]`; `action_offsets.npy` locates the original chunk
boundaries. V1 fixed-size and V2 structured replay remain unchanged. V3 currently
supports isolated video replay, not reference scoring, visual re-anchoring or pooled workers.

Outputs include server configuration/source/asset hashes, lifecycle receipt,
`replay/summary.json`, per-view videos, and `report/index.html`.
`boundary_diagnostics.json` reports decoded-frame changes and near-static runs; it
cannot establish semantic view correctness, successful manipulation or physical reliability.
In action mode the official server seeds memory from initial RGB, then samples its
own generated-frame history. FK-derived trajectory bands assume stationary torso/head.

The pinned upstream revision requires the reviewed
`scripts/setup/patches/ge_sim_v2_finite_tail.patch`
for short band/camera conditioning windows. Setup applies it idempotently and refuses
source changes that conflict with the patch. This is **upstream plus a compatibility patch**, not an
unmodified-official parity claim. Source hashes in the receipt identify the executed code.

## GE-Sim action-mode validation

The official `examples/replay.py` selects `conditioning="episode"` and uses recorded
future EEF trajectories. Our action replay selects `conditioning="action"`, rendering
trajectory bands and moving wrist-camera geometry from joint commands via the official FK.
Comparing those two modes directly is not adapter parity.

```bash
environment/world_models/ge_sim_v2/bin/python scripts/acceptance/world_model_parity.py \
  --model ge_sim --input datasets/ge_sim_v2/official/demo_000 \
  --output results/gesim_parity --gpu 0 --memory-gib 28 --chunks 2 --tail-actions 6
```

This compares the official `WorldModelEnv` in action mode against the framework adapter.
It launches two independently seeded servers with the same weights/configuration and
compares decoded RGB and predicted state, including HTTP JPEG transport. The test uses
a **56-action prefix** (25 + 25 + 6), not the episode's last six actions. A separate full
replay checks complete-episode coverage. Temporary raw comparison arrays are removed;
`report.json` retains hashes, frame counts, errors and process lifecycle records.

A conditioning check perturbs every recorded state/EEF/camera row after frame zero
while holding commanded actions and initial conditions fixed. Unchanged FK bands and
camera conditions establish absence of those future-state dependencies in this tested
route. This is not a physical counterfactual test or a guarantee for arbitrary upstream versions.

## Recorded-state replay and action-sensitivity controls

Ctrl-World's native DROID protocol conditions on **measured absolute EEF/gripper
state trajectories**, not raw actuator commands. Use the pinned official examples
before interpreting a generic policy's output in this space.

```bash
# Installed model assets required; no VLA is loaded.
environment/world_models/ctrl_world/bin/python scripts/acceptance/prepare_world_model_cases.py \
  --model ctrl_world --output results/ctrl_cases

environment/core/bin/python scripts/acceptance/branch_example.py --model ctrl_world \
  --prepared results/ctrl_cases/cases/ctrl_released_long_ep199_seed0 \
  --output results/ctrl_controls --gpu 0 --seed 0

python -m cli branches --manifest results/ctrl_controls/branches.json \
  --output results/ctrl_controls/evaluation --check-only
python -m cli branches --manifest results/ctrl_controls/branches.json \
  --output results/ctrl_controls/evaluation --execute
python -m cli report --source results/ctrl_controls/evaluation/summary.json \
  --output results/ctrl_controls/report
```

Repeat for `ctrl_released_long_ep18599_seed0` with a different output directory.
Both are official published examples; this is not the full paper test set or an
independent held-out estimate. Keep the configuration fixed across cases.

| Branch | Intervention | Valid interpretation |
| --- | --- | --- |
| `recorded` | All recorded native states, inclusive windows, original tail convention | Paired RGB fidelity against the recorded future |
| `hold_initial_pose` | Repeat the initial absolute EEF pose and gripper value | Zero-motion condition, **not** zero coordinates |
| `shuffle_future` | Seeded permutation of unique future states; shared window anchors and initial context preserved | Temporal-order sensitivity; potentially nonphysical trajectories |

Only `recorded` receives a reference packet. Alternative branches must not be scored
as if the original trajectory were their true physical outcome. The framework keeps
physical branch contrasts `unknown`, while separately reporting same-seed generated
video MAE/MSE as `prediction_sensitivity`. These values use decoded H264 RGB; baseline
reference metrics use raw predictions before encoding. Equal seeds alone do not
certify identical diffusion-noise traces.

The report provides synchronized videos, prediction/recorded-future comparisons,
PSNR, SSIM, frozen-initial-image baseline, temporal RGB-change error, motion diagnostics
and chunk-boundary changes. These measure appearance and sensitivity, not robot
contact accuracy, task success or policy-ranking validity.
