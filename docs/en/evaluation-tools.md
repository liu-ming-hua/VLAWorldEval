# Consistency, branches & offline reports

[User manual](README.md) · [简体中文](../zh/evaluation-tools.md)

Use these tools after choosing a [compatible protocol](compatibility.md). They reuse the
native replay executor and do not require a second evaluation pipeline.

## Choose the right check

| Tool | Question | What it cannot establish |
| --- | --- | --- |
| Native parity | Does this adapter reproduce the captured native path? | Physical correctness or all checkpoint versions |
| Action branches | How do predictions change from the same observed context? | Counterfactual correctness without executed branch references |
| Offline report | What ran, what failed, and where are the videos and metrics? | A cross-domain model ranking |

## 1. Compare native inference and the adapter

Install the [real-model assets](first-rollout.md), then run each profile in an environment
containing its model dependencies. The parent command uses the core environment:

```bash
environment/core/bin/python scripts/acceptance/native_parity.py \
  --profile openvla_bridge --python environment/vla/openvla/bin/python \
  --gpu 0 --seed 0 --output results/parity/openvla

environment/core/bin/python scripts/acceptance/native_parity.py \
  --profile worldgym_bridge --python environment/world_models/worldgym/bin/python \
  --gpu 1 --seed 0 --output results/parity/worldgym
```

Choose an installed interpreter from your environment registry. Both captures use the same
checkpoint lock, seed and protocol. Input structure, dtype and values must match exactly;
output tolerances are fixed in [the profile configuration](../../configs/native_parity_v1.json).
Native and adapter processes run separately under a 20 GiB allocation budget.

**Scope matters.** OpenVLA compares against the pinned WorldGym OpenVLA integration, with
observation hooks replacing world-model generation and cloud scoring. It is not the original
OpenVLA dependency environment. WorldGym compares against the pinned runtime with disclosed
CPU-loading/local-VAE compatibility patches. It checks three autoregressive chunks.
Other models have no real-weight parity receipt from this check yet.

For independently captured numeric NPZ traces, use `vlaworldeval parity-check --native ...
--adapter ... --policy ... --output ...`. Captures must follow
[evaluation/validation/parity.py](../../evaluation/validation/parity.py); matching arbitrary
traces is not certification of their provenance.

## 2. Evaluate action branches

The example prepares **recorded actions versus all-zero packed actions** from the official
WorldGym Bridge episode 2. Zero is a numerical intervention, not necessarily a physical hold
command. It downloads nothing and invents no counterfactual reference.

```bash
environment/core/bin/python scripts/acceptance/branch_example.py \
  --output results/branch-inputs --gpu 0 --chunks 3
vlaworldeval branches --manifest results/branch-inputs/branches.json \
  --output results/branches --check-only
vlaworldeval branches --manifest results/branch-inputs/branches.json \
  --output results/branches --execute
```

The version-1 manifest binds a numeric native replay config and every input/reference with
workspace-relative paths and SHA256 hashes. It contains distinct seeds and families with:

- An `id`, `initial_condition.kind` (`observation_only` or `simulator_snapshot`).
- At least two branches, each with `id`, `input: {path, sha256}` and optional `reference`.
- Exactly identical initial views, complete reset/history and instruction within a family.
- Different actions with the same protocol, chunk count and time axis.

A simulator snapshot requires a hashed `evidence` asset. This records the caller's provenance;
it does not automatically restore or certify a simulator. Native references stay outside
inference inputs. Each branch/seed receives an isolated worker; model loading is included in
costs. All planned slots, failures and unfinished slots remain in `summary.json`.

With both executed references, the report computes paired response MSE:
`mean(((prediction_A - prediction_B) - (reference_A - reference_B))²) / 255²`, excluding
the initial frame. Predictions are decoded MP4 RGB; references are raw RGB. This is an
image-space diagnostic, not a contact, trajectory or success metric. A shared seed does not
prove identical diffusion-noise traces. Missing branch truth is **unknown**, never zero error.
Seeds are repeated model samples, not independent physical trajectories.

## 3. Share one offline report

```bash
vlaworldeval report --source results/branches/summary.json \
  --source results/evaluation.json --output results/report
python -m http.server 8080 --directory results/report
```

Omit the second source unless you have a saved `evaluate` report. Native replay
`summary.json` and frozen native-study `study.json` are also accepted. The latter binds
progress to the frozen queue and keeps failed/unrun cases in the denominator. Open `index.html` locally or serve the directory; for a
remote machine forward port 8080 with SSH. Copy the whole report directory to share it.
There are no CDN dependencies. Videos, synchronized playback/seek, per-frame error plots,
reference availability, costs, protocol details and errors are included alongside `report.json`.
Native artifacts are checksum-verified before copying. Planned-but-unrun cases remain visible.

Outputs must use new directories. These commands do not overwrite earlier evidence or
claim model reliability from plausible-looking videos. See [metric limits](reliability.md).

[2026-10-31: OpenVLA / WorldGym parity receipts](../../configs/native_parity_evidence_v1.json).

## Native DROID world-model acceptance

The real-weight checks also have dedicated **Ctrl-World** and **Cosmos DROID community
20k-EMA** profiles. These compare the pinned public inference methods with the adapter,
sharing resident weights to isolate conditioning, recurrent state and output conversion.
They do **not** independently validate the loader or certify pristine upstream CLI behavior.

```bash
# Ctrl: use an available physical GPU; defaults are two AR chunks, 50 diffusion steps.
CUDA_VISIBLE_DEVICES=0 environment/world_models/ctrl_world/bin/python \
  scripts/acceptance/world_model_parity.py --model ctrl_world \
  --output results/ctrl-parity

# Prepare the fixed native long-replay cases. These commands do not run models.
environment/world_models/ctrl_world/bin/python scripts/acceptance/prepare_world_model_cases.py \
  --model ctrl_world --output results/ctrl-inputs
environment/world_models/cosmos3_nano_droid_fd/bin/python scripts/acceptance/prepare_world_model_cases.py \
  --model cosmos_droid --output results/cosmos-inputs

CUDA_VISIBLE_DEVICES=0,1 environment/world_models/cosmos3_nano_droid_fd/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/acceptance/world_model_parity.py --model cosmos_droid \
  --input results/cosmos-inputs/cosmos_ep0/input.npz --output results/cosmos-parity

# Existing shared framework runner; change the prefix for Ctrl.
vlaworldeval native-study prepare --plan results/cosmos-inputs/plan.json \
  --directory results/cosmos-acceptance
vlaworldeval native-study run --directory results/cosmos-acceptance --reuse-worker
```

Use `--reuse-worker` only for a runtime declaring that capability (Cosmos supports it;
Ctrl currently loads once per episode). `--gpu N` can explicitly remap a single-GPU
Ctrl study. The Cosmos preparation freezes a two-GPU, 38 GiB-per-GPU budget; `--gpus A B`
remaps its two ranks without changing the model configuration. Keep proxies disabled and
use already verified local assets; the checks need no new downloads.

| Profile | Native data and timing | Interpretation |
| --- | --- | --- |
| Ctrl-World | Both public episodes 199 and 18599; 5 inclusive waypoints at 5 FPS, advancing 4 frames; official final-state padding, padded predictions excluded | Full published example episodes, not the paper's full test set |
| Cosmos community 20k-EMA | Fixed development episodes 0, 351, 795 from the pinned Cosmos3-DROID snapshot; 16 future actions/frames at 15 FPS; all complete chunks | Engineering acceptance, not independent held-out evaluation or official NVIDIA DROID weights |

Ctrl's parity path encodes only the initial RGB using the upstream method and aligns RNG;
it does not mimic the upstream script's unnecessary sampling of future reference latents.
Cosmos invokes the public JSON inference entry with the same local compatibility config and
shared pipeline. Full effective numerical sample arguments are compared; operational paths
are excluded while conditioning pixels and actions are compared separately. Neither profile
feeds the real future into model inference. Missing physical branch truth remains unknown.
