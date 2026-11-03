# Model support and validation evidence

[Manual](README.md) / Support evidence · [简体中文](../zh/support.md)

## Five separate questions

| Question | Evidence required |
| --- | --- |
| Is a pair registered? | Valid pair manifest and runtime binding |
| Can its dependencies be installed? | Fresh environment installation, dependency check and package versions |
| Have the real models run together? | Completed policy-loop execution, nonzero VLA/WM calls, valid saved artifacts and clean worker exit |
| Does it match the official implementation? | Native input/conversion checks and a separately scoped numerical parity experiment |
| Can it rank policies reliably? | Appropriate physical-ground-truth validation; none is certified by this support table |

Run `environment/core/bin/python -m cli support` for machine-readable evidence.
The README table is generated from [the versioned evidence catalog](../../configs/support_evidence_v1.json).
The catalog contains report hashes, artifact hashes, executed contracts, model paths,
package versions and historical run identifiers. These are compact receipts, not bundled raw runs.

## Status meanings

- **`historical_pass`**: the recorded execution passed and its remaining artifacts were re-audited.
  This does not assert that current code or a different machine has been tested.
- **`not_verified`**: no qualifying evidence for that dimension and exact pair version.
- **`contract changed`**: evidence exists for that ID, but its execution contract differs.
  Defaults are normalized; descriptive metadata changes alone do not invalidate a contract.
- **`other_contract_evidence`**: earlier evidence for the same adapter names, with a different pair ID.
  It is disclosed but never promoted to a pass for the newer protocol.

For example, historical Ctrl-World v1 loops cannot certify the current v2 temporal
protocol. Environment registry labels such as `ready` are configuration declarations,
not substitutes for a clean-install receipt. Likewise, the existence of a weight path
is not a checkpoint hash verification. Older receipts disclose this limitation.

Use `evaluate --check-only` to check **your current local** assets, runtimes and data;
use `evaluate --execute` for a new inference result. Official numerical parity and
physical reliability remain unverified unless separately demonstrated. State from an
action target is an assumption, not measured future proprioception.

## Maintain the table

After a real shared-framework run:

```bash
environment/core/bin/python scripts/acceptance/support_evidence.py \
  --report results/delivery/my_first_rollout/evaluate.stdout
environment/core/bin/python scripts/acceptance/support_evidence.py --check
```

The importer rejects preflight-only, replay-only, incomplete and failed-artifact records.
It preserves exact pair versions. Both READMEs are rendered from the same data;
`--check` detects stale tables without rewriting them. Raw logs and videos remain
under `results/`, outside the release. Historical receipts do not prove the current
source revision; the fresh recipe additionally records the installed wheel SHA256.

[Run the fresh-install recipe](first-rollout.md) · [Interpret metrics](reliability.md)

To attach a fresh-install receipt, run the importer with `--fresh-report results/delivery/my_first_rollout/report.json`. The CLI compares installed-wheel framework source hashes with the current source; `matches_tested_wheel` applies to code identity only, not current assets, dependencies or hardware.

`official_input_alignment=declared_only_not_certified` distinguishes the declared camera/action protocol from an independently verified official-input audit. It does not imply numerical parity. `last_tested_attempt` identifies the latest imported fresh run when available.

## Native world-model acceptance

These are separate from VLA-pair certification. See the [reproduction commands](evaluation-tools.md)
and [checksum-bound receipts](../../configs/world_model_acceptance_v1.json).

| Model / protocol | Registered | Real native replay | Scoped native-entry parity | Long-replay coverage |
| --- | --- | --- | --- | --- |
| Ctrl-World native v2 | Yes | Passed | Two chunks, exact model conditions and decoded frames | 2 public episodes, 26 calls, 33 / 72 frames per view |
| Cosmos DROID community 20k-EMA | Yes | Passed | Two chunks, exact effective sample settings and decoded frames | 3 development episodes, 62 calls, 465 / 241 / 289 frames |
| GE-Sim 2.0 action mode | Catalogued | Passed | 25 + 25 + 6 actions, exact decoded RGB and predicted state | 1 official example, 456 actions, 19 calls, 457 frames per view |

The Ctrl-World and Cosmos checks share loaded weights; loader parity is outside their scope. GE-Sim uses two independently seeded servers with the same checkpoint/configuration. Cosmos action conversion
was also compared with the public DROID dataset method on all 992 actions (maximum difference
1.2e-7, within predeclared 1e-6 absolute/relative tolerance). This is numerical agreement,
not bitwise action equality. Real future frames are reserved for scoring.

Ctrl-World additionally supports recorded-state replay against real future RGB, plus hold-pose and shuffled-state controls through the [shared branch workflow](workflows.md#recorded-state-replay-and-action-sensitivity-controls). The two released examples are development diagnostics, not independent held-out validation. Its native conditions are measured absolute EEF/gripper states, not raw controller commands. Prediction differences under controls measure sensitivity; only recorded replay has executed future truth.

`support` includes a separate `runtime_acceptance` list with `current_adapter_sources_match`.
It checks only the listed adapter source files against the receipt, not current installed assets
or the entire framework. These results do not upgrade any VLA pair, establish physical action
tracking, or constitute a held-out model-quality ranking. Other registered world models retain
their existing evidence status; no new native-entry parity is implied for them.

## Find a world-model replay route

```bash
environment/core/bin/python -m cli world-models
environment/core/bin/python -m cli world-models --model ctrl_world
```

This read-only index covers every checkpoint profile in the model catalog. It joins
native replay configurations, input/output declarations, training domains, runtime
asset bindings, historical native acceptance receipts and exact VLA pair evidence.
It does not load a model or download files. Missing routes and missing published
receipts are explicit gaps, not inferred failures of the model.

| Route inventory | Checkpoint profiles |
| --- | --- |
| Shared replay configuration exists | GE-Sim 2.0 action mode, A2World pretrained, WorldGym, Ctrl-World, Cosmos DROID 20k EMA, Cosmos DROID joint best5600, DreamDojo GR1, IWS PushT, PlayWorld SVD, DreamTrue three-view, IRASim LanguageTable |
| Adapter exists; shared replay configuration still missing | τ₀-WM Simulator, GigaWorld-1, base Cosmos3-Nano |
| Compact native parity / long-replay acceptance receipts | Ctrl-World v2, Cosmos DROID 20k EMA and GE-Sim action mode; consult each receipt's scope and source match |

A replay configuration is not proof of a successful official example, full episode,
or VLA loop. Consult `native_acceptance_evidence` and `closed_loop_evidence` separately.
Older replay versions remain listed for reproducibility; select the protocol explicitly.
The base Cosmos checkpoint does not inherit the DROID checkpoint's validation.

For a first run, follow [installation](installation.md) and [model protocols](models.md),
then prepare the matching input and reference packets using the [workflow guide](workflows.md).
Replace `CASE` and `MY_NEW_RUN` in the emitted command templates with your prepared case
and a new output directory. Run admission before execution. The reference packet is
used only for scoring, never as the model's future observation. Build the offline
report using the emitted `report` command. These templates deliberately do not invent
an input converter for an unsupported native dataset.

### Independent validation stages

```bash
python -m cli world-models --table
python -m cli world-models --model ge_sim_v2
```

The matrix separately reports interface tests, real-weight execution, native replay,
official-entry parity and ground-truth comparison. A passed interface test cannot
upgrade the other stages. `not_recorded` means no compact published receipt, not a
failed model. `historical_pass_source_changed` keeps old evidence visible while
warning that the listed adapter source no longer matches. Evidence binds the exact
checkpoint profile; base and post-trained checkpoints do not inherit each other's results.

GE-Sim has an official action-mode entry comparison (25 + 25 + 6 actions, exact decoded
RGB and predicted-state equality), plus one complete 456-action replay. Its example
has no recorded future RGB, so the GT comparison remains unavailable. The first chunk
boundary in the long video has a pixel-change peak; a completed rollout is not a
smoothness or physics guarantee. τ₀-WM currently has interface-test evidence only.
The compact machine-readable source is `configs/world_model_acceptance_v1.json`.
