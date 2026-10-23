# Interpret results responsibly

[Manual](README.md) / Evaluation boundaries · [简体中文](../zh/reliability.md)

## What the framework measures

| Evidence | What it supports | What it does not establish |
| --- | --- | --- |
| Validated contracts / successful execution | The declared integration ran and artifacts can be audited | Correct dynamics or task success |
| Recorded-action replay + paired real future | Prediction accuracy along the recorded action sequence | Accuracy for unexecuted action branches |
| Frozen-frame baseline | Whether prediction improves on copying the initial image | Reliable action causality |
| Action sensitivity / ablations | Whether output changes with conditions | Whether the change is physically correct |
| Domain-qualified motion / interaction diagnostics | Measured image-space tracks or events in the calibrated domain | Universal 3D contact or robot-state truth |
| Same-initial-state executed action branches | A basis for counterfactual action-following validation | Validity beyond the tested robot/task distribution |

A recorded trajectory contains real actions and their realized future. It does **not** contain
the physical future for every alternative action. Random/zero-action generated videos therefore
do not have physical branch ground truth unless those alternatives were actually executed.

## Reporting protocol

1. Pin source, checkpoint, native data split, camera/action conversion and inference settings.
2. Freeze selection before seeing output; identify episodes, windows, horizons and seeds separately.
3. Keep future reference outside inference. Exclude context and padded tails from scoring.
4. Record planned / available / completed / scored counts and missing-data reasons.
5. Group by domain, protocol, horizon and checkpoint. Never rank across incompatible conditions.
6. Add action-independent baselines and branch truth where available; assess uncertainty by episode.

## Metric definitions

| Metric | Direction | Interpretation |
| --- | --- | --- |
| MSE | ↓ | Pixel error against paired future; also compare with frozen first frame |
| PSNR | ↑ | Logarithmic image reconstruction measure; scale and pixel protocol must match |
| SSIM | ↑ | Structural image similarity, not action correctness |
| LPIPS | ↓ | Perceptual similarity; this implementation scores decoded video frames |
| Motion / event diagnostics | Protocol-specific | Domain-calibrated tracks/events; missing annotations are not zero error |

Raw-pixel MSE/PSNR/SSIM and decoded-video LPIPS do not have identical processing paths.
Do not claim exact paper reproduction without matching selection, model variant and all metric details.
A static scene can score well even when robot motion is wrong.

## README reference-table scope

| Entry | Sampling / aggregation boundary |
| --- | --- |
| Cosmos EEF FD | 400 unique DROID episodes; 90 horizon/protocol groups; table ranges are group means |
| IRASim | 400 unique Language-Table episodes; 26 horizon/protocol groups; ranges are group means |
| DreamDojo | 100 released GR1 in-lab windows; 48 pure-future frames |
| IWS | 10 PushT validation episodes; displayed 192-frame protocol aggregates three seeds |
| Ctrl-World | 97 overlapping short windows from two released episodes; four future intervals |
| WorldGym | One released Bridge test episode, three seeds; not full OXE validation |

No aggregate table above is a policy ranking or a claim that all paper test sets were reproduced.
The values are compact reference observations; new experiments should use frozen plans and versioned reports.

## Source of the compact observations

The public [reference manifest](../assets/reference-results.json) identifies local report versions,
hashes and displayed values without shipping the experimental result trees.
Those raw report paths are provenance identifiers, not files included with every source checkout.
Recompute results using [workflows](workflows.md) and licensed native assets.

[Model domains](models.md) · [Architecture](architecture.md)
