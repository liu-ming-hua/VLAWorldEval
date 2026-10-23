# Dataset inventory

Native examples and test inputs belong here. Payloads are not part of the framework source release;
small `manifest.json` files describe origin, revision, layout and intended use.

## Select by checkpoint, not by convenience

| Model family | Canonical input location | Native-protocol requirement |
| --- | --- | --- |
| WorldGym | `worldgym/official/bridge` | Released Bridge MP4/NPZ; 9 train + 1 test examples, not full OXE validation |
| Ctrl-World | `ctrl_world/official/droid_subset` | Three DROID cameras; absolute EEF targets and inclusive windows |
| Cosmos DROID FD | `cosmos3_droid/` | Version-specific DROID conversion and holdout selection |
| A2World | `a2world/official/droid` | Official pretrained PNG/action bundle |
| GE-Sim 2.0 | `ge_sim_v2/official/demo_000` | G01 / OmniPicker calibration and FK bundle |
| τ₀-WM | `tau0_wm/` | Released simulator needs 20D API; TacoPlay 7D sample is not compatible |
| Giga-World-1 | `gigaworld1/official/toydata` | Control-video inference; no raw robot-action bridge |
| IWS | `iws/` | Real ALOHA PushT observations and official FK |
| DreamDojo | `dreamdojo/` | GR1 post-train evaluation conditions |
| IRASim | `irasim/` | Language-Table initial latents, RGB and XY annotations |
| DreamTrue | `dreamtrue/` | Calibrated rendered geometry and matching reference |
| PlayWorld | `playworld/` | Public initialization samples are not recorded-future trajectories |

A path in this table does not guarantee a complete local download.
Use manifests, checksums and the actual inventory rather than historical status notes.

```bash
environment/core/bin/python scripts/checks/datasets.py
```

The legacy `gigaworld1_toy/manifest.json` is retained as metadata only.
New integrations use `gigaworld1/manifest.json`.

For evaluation, distinguish training membership, released examples, frozen heldout selection,
downloaded payload, executed cases and scored cases. Missing assets never become synthetic ground truth.
Reference future is prepared separately and must not enter inference.

[Model domains](../docs/en/models.md) · [Evaluation workflow](../docs/en/workflows.md) · [中文手册](../docs/zh/README.md)
