# Model reference: inputs, outputs & training domains

[Manual](README.md) / Model reference · [简体中文](../zh/models.md)

These are implemented adapter/checkpoint contracts, not promises about every model sharing a name.
`T` is time and `V` is views. Images cross the common boundary as uint8 RGB HWC;
model-specific tensor conversion happens inside adapters.
Equal action width does not establish equal coordinates, control semantics, statistics or embodiment.

## VLA interfaces

| Model | Image / instruction input | Adapter action output | Robot state input | Training / usage domain |
| --- | --- | --- | --- | --- |
| OpenVLA | RGB + text | [T,10] ← native 7D | — | OXE / Bridge |
| SpatialVLA | RGB + text | [T,10] ← native 7D | — | OXE + RH20T |
| Octo | RGB history + text | [T,10] ← native 7D | — | OXE / Bridge |
| RT-1-X | RGB 256×320 + USE512/text | [1,10] ← native action dict | — | OXE / Bridge |
| GR00T N1.5 | 3 DROID RGB + text | [16,7] relative EEF + gripper | EEF pose + gripper | OXE-DROID |
| GR00T N1.7 base | 2 views × history/current + text | [40,17] decoded absolute targets | EEF9 + joints7 + grip1 | Generalist / OXE-DROID |
| GR00T N1.7-DROID | 2 current views + text | [40,17] decoded absolute targets | EEF9 + joints7 + grip1 | DROID specialist |
| π0 DROID | Exterior + wrist + text | [10,8] joint velocity + gripper | joints7 + grip1 | DROID |
| π0.5 DROID | Exterior + wrist + text | [15,8] joint velocity + gripper | joints7 + grip1 | DROID / RoboArena |
| π0-FAST DROID | Exterior + wrist + text | [10,8] joint velocity + gripper | joints7 + grip1 | DROID |
| SmolVLA base | Named RGB views + text | [50,6] | 6D | Community-data base; robot fine-tune needed |
| Cosmos DROID Policy | 3 DROID RGB + text | [32,8] absolute joints / grip | joints7 + grip1 | DROID post-train; no registered WM pair |
| DreamZero DROID | 3 views; first1 then4 frames + text | [24,8] absolute joints / grip | joints7 + grip1 | DROID; native GPU readiness pending |

## World-model interfaces

Most entries produce video only. GE-Sim's native Pose-Expert is a model-specific exception,
not a promise of generic robot-state prediction.
Ctrl's inclusive window contains context; five returned frames are not five new future intervals.

| Checkpoint | Observation conditions | Action conditions | Output | Training / input domain |
| --- | --- | --- | --- | --- |
| WorldGym | 1 RGB 256² | [T,10]; Bridge 7D core | RGB video; no state | Nine OXE datasets; Bridge V2 / RT-1 named; full mixture unpublished |
| Ctrl-World | 3 RGB 192×320, sparse history | [5,7] normalized absolute EEF | Inclusive 5-frame × 3-view video; no state | DROID |
| Cosmos EEF FD · 20k EMA | 3-view canvas 540×640 → 528×640 | [16,10] relative OpenCV EEF rot6D + flipped grip | 16 future frames at 15 FPS; no state | Community DROID post-train; not NVIDIA base weights |
| Cosmos joint FD · best5600 | 3-view canvas + current state8 | [32,8] absolute joint / grip | 32 future frames; no state head | Separate DROID joint export |
| Cosmos3-Nano base | RGB + domain-dependent conditions | Domain-dependent; not EEF FD post-train | Video; no state in FD interface | Generic pretrained base |
| A2World | 2 or 3 synchronized RGB views | [20,14] native embodiment | Multi-view RGB 256²; no state | 2.1M+ trajectories / 20+ embodiments |
| GE-Sim 2.0 | 3 calibrated G01 views + FK assets | [T,16] absolute joints | 3-view RGB + Pose-Expert [T,16] | G01 / OmniPicker release |
| τ₀-WM Simulator | Views [V,3,192,256] | [T,20] relative bimanual EEF API; 26D internally | Terminal views / video; no state | Released whole-body simulator; not TacoPlay 7D |
| DreamDojo 2B GR1 | GR1 initial RGB + native conditioning | 29 normalized joints encoded into 384 slots | 49-frame native window incl. context; no state | Human-video pretrain + GR1 post-train |
| IWS PushT cam1 | RGB 128² + initial world XY | [T,4] absolute world XY via official FK | RGB 128²; no state | Task-specific real ALOHA PushT |
| PlayWorld SVD | Task-native images / conditions | Task-specific native encoding | Video; no state | DROID initialization + robot-play post-train |
| DreamTrue | Calibrated multi-view rendered geometry | Structured geometry, not a raw action vector | Multi-view RGB; no state | AgiBot / DROID / RoboMIND / RoboTwin |
| IRASim Language-Table | Native initial RGB / latent | Language-Table planar XY | RGB video; no state | Language-Table Frame-Ada public checkpoint |
| Giga-World-1 FunCtrl | Initial image + Plücker/Ray-Map control video | Control video, not robot action tensor | RGB video; no state | Heterogeneous video / robot / ego corpus; Toydata is only a demo |

## Dataset selection rules

- WorldGym's released Bridge example contains 9 train + 1 test items, not the full OXE training mixture or paper test set.
- DROID post-trains require DROID cameras, embodiment and action semantics; an arbitrary arm video is not equivalent.
- IWS uses observed joints passed through official FK to world XY; stored commands are not observed positions.
- Giga-World-1 FunCtrl does not accept raw robot actions; Toydata does not validate VLA action-conditioned loops.
- τ₀ TacoPlay 7D data is not compatible with the released 20D API checkpoint by zero padding.
- Keep N1.7 base / DROID, Cosmos EEF / joint FD, and PlayWorld SVD / AR checkpoint versions distinct.

## Runtime keys & official sources

The [catalog](../../configs/model_catalog.json) is the machine-readable interface reference;
the [runtime registry](../../configs/runtime_registry.json) defines constructors.
Use the [installation guide](installation.md) for assets and [workflows](workflows.md) for native evaluation.

| ID | Environment key | Source |
| --- | --- | --- |
| `openvla` | `vla.openvla` | [Upstream](https://openvla.github.io/) |
| `spatialvla` | `vla.spatialvla` | [Upstream](https://github.com/SpatialVLA/SpatialVLA) |
| `octo` | `vla.octo` | [Upstream](https://github.com/octo-models/octo) |
| `rt1_x` | `vla.rt1` | [Upstream](https://github.com/google-deepmind/open_x_embodiment) |
| `groot_n15` | `vla.groot_n15` | [Upstream](https://github.com/NVIDIA/Isaac-GR00T) |
| `groot_n17` | `vla.groot_n17` | [Upstream](https://github.com/NVIDIA/Isaac-GR00T) |
| `pi0_droid` | `vla.openpi` | [Upstream](https://github.com/Physical-Intelligence/openpi) |
| `pi05_droid` | `vla.openpi` | [Upstream](https://github.com/Physical-Intelligence/openpi) |
| `smolvla` | `vla.smolvla` | [Upstream](https://github.com/huggingface/lerobot) |
| `groot_n17_droid` | `vla.groot_n17` | [Upstream](https://github.com/NVIDIA/Isaac-GR00T) |
| `pi0_fast_droid` | `vla.openpi` | [Upstream](https://github.com/Physical-Intelligence/openpi) |
| `cosmos_policy_droid` | `vla.cosmos_policy_droid` | [Upstream](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) |
| `dreamzero_droid` | `vla.dreamzero` | [Upstream](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) |
| `worldgym` | `world_models.worldgym` | [Upstream](https://arxiv.org/html/2506.00613v3) |
| `ctrl_world` | `world_models.ctrl_world` | [Upstream](https://github.com/Robert-gyj/Ctrl-World) |
| `cosmos3_nano` | `world_models.cosmos3_nano` | [Upstream](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/action) |
| `cosmos3_nano_droid_fd_20k_ema` | `world_models.cosmos3_nano_droid_fd` | [Upstream](https://huggingface.co/jere-mybao/cosmos3-nano-droid-forward-dynamics) |
| `cosmos3_droid_joint_fd_best5600` | `world_models.cosmos3_droid_joint_fd` | [Upstream](https://huggingface.co/anshks/cosmos3-nano-droid-fd-wm) |
| `a2world` | `world_models.a2world` | [Upstream](https://github.com/LogosRoboticsGroup/A2World) |
| `ge_sim_v2` | `world_models.ge_sim_v2` | [Upstream](https://github.com/AgibotTech/GE-Sim-V2) |
| `tau0_wm_simulator` | `world_models.tau0_wm_simulator` | [Upstream](https://github.com/sii-research/tau-0-wm) |
| `dreamdojo_2b_gr1` | `world_models.dreamdojo` | [Upstream](https://github.com/NVIDIA/DreamDojo) |
| `iws_pusht_cam1` | `world_models.iws` | [Upstream](https://github.com/WangYixuan12/interactive_world_sim) |
| `playworld_svd` | `world_models.playworld` | [Upstream](https://github.com/irom-princeton/open-world) |
| `dreamtrue_threeview_v1` | `world_models.dreamtrue` | [Upstream](https://github.com/brave-eai/DreamTrue) |
| `irasim_languagetable_frame_ada` | `world_models.irasim` | [Upstream](https://github.com/bytedance/IRASim) |
| `gigaworld1` | `world_models.gigaworld1` | [Upstream](https://github.com/open-gigaai/giga-world-1) |

[Compatibility](compatibility.md) · [Metric interpretation](reliability.md)
