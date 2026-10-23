# Pair compatibility

[Manual](README.md) / Choose a pair · [简体中文](../zh/compatibility.md)

## Start from an explicit configuration

There are 16 registered routes. B denotes a catalog baseline route; E denotes an experimental bridge.
These are not quality rankings or guarantees that assets are installed on your machine.
Multi-GPU suite variants do not count as additional model combinations.

| Pair | Tier | Configuration |
| --- | :---: | --- |
| OpenVLA × WorldGym | B | [`openvla_worldgym_v1.json`](../../configs/experiments/openvla_worldgym_v1.json) |
| SpatialVLA × WorldGym | B | [`spatialvla_worldgym_v1.json`](../../configs/experiments/spatialvla_worldgym_v1.json) |
| Octo × WorldGym | B | [`octo_worldgym_v1.json`](../../configs/experiments/octo_worldgym_v1.json) |
| RT-1-X × WorldGym | B | [`rt1_x_worldgym_v1.json`](../../configs/experiments/rt1_x_worldgym_v1.json) |
| GR00T N1.5 × Cosmos EEF FD | B | [`engineering_groot_cosmos_v1.json`](../../configs/experiments/engineering_groot_cosmos_v1.json) |
| GR00T N1.5 × Ctrl | E | [`groot_n15_ctrl_world_droid_v1.json`](../../configs/experiments/groot_n15_ctrl_world_droid_v1.json) |
| GR00T N1.7 base × Cosmos EEF FD | E | [`groot_n17_cosmos_droid_3gpu_v1.json`](../../configs/experiments/groot_n17_cosmos_droid_3gpu_v1.json) |
| GR00T N1.7 base × Ctrl | E | [`groot_n17_ctrl_world_droid_v1.json`](../../configs/experiments/groot_n17_ctrl_world_droid_v1.json) |
| GR00T N1.7-DROID × Cosmos EEF FD | E | [`groot_n17_droid_cosmos_v1.json`](../../configs/experiments/groot_n17_droid_cosmos_v1.json) |
| GR00T N1.7-DROID × Ctrl | E | [`groot_n17_droid_ctrl_v1.json`](../../configs/experiments/groot_n17_droid_ctrl_v1.json) |
| π0 DROID × Cosmos EEF FD | E | [`pi0_cosmos_droid_v1.json`](../../configs/experiments/pi0_cosmos_droid_v1.json) |
| π0 DROID × Ctrl | E | [`pi0_ctrl_world_droid_v1.json`](../../configs/experiments/pi0_ctrl_world_droid_v1.json) |
| π0.5 DROID × Cosmos EEF FD | E | [`pi05_cosmos_droid_v1.json`](../../configs/experiments/pi05_cosmos_droid_v1.json) |
| π0.5 DROID × Ctrl | E | [`pi05_ctrl_world_droid_v1.json`](../../configs/experiments/pi05_ctrl_world_droid_v1.json) |
| π0-FAST DROID × Cosmos EEF FD | E | [`pi0_fast_droid_cosmos_v1.json`](../../configs/experiments/pi0_fast_droid_cosmos_v1.json) |
| π0-FAST DROID × Ctrl | E | [`pi0_fast_droid_ctrl_v1.json`](../../configs/experiments/pi0_fast_droid_ctrl_v1.json) |

## Match every boundary

| Condition | What to verify |
| --- | --- |
| Embodiment / action | EEF vs joints; delta vs absolute; coordinates; rotations; gripper sign and units |
| Views | Camera identity, order, resolution and crop; wrist and external cameras are not interchangeable |
| Time | Action/video rates, chunks, context, duplicate initial frames and tail padding |
| Normalization | Checkpoint-specific statistics; never normalize an already converted action twice |
| Extra policy inputs | Required state/history must be supplied explicitly; video output does not imply robot-state output |
| Data domain | Native training distribution and checkpoint-specific conditions |

The Bridge path has an explicit 7D-to-WorldGym-10D mapping; it is not a universal padding rule.
DROID paths include FK, temporal alignment and absolute/relative conversion.
Their non-measured recurrent robot inputs remain assumption-bound and provenance-labelled;
they are not WM-predicted states or real execution results.

## Read support tiers correctly

- **supported**: scoped to named baseline pairs, not scientific validity.
- **native_replay**: native-condition replay, not an arbitrary policy pairing.
- **experimental**: startup, specialized bridges or incomplete-resource paths.
- **incompatible**: incompatible with this robot-action loop, not necessarily invalid native inference.

Use `cli models` for the catalog, `cli capabilities` for explicit pairs and
`cli runtimes` for constructors/loaders. Registering a model does not automatically enable a pair.

[Model reference](models.md) · [Run a workflow](workflows.md) · [Add an adapter](extending.md)
