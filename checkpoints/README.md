# Checkpoint storage

Downloaded weights are local runtime artifacts and are intentionally ignored by
Git. Keep them under this directory so that adapters, setup scripts, and model
metadata use stable paths.

New action-world-model releases use `world_models/{dreamdojo,iws,playworld,dreamtrue,irasim}`,
with exact variant names preserved inside each directory. See the
[integration record](../docs/en/models.md) for immutable
revisions and actual download/inference status. Code revisions, base weights,
post-trained checkpoints and samples are separate identities; do not replace
one model variant in place or label a partial download complete.

```text
checkpoints/
├── vla/
│   ├── openvla-7b/
│   ├── spatialvla-4b-224-pt/
│   ├── octo-base-1.5/
│   ├── rt1-x/
│   ├── groot/{n1.5-3b,n1.7-3b}/
│   ├── openpi/{pi0-droid,pi05-droid}/
│   └── smolvla/{base,smolvlm2-500m-video-instruct}/
└── world_models/
    ├── worldgym/checkpoint/
    ├── ctrl_world/{svd,clip,ctrl_world}/
    ├── cosmos3_nano/Cosmos3-Nano/
    ├── cosmos3_nano_droid_fd/<revision>/
    ├── a2world/{base,*.pt}/
    ├── ge_sim_v2/checkpoints/<revision>/
    └── tau0_wm_simulator/{tau-0-wm,Wan2.2-TI2V-5B}/
```

GR00T N1.7's separately loaded `nvidia/Cosmos-Reason2-2B` backbone follows the
Hugging Face cache layout under the project-local
`.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/`. It is intentionally
not duplicated under `checkpoints/vla/groot/`.

The directory name must include the model family and, when multiple releases
exist, the checkpoint revision or protocol. Do not copy weights into
`third_party/`, `scripts/`, or the Python package. The exact checkpoint path
used by an adapter belongs in `configs/world_models/` or in the replay command.

The authoritative input/output and training-domain records are in
[`configs/model_catalog.json`](../configs/model_catalog.json).
