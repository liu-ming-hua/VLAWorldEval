# Outputs & result viewing

[Manual](README.md) / Artifacts · [简体中文](../zh/artifacts.md)

## Find the actual job output

`evaluate` returns a report containing per-job paths. Do not guess an attempt directory.
A policy-loop artifact typically includes:

```text
job/
├── rollout.mp4           # continuous overview
├── views/*.mp4           # continuous named camera videos
├── actions.npz          # native and converted actions
├── states.npz           # available robot inputs, masks and provenance
├── metadata.json        # shapes, timing, protocol and view mapping
└── status.json          # lifecycle / completion information
```

Exact files vary by mode. Native replay reports reference their own videos, summaries,
action evidence and optional paired metrics; use the returned paths rather than assuming the loop layout.
Missing state data is not silently promoted to a predicted physical state.

## Inspect without loading models

```bash
environment/core/bin/python -m cli results --root results/experiments
environment/core/bin/python -m cli inspect --artifact results/first_demo
```

`inspect` checks artifact structure and video diagnostics. It does not assign task success.
`regression-report --report REPORT.json` re-audits execution reports; it is not a scientific WM score.

For arrays, use NumPy's NPZ keys first, then inspect shape/dtype/finite values.
Do not assume every adapter uses the same array key or action width.

## Video on GitHub or a remote server

The README uses a GIF preview linked to MP4 because GitHub rendering of local HTML video tags is inconsistent.
The preview is downsampled for size; the MP4 preserves the saved rollout's 15 FPS.

For your own artifacts, open the returned MP4 locally or securely copy it to your computer.
A server path alone is not a browser URL. If serving a directory, bind to localhost and tunnel it;
do not expose the workspace, datasets or credentials on a public HTTP server.

## Retention

`results/` is ignored and excluded from public distributions.
`cli cleanup-results` previews retention; `--apply` archives eligible generated outputs recoverably.
It must not touch checkpoints, datasets or environments.
Public demo media are a small, explicitly allowlisted exception under `docs/assets/`, with provenance.

[Evaluation workflow](workflows.md) · [Troubleshooting](troubleshooting.md)
