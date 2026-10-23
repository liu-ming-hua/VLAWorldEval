# Troubleshooting

[Manual](README.md) / Troubleshooting · [简体中文](../zh/troubleshooting.md)

| Symptom | First checks | Avoid |
| --- | --- | --- |
| Dependency resolution conflict | Separate model runtimes; inspect environment key and pinned requirements | Installing every VLA extra together |
| CUDA driver mismatch | Compare driver with the model runtime's Torch/CUDA build | Updating shared system drivers blindly |
| Download timeout / no route | Confirm proxies are cleared, host is reachable, mirror and authentication apply to that host | Assuming HF mirror covers Drive/GCS/GitHub |
| Missing runtime code | Run the model's setup entry; check pinned source identity | Copying another model's networks into the core |
| GPU admission failure | Inspect physical GPU IDs, free memory, declared distributed support and both worker budgets | Treating an allocator cap as compression |
| Missing state/history/view | Read the native contract and selected pair manifest | Inventing measured state or duplicating camera identities |
| Frozen / corrupt video | Verify RGB layout, normalization, action coordinates/rate, context and per-camera VAE handling | Immediately blaming checkpoint quality or amplifying actions |
| Output already exists | Select a new versioned output directory | Deleting evidence to retry |
| Slow inference | Separate loading time, denoising cost, AR dependence and scoring; qualify worker reuse/batching | Reducing scientific settings without recording the change |
| MP4 will not play | Inspect with ffprobe, check codec/pixel format, copy locally | Expecting a remote filesystem path to be a URL |

## Useful read-only checks

```bash
environment/core/bin/python -m cli validate
environment/core/bin/python -m cli environments
environment/core/bin/python -m cli env-audit --environment vla.openvla --dependencies
environment/core/bin/python -m cli inspect --artifact results/first_demo
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/openvla_worldgym_v1.json --experiment loop_smoke --check-only
```

CLI errors are structured; `--debug` before the command exposes the traceback when needed.
Keep the config, exact error, model revision and environment audit when reporting an issue.
Never include tokens, private keys, private dataset frames or another user's paths.

[Installation](installation.md) · [Model protocols](models.md) · [Output viewing](artifacts.md)
