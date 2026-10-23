<p align="center">
  <img src="docs/assets/banner.png" alt="VLAWorldEval — Policies meet world models." width="100%">
</p>

<p align="center">
  <a href="README_CN.md">简体中文</a> ·
  <a href="docs/en/README.md">User manual</a> ·
  <a href="#model-combinations">Model combinations</a> ·
  <a href="#native-domain-reference-results">Reference results</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Framework-MIT-38bdae?style=flat-square" alt="Framework: MIT"></a>
  <img src="https://img.shields.io/badge/Core-Python_3.11-3776AB?style=flat-square" alt="Core: Python 3.11">
  <img src="https://img.shields.io/badge/Runtimes-uv_isolated-7856FF?style=flat-square" alt="Isolated uv runtimes">
  <img src="https://img.shields.io/badge/Execution-Policy_loop_%2B_native_replay-137C99?style=flat-square" alt="Policy loop and native replay">
</p>

# VLAWorldEval

**One framework to connect robot policies, action-conditioned world models, and reproducible evaluation.**

Run a VLA against generated observations, or replay recorded actions without a policy.
Keep model dependencies isolated, action semantics explicit, and videos, arrays and provenance together.
Designed for robotic-arm and gripper workflows—not an unrestricted Cartesian product of models.

**Our vision:** As world models become more faithful to robot actions, make reproducible, scalable VLA closed-loop evaluation a practical complement to real-world testing.

> [!IMPORTANT]
> A world model can produce plausible video while ignoring actions or hallucinating interactions.
> A runnable adapter is not evidence of physical accuracy, robot task success, or valid VLA rankings.
> Validate the checkpoint in its own action and observation domain. [Interpretation guide →](docs/en/reliability.md)

## See it in action

<p align="center">
  <a href="docs/assets/demos/cosmos-droid-task.mp4">
    <img src="docs/assets/demos/cosmos-droid-task.gif" width="840" alt="64.1-second continuous task replay: generated future on the left, recorded future on the right">
  </a>
</p>

<p align="center">
  <b>Task: Put the bottles and orange plushie onto the white plate</b><br>
  Cosmos DROID FD · 64.1 s · 60 autoregressive chunks · original 15 FPS<br>
  <sub>Left: generated future. Right: recorded execution. No slow motion, episode splicing or repeated playback.<br>
  Recorded-action WM replay, not a successful VLA rollout. Click for the full continuous MP4.
  <a href="docs/assets/provenance.json">Media provenance</a></sub>
</p>

<details>
<summary>Long-horizon replay — 2 min 19 s, 130 autoregressive chunks</summary>

[![138.7-second long-horizon replay, click for the complete video](docs/assets/demos/cosmos-droid-long.png)](docs/assets/demos/cosmos-droid-long.mp4)

Task: “Move the blocks to the left and then straighten the cloth.”
The complete saved 138.7-second replay runs at its original 15 FPS.
Left: generated future; right: recorded future. Long-horizon errors are kept visible, not edited out.

</details>

<details>
<summary>Also watch a VLA closed loop — GR00T N1.5 × Cosmos, 21.4 s</summary>

[![GR00T N1.5 and Cosmos continuous VLA loop](docs/assets/demos/groot-cosmos.gif)](docs/assets/demos/groot-cosmos.mp4)

Instruction: “Pour the contents of the yellow cup into the bowl.” Generated observations
are fed back to the policy. This is a longer continuous loop, not a verified task-success demonstration.

</details>

## How it works

<p align="center">
  <a href="docs/assets/framework.svg">
    <img src="docs/assets/framework.png" width="100%" alt="VLAWorldEval framework: observation, VLA, world model, evaluation and feedback">
  </a>
</p>

| Building block | What you get |
| --- | --- |
| Model adapters | Separate VLA and WM interfaces; reuse official inference implementations |
| Protocol contracts | Explicit camera mapping, action representation, normalization, history and FPS |
| Isolated workers | Persistent model processes in dedicated uv environments; declared multi-GPU support |
| Two evaluation routes | Autoregressive VLA × WM loops and recorded-action replay with paired future comparison |
| Reproducible artifacts | Continuous per-view video, native/converted actions, timing, failure records and checksums |
| Extensible evaluation | Visual metrics, frozen-frame controls, coverage reports and domain-qualified motion diagnostics |

## Quick start

Linux · Python 3.11 · [uv](https://docs.astral.sh/uv/) · FFmpeg.
The core does **not** install Torch, JAX, TensorFlow or model weights.

```bash
bash scripts/setup/core.sh
environment/core/bin/python -m cli demo --output results/first_demo --chunks 30
environment/core/bin/python -m cli inspect --artifact results/first_demo
```

The demo is a synthetic integration fixture. For pretrained models, prepare the
[model runtimes and assets](docs/en/installation.md), then check a configuration before inference:

```bash
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/openvla_worldgym_v1.json \
  --experiment loop_smoke --check-only

# After selecting GPUs and budgets with "configure":
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/openvla_worldgym_v1.json \
  --experiment loop_smoke --execute
```

Setup/download helpers disable proxies and use configurable PyPI/HF mirrors.
Official GitHub, Google Drive and GCS assets retain their actual hosting and access terms.

[First real rollout →](docs/en/quickstart.md) · [Recorded-action evaluation →](docs/en/workflows.md)

## Model combinations

The runtime registry contains **16 explicit pair configurations**. B = baseline route;
E = experimental, embodiment-specific bridge; — = no registered pair.
These describe integration scope, not installation readiness or dynamics quality.

| VLA / checkpoint | WorldGym · Bridge | Cosmos · DROID EEF FD | Ctrl-World · DROID |
| --- | :---: | :---: | :---: |
| OpenVLA | B | — | — |
| SpatialVLA | B | — | — |
| Octo | B | — | — |
| RT-1-X | B | — | — |
| GR00T N1.5 · OXE-DROID | — | B | E |
| GR00T N1.7 · base | — | E | E |
| GR00T N1.7 · DROID | — | E | E |
| π0 · DROID | — | E | E |
| π0.5 · DROID | — | E | E |
| π0-FAST · DROID | — | E | E |

State-dependent policies still require valid robot inputs; video-only WMs do not supply them.
DROID routes use explicit, assumption-bound conversions rather than measured future robot state.
The Cosmos EEF FD route uses the **community 20k-EMA checkpoint**, not the generic Nano base model.

Additional catalog entries include SmolVLA, Cosmos Policy, DreamZero, A2World, GE-Sim 2.0,
τ₀-WM, DreamDojo, IWS, PlayWorld, DreamTrue, IRASim and Giga-World-1.
Their native protocols and readiness differ; catalog presence does not enable arbitrary pairing.

[Input/output and training domains →](docs/en/models.md) · [Pair selection →](docs/en/compatibility.md)

## Native-domain reference results

Paired prediction metrics from existing runs; **not a cross-domain leaderboard**.
Lower LPIPS and higher SSIM indicate closer images, not validated physics.

| Released checkpoint | Evaluation domain | Unique episodes | Future frames | SSIM ↑ | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Cosmos DROID FD · community 20k EMA | DROID | 400 | 16–2704 | 0.301–0.951* | 0.151–0.723* |
| IRASim · Language-Table | Language-Table | 400 | 15–49 | 0.826–0.869* | 0.151–0.191* |
| DreamDojo · 2B GR1 | GR1 in-lab | 100 | 48 | 0.733 | 0.239 |
| IWS · PushT camera 1 | Real ALOHA PushT | 10 | 192 | 0.869 | 0.121 |
| Ctrl-World | Released DROID subset | 2 | 4 | 0.776 | 0.196 |
| WorldGym | Released Bridge test example | 1 | 21 | 0.833 | 0.130 |

*Ranges are **per-horizon group means**, not confidence intervals or cherry-picked individual cases.
Ctrl-World's short windows reuse two episodes; WorldGym uses three seeds on one episode.
Different domains, horizons and sample coverage prevent an overall ranking or a paper-reproduction claim.

[Metric definitions and scope →](docs/en/reliability.md)

## Documentation

| Start here | Go deeper |
| --- | --- |
| [Quick start](docs/en/quickstart.md) | [Model contracts and training domains](docs/en/models.md) |
| [Installation and mirrors](docs/en/installation.md) | [Architecture](docs/en/architecture.md) |
| [Run and score](docs/en/workflows.md) | [Add an adapter](docs/en/extending.md) |
| [Outputs and viewing](docs/en/artifacts.md) | [Troubleshooting](docs/en/troubleshooting.md) |
| [中文使用手册](docs/zh/README.md) | [Release and license boundaries](docs/en/release.md) |

## Contributing & license

Small adapters, explicit protocols, shared execution. Read [CONTRIBUTING.md](CONTRIBUTING.md)
before adding a model. Original framework code is [MIT](LICENSE); upstream implementations,
weights, datasets and derived demo media retain their applicable terms. See [license scope](LICENSE_SCOPE.md).

## Acknowledgements

We thank the authors and maintainers of the open models, datasets and tools that make this framework possible:

- **Policies:** [OpenVLA](https://github.com/openvla/openvla), [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA),
  [Octo](https://github.com/octo-models/octo), [Open X-Embodiment / RT-1-X](https://github.com/google-deepmind/open_x_embodiment),
  [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [OpenPI](https://github.com/Physical-Intelligence/openpi),
  [LeRobot / SmolVLA](https://github.com/huggingface/lerobot), [Cosmos Policy](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
  and [DreamZero](https://huggingface.co/GEAR-Dreams/DreamZero-DROID).
- **World models:** [WorldGym](https://github.com/world-model-eval/world-model-eval),
  [Cosmos](https://github.com/NVIDIA/cosmos), [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World),
  [A2World](https://github.com/LogosRoboticsGroup/A2World), [GE-Sim 2.0](https://github.com/AgibotTech/GE-Sim-V2),
  [τ₀-WM](https://github.com/sii-research/tau-0-wm), [Giga-World-1](https://github.com/open-gigaai/giga-world-1),
  [IWS](https://github.com/WangYixuan12/interactive_world_sim), [DreamDojo](https://github.com/NVIDIA/DreamDojo),
  [IRASim](https://github.com/bytedance/IRASim), [DreamTrue](https://github.com/brave-eai/DreamTrue)
  and [PlayWorld](https://github.com/irom-princeton/open-world).
- **Data and evaluation:** [DROID](https://droid-dataset.github.io/),
  [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment),
  [EWMBench](https://github.com/AgibotTech/EWMBench) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity).

The featured DROID FD weights are the [community 20k-EMA release](https://huggingface.co/jere-mybao/cosmos3-nano-droid-forward-dynamics).
Upstream work retains its own licenses and attribution; acknowledgement does not imply endorsement.
