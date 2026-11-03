# 可复现的真实模型 rollout

[使用手册](README.md) / 首次真实运行 · [English](../en/first-rollout.md)

固定组合为 **OpenVLA-7B × WorldGym**，输入为官方 Bridge 测试 episode 2，指令为
“put the eggplant in the pot”。seed 固定为 0，执行 **20 次 VLA 与 20 次世界模型推理**。
这是工程验收，不是任务成功认证。按声明的 20 FPS，21 帧视频约 1.05 秒，刻意保持验收成本低；
它不代表完整任务演示。

## 资源要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Linux x86-64、Python 3.11、uv、FFmpeg、兼容的 NVIDIA 驱动 |
| GPU | 两张不同的 CUDA 卡，VLA 至少空余 18 GiB，WM 至少空余 20 GiB；启动时重新检查 |
| 权重和数据 | 约 25.2 GB，内容由 `configs/first_rollout_assets_v1.json` 的 SHA256 固定 |
| 磁盘 | 建议预留约 50 GB，包含资源、软件包下载和新环境；不能硬链接缓存时需要更多 |
| 访问 | WorldGym 权重来自 Google Drive；SD3 VAE 需满足原作者访问条款 |

## 1 · 显式准备资源

在源码目录执行：

```bash
bash scripts/setup/core.sh
bash scripts/setup/first_rollout_assets.sh
```

准备脚本关闭代理，PyPI 和 HF 使用可配置镜像。GitHub、Google Drive 仍从真实托管地址下载，
不会声称 HF 镜像能代替它们。VAE 如需授权，通过 `HF_TOKEN` 提供，不要写进配置文件。
已有资源必须通过内容校验才能复用；其他 checkpoint 应使用新的配置版本，不能静默替换锁文件。

## 2 · 一条命令完成安装与真实运行

选择两张可用卡，输出目录必须尚不存在：

```bash
environment/core/bin/python scripts/acceptance/first_rollout.py \
  --output results/delivery/my_first_rollout --vla-gpu 0 --wm-gpu 1
```

命令校验全部资源，构建当前 wheel，新建核心和模型环境，安装固定版本依赖，
从源码目录之外验证导入，再调用**安装后的 CLI 和已有公共执行器**。
VLA 与 WM 各自使用常驻进程；该组合依赖兼容，因此两个进程共享一个全新模型环境。
不会额外编写另一套 rollout 循环。

允许且明确记录软件包缓存复用；`--offline` 要求缓存完整，缺失时直接失败，不回退到旧环境。
这里验证的是全新环境安装，**不是从零重新下载所有权重**。模型推理关闭网络访问，
不隐式复用其他模型环境或用户 site-packages。

## 3 · 查看结果

| 输出目录内文件 | 内容 |
| --- | --- |
| `report.json` | 总状态、安装记录、资源锁和 wheel 哈希、失败原因 |
| `evaluate.stdout` | 公共框架执行和产物审计结果 |
| `inspect.stdout` | 新安装核心环境产生的产物检查 |
| `core_freeze.stdout`、`models_freeze.stdout` | 实际安装版本 |
| `suite.json`、`environments.json` | 推理配置及显式环境映射 |
| `rollouts/attempt_*/loop_long/episode_000002/seed_000/` | 连续视频、action、metadata、逐 chunk 记录 |

通过要求两个真实模型各调用 20 次、产物完整且检查成功。预检查成功、文件存在、模拟输入测试成功都不算真实运行通过。
失败会保留日志与部分结果；修正后换一个新的输出目录，不覆盖失败记录。已安装环境不可随意搬动路径。

## 常见问题

- **资源哈希不符**：检查下载来源和版本，不能为了通过而重建锁。
- **VAE 访问受限**：接受上游条款并获得授权，或在本地准备相同的合法资源。
- **离线安装缺包**：去掉 `--offline`，走镜像安装；代理仍保持关闭。
- **驱动库版本不一致**：修复宿主机驱动；验收脚本不会修改系统库。
- **显存不足**：换两张空余显存足够的卡。显存预算不代表模型效果保证。

[组合验证证据](support.md) · [更长实验](workflows.md)

新环境统一保存在 `environment/verification/first_rollout_<id>/`，具体位置写入报告；结果目录仅保存凭证与 rollout 产物，不混放虚拟环境。验收还会检查原生／转换后 action 均为有限值 `[20,1,10]`，以及视频确实为预期的21帧。

## 可选路线：不使用 VLA 的 GE-Sim 完整回放

使用官方 `demo_000` 标定样例与真实记录的 **16 维绝对关节命令**。
这是完整轨迹的工程示例，不是任务成功率评测。只提供初始 RGB，后续 RGB 来自模型；
假定头部与躯干静止。样例未附带可用于评分的真实未来 RGB。

```bash
# 显式准备：关闭代理，依赖与权重走镜像。
bash scripts/setup/core.sh
bash scripts/setup/ge_sim_v2.sh

# 不启动服务、不分配 GPU 显存：检查已有资产和完整输入。
environment/core/bin/python scripts/replay/ge_sim_v2.py \
  --check-only --output results/gesim_check --gpu 0 --memory-gib 28

# 使用新的输出目录，自动启动并结束自己创建的服务。
environment/core/bin/python scripts/replay/ge_sim_v2.py \
  --output results/gesim_full --gpu 0 --memory-gib 28 --chunks all

# 本地打开 report/index.html；远程使用时将 8080 端口转发到电脑。
environment/core/bin/python -m http.server 8080 --bind 127.0.0.1 \
  --directory results/gesim_full
```

通过 `--gpu` 选择有空余显存的物理卡；28 GiB 是已测试的显存分配上限，
不保证适用于不同驱动、权重或加速配置。
安装会将源码附带的样例链接到 `datasets/ge_sim_v2/official/demo_000`，不覆盖用户数据。
可用 `--episode PATH` 指定其他标定样例。检查失败须先修复；每次运行使用新输出目录。

| 结果 | 含义 |
| --- | --- |
| `replay/head.mp4`、`left_wrist.mp4`、`right_wrist.mp4` | 三路同步连续视频，本例为初始帧加 456 张未来帧 |
| `replay/actions.npy`、`action_offsets.npy` | 原样保存全部 456 个命令及 18×25 + 6 的分段边界 |
| `replay/summary.json`、`receipt.json` | 调用数、帧数、哈希、服务生命周期及验证限制 |
| `boundary_diagnostics.json` | 分段边界的像素变化，不是任务或物理评分 |
| `report/index.html` | 可移植回放查看页面 |

原生入口对照及其限制见[GE-Sim 验证](workflows.md)。
