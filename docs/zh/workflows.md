# 运行、回放与评价

[手册首页](README.md) / 操作流程 · [English](../en/workflows.md)

## A · VLA × 世界模型闭环

1. 选择[显式组合](compatibility.md)，准备[环境与资产](installation.md)。
2. 使用 `configure` 创建个人配置，设置 GPU、预算和可选的 `--chunks N`。
3. 先 `evaluate --check-only`，检查通过后再 `evaluate --execute`。
4. 检查每个 job 的产物，按相机保存连续视频。

```bash
environment/core/bin/python -m cli configure \
  --config configs/experiments/openvla_worldgym_v1.json \
  --output configs/experiments/my_long_loop.json \
  --vla-gpus 0 --wm-gpus 1 --budget-gib 20 --vla-budget-gib 18 --chunks 100
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/my_long_loop.json --experiment loop_smoke --check-only
```

增加 chunk 不会让数据窗口无限延长；历史或动作不足时，准入会拒绝执行。
不能通过静默重复图像、注入真实未来等方式延长闭环。

## B · 真实动作回放，不调用 VLA

仍通过**同一套框架 Worker**执行 WM。数据准备阶段输出原生推理 packet 和独立的真实未来 reference。
真实未来图像只能用于推理后评分，不能进入生成过程。

```bash
environment/core/bin/python scripts/benchmarks/native_replay.py --help
environment/core/bin/python -m cli replay --help
```

完成原生 packet 准备后，例如：

```bash
environment/core/bin/python -m cli replay \
  --config configs/replay/worldgym_bridge_native_v1.json \
  --input results/my_case/input.npz --reference results/my_case/reference.npz \
  --output results/my_case/replay --admit-only

# 确认输入、配置与资源后，去掉 --admit-only 才会推理。
```

上述 packet 路径是示意，实际使用准备工具返回的路径。
按[模型原生协议](models.md)选择 model 与数据，不能对所有 WM 套用 DROID 转换。
若原生转换依赖 Torch、FK 或其他可选库，请在相应模型 / 数据环境运行准备入口。

## C · 冻结多样本评估

`native-study` 在推理前冻结样本身份、配置 / 输入 / reference 哈希与随机种子。
子集选择不能参考模型结果；计划、可用、完成和已评分数量应分别记录。

```bash
environment/core/bin/python -m cli native-study prepare --help
environment/core/bin/python -m cli native-study run --help
environment/core/bin/python -m cli native-study score --help
environment/core/bin/python -m cli replay-report --help
```

已有准备好的 study 时：

```bash
environment/core/bin/python -m cli native-study run \
  --directory results/my_study --exclusive-gpu
environment/core/bin/python -m cli native-study score \
  --directory results/my_study --output results/my_scoring --workers 2
```

评价写入独立版本目录，修改指标不需要重新加载 WM。
使用 `replay-report --directory RUN --plan PLAN --metrics SCORING --output NEW_REPORT`
显式选择评价版本。不得对重复样本挑选最好结果后合并。

## 资源与吞吐选项

跨停止队列与已迁回的结果恢复全量评估时，使用
\`native-study remainder --plan FULL_PLAN --directory NEW_REMAINDER --gpus 0 1 2 3\`，
并重复指定 \`--source-directory RUN_ROOT\`。它验证已完成产物哈希，冻结互不重叠的缺失样本计划，
不加载模型。评分分母仍须使用原始全量计划，不能把剩余队列单独称为完整测试集。

- `--exclusive-gpu` 是合作队列锁，不会清退其他人的进程。
- `--reuse-worker` 仅对明确声明支持 reset 的模型提供跨轨迹常驻。
- `--gpu` / `--gpus` 仅迁移允许的资源设置，不更改科学输入。
- `score --metric-gpus ... --metric-frame-batch-size N` 批量计算 LPIPS，不是生成模型 batch。
- Cosmos 原生批量推理有独立实测入口：
  [native_batch_benchmark](../../evaluation/execution/native_batch_benchmark.py) 与
  [native_batch_stream](../../evaluation/execution/native_batch_stream.py)，不能对任意 adapter 直接开启。

显存预算不保证吞吐。图像 FPS、动作频率、去噪步数与自回归 chunk 数不是同一个概念。
优化必须保留 checkpoint、动作转换、分辨率、种子与评价长度的可追溯性。

[查看产物](artifacts.md) · [解释指标](reliability.md)

## A2World：通过统一回放运行官方示例

pretrained DROID 示例包含三路初始图像和**一个 20×14 动作 chunk**，没有真实未来 RGB。
这条入口验证生成链路，不代表长时序效果或预测准确率。它复用统一的隔离 worker、
动作保存、各视角连续视频和结果摘要格式。

```bash
environment/core/bin/python -m cli prepare-replay \
  --config configs/replay/a2world_pretrained_native_v1.json \
  --dataset datasets/a2world/official/droid \
  --output results/a2world_case/input.npz --chunks 1
environment/core/bin/python -m cli replay \
  --config configs/replay/a2world_pretrained_native_v1.json \
  --input results/a2world_case/input.npz \
  --output results/a2world_case/replay --admit-only
```

准入通过后去掉 `--admit-only` 执行，不要伪造 `--reference`。
结果目录必须未被占用；在复制的配置中选择自己的 GPU。
也可用 `scripts/replay/a2world.py --output results/a2world_example --gpu 0 --prepare-only`
准备有效配置；去掉 `--prepare-only` 后同样使用统一回放执行。

获取 NVIDIA 基础 tokenizer 的访问权限后，使用 `scripts/setup/a2world.sh` 安装。
配置中的 `tokenizer_path` 必须对应
`base_checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth`。
基础目录存在不等于安装完整；缺少文件会在模型 worker 启动前拒绝准入。
镜像返回 403 时不能跳过这个依赖。该模型不输出机器人 state。

## GE-Sim 2.0：带标定的动作条件回放

安装后通过 `scripts/serve/ge_sim_v2.sh` 启动独立服务器，用 `CUDA_VISIBLE_DEVICES`
选择 GPU、`GE_SIM_PORT` 选择端口，并将复制的回放配置中 `kwargs.server_url` 改为相同本地地址。
统一 worker 是 CPU 客户端（`gpus: []`），它的显存预算**不限制**单独启动的 GPU 服务器。
运行结束后关闭自己启动的服务器。

```bash
environment/core/bin/python -m cli prepare-replay \
  --config configs/replay/ge_sim_v2_action_native_v1.json \
  --dataset datasets/ge_sim_v2/official/demo_000 \
  --output results/gesim_case/input.npz --chunks 2
environment/core/bin/python -m cli replay \
  --config configs/replay/ge_sim_v2_action_native_v1.json \
  --input results/gesim_case/input.npz --output results/gesim_case/replay
```

上述命令使用官方 456 个动作中的前 50 个，连续执行两个 25 帧 chunk，不压缩动作。
三路 384×512 标定图像和 16 维绝对关节指令保持官方排列，其余动作明确计数，不补齐、不重复。
示例不包含真实未来 RGB，不能据此计算预测精度。
`action` 模式由动作经 FK 生成轨迹图，假设躯干和头部静止；`episode` 模式则上传预录制的未来末端轨迹。
两者是不同实验，此处统一配置明确选择 `action`。
服务器的权重、配置和日志需与运行结果共同保留；仅凭客户端记录不能证明远端用了哪个 checkpoint。

### 自动管理服务器，执行 GE-Sim 完整轨迹

```bash
environment/core/bin/python scripts/replay/ge_sim_v2.py \
  --output results/gesim_full --gpu 0 --memory-gib 28 --chunks all
```

入口检查空闲显存、获取框架协作式 GPU 锁，启动独立本地端口的服务器，并设置服务器随机种子和
PyTorch 显存分配上限。数据准备后使用**统一原生回放**执行，生成离线报告；正常或异常退出时均清理
自己启动的服务器，不影响其他用户进程。服务器环境移除代理变量，模型离线加载；环境、权重和标定示例
需要提前准备。可用 `--server-config PATH` 指定并记录另一份服务器配置。

官方示例的 `all` 对应全部 456 个动作：18 个 25 步 chunk 加最后 6 步。
三路连续视频各 457 帧（含初始图像），16 FPS 下约 28.6 秒。
原生服务器内部会重复最后一个 action token，凑齐固定的 25-token 网络窗口，然后只返回请求的 6 帧。
框架提交和保存的仍然只有原始 456 个动作。短尾段必须是最后一轮；不能用服务器内部补齐的历史继续下一轮。

完整配置使用回放 schema **v3**：数值动作 chunk 列表，仅允许最后一段较短，每个动作对应一帧输出。
`actions.npy` 保存平铺的 `[456,16]` 动作，`action_offsets.npy` 保存各段边界。
V1 的固定长度、V2 的结构化条件协议保持原有行为。V3 当前仅支持独立视频回放，
不支持参考视频评分、真实图像重新注入或复用 worker 池。

结果包括服务器配置／源码／权重哈希、生命周期记录、`replay/summary.json`、各视角视频和
`report/index.html`。`boundary_diagnostics.json` 提供解码后帧间变化和近静止片段统计，
不能证明视角语义正确、操作成功或物理可靠性。
官方 action 模式首轮读取初始 RGB，此后从自己的生成帧历史采样上下文；FK 轨迹图假设躯干和头部静止。

固定的上游版本需要 `scripts/setup/patches/ge_sim_v2_finite_tail.patch` 兼容补丁，
将短尾段的轨迹图和相机条件补齐到网络窗口。安装脚本会幂等应用补丁，遇到与补丁冲突的源码改动则停止。
这里使用的是**上游代码加兼容补丁**，不是未经修改的官方一致性验证；运行记录中的源码哈希明确对应执行版本。

(ge-sim-action-mode-validation)=
## GE-Sim action 模式验证

官方 `examples/replay.py` 使用 `conditioning="episode"`，会读取记录的未来 EEF 轨迹。
框架动作回放使用 `conditioning="action"`，通过官方 FK 将关节命令转成轨迹图和腕部相机几何。
直接比较这两种模式，不等于验证适配器一致性。

```bash
environment/world_models/ge_sim_v2/bin/python scripts/acceptance/world_model_parity.py \
  --model ge_sim --input datasets/ge_sim_v2/official/demo_000 \
  --output results/gesim_parity --gpu 0 --memory-gib 28 --chunks 2 --tail-actions 6
```

此命令比较官方 `WorldModelEnv` 的 action 模式与框架适配器。
两次独立启动服务，固定相同权重、配置和随机种子，对照解码 RGB 与预测 state，包含 HTTP JPEG 传输。
使用 **56 个动作的前缀**（25 + 25 + 6），不是轨迹末尾的六个动作；完整轨迹覆盖由另一次全量回放验证。
临时比较数组自动删除，`report.json` 保留哈希、帧数、误差和服务生命周期记录。

条件检查会扰动初始帧之后全部记录的 state、EEF 与相机外参，保持动作和初始条件不变。
若 FK 轨迹图和相机条件不变，只能说明本次路线不依赖这些未来状态字段，
不能代替有物理真值的动作分支实验，也不保证任意上游版本均如此。

## 真实状态轨迹回放与动作敏感性对照

Ctrl-World 的原生 DROID 协议以**记录的绝对 EEF／夹爪状态轨迹**作为条件，
不是执行器原始控制命令。先验证官方样例，再解释一般策略输出与该输入空间的关系。

```bash
# 需要已安装模型与数据；不加载 VLA。
environment/world_models/ctrl_world/bin/python scripts/acceptance/prepare_world_model_cases.py \
  --model ctrl_world --output results/ctrl_cases

environment/core/bin/python scripts/acceptance/branch_example.py --model ctrl_world \
  --prepared results/ctrl_cases/cases/ctrl_released_long_ep199_seed0 \
  --output results/ctrl_controls --gpu 0 --seed 0

python -m cli branches --manifest results/ctrl_controls/branches.json \
  --output results/ctrl_controls/evaluation --check-only
python -m cli branches --manifest results/ctrl_controls/branches.json \
  --output results/ctrl_controls/evaluation --execute
python -m cli report --source results/ctrl_controls/evaluation/summary.json \
  --output results/ctrl_controls/report
```

对 `ctrl_released_long_ep18599_seed0` 使用新目录重复执行，保持模型配置不变。
两条都是官方公开样例，不是论文完整测试集，也不是独立留出集估计。

| 分支 | 干预 | 可支持的判断 |
| --- | --- | --- |
| `recorded` | 全部原生记录状态，保留窗口锚点与末尾处理 | 与实际记录未来的 RGB 保真度对照 |
| `hold_initial_pose` | 重复初始绝对末端姿态与夹爪值 | 零运动条件，**不是坐标归零** |
| `shuffle_future` | 固定种子打乱不重复的未来状态，保留初始条件与窗口共享锚点 | 时间顺序敏感性，轨迹可能不符合物理约束 |

只有 `recorded` 配有 reference。不能把原轨迹未来当作干预分支的真实物理结果。
框架将物理动作分支对照保留为 `unknown`，另以 `prediction_sensitivity` 展示同种子生成视频间的 MAE/MSE。
后者在 H264 解码 RGB 上计算；真实回放的参考指标在视频编码前的原始预测上计算。
相同种子本身不构成扩散噪声轨迹完全一致的认证。

报告提供同步视频、预测／真实未来并排视频、PSNR、SSIM、冻结初始帧基线、时间 RGB 变化误差、
运动诊断与分段边界变化。这些是图像和敏感性指标，不是接触准确性、任务成功率或策略排名有效性的证明。
