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
