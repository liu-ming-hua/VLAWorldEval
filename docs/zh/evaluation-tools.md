# 一致性检查、动作分支与离线报告

[使用手册](README.md) · [English](../en/evaluation-tools.md)

选择[兼容的协议](compatibility.md)后使用以下工具。动作分支复用现有原生回放执行器，不另建推理链路。

## 选择检查内容

| 工具 | 回答的问题 | 不能证明什么 |
| --- | --- | --- |
| 原生一致性检查 | Adapter 是否复现捕获的原生推理路径？ | 物理正确性、所有权重版本的一致性 |
| 动作分支评测 | 同一初始观测下，不同动作如何改变预测？ | 没有执行真值的反事实预测是否正确 |
| 离线报告 | 哪些任务完成、失败，视频和指标在哪里？ | 跨训练域、跨协议的统一模型排名 |

## 1. 对照原生推理与 Adapter

准备[真实模型资源](first-rollout.md)，选择安装了相应模型依赖的解释器：

```bash
environment/core/bin/python scripts/acceptance/native_parity.py \
  --profile openvla_bridge --python environment/vla/openvla/bin/python \
  --gpu 0 --seed 0 --output results/parity/openvla

environment/core/bin/python scripts/acceptance/native_parity.py \
  --profile worldgym_bridge --python environment/world_models/worldgym/bin/python \
  --gpu 1 --seed 0 --output results/parity/worldgym
```

解释器路径以环境注册表和实际安装为准。原生与 Adapter 在独立进程中运行，使用相同权重锁、seed 和协议，
显存分配预算为 20 GiB。输入结构、dtype、数值必须完全一致；输出容差由
[配置文件](../../configs/native_parity_v1.json)预先确定。

**验证范围有边界：** OpenVLA 对照固定版本的 WorldGym OpenVLA 集成代码，使用观测钩子替代世界模型生成与云端评分；
这不是原始 OpenVLA 依赖环境的复现。WorldGym 对照包含已说明的 CPU 加载、本地 VAE 兼容补丁的固定运行时，
测试三轮自回归。其他模型尚未通过这个入口完成真实权重一致性验证。

独立捕获的数值 NPZ trace 可通过 `vlaworldeval parity-check --native ... --adapter ... --policy ... --output ...`
比较，格式见 [parity.py](../../evaluation/validation/parity.py)。两个任意 trace 相等不构成来源认证。

## 2. 同起点动作分支评测

示例使用 WorldGym 官方 Bridge episode 2，准备“真实动作”和“全零打包动作”。全零是数值干预，
不保证代表物理上的保持不动。脚本不下载数据，也不伪造零动作分支的真实未来。

```bash
environment/core/bin/python scripts/acceptance/branch_example.py \
  --output results/branch-inputs --gpu 0 --chunks 3
vlaworldeval branches --manifest results/branch-inputs/branches.json \
  --output results/branches --check-only
vlaworldeval branches --manifest results/branch-inputs/branches.json \
  --output results/branches --execute
```

版本 1 的 manifest 使用工作区相对路径和 SHA256 绑定数值动作回放配置、输入与参考文件，包含不重复的 seeds 和若干 families：

- family 包含 `id`、`initial_condition.kind`，后者为 `observation_only` 或 `simulator_snapshot`。
- 每个 family 至少两个不同动作分支，分支包含 `id`、`input: {path, sha256}` 和可选的 `reference`。
- 同 family 的初始视角、完整 reset/history、指令必须相同，动作的协议、chunk 数和时间轴必须一致。

若声明 simulator snapshot，必须提供带哈希的 `evidence`。框架记录调用方提供的证据，不自动恢复或认证模拟器。
真实未来只参与评分，不送入模型。每个分支/seed 使用独立 worker，成本包括模型加载；所有计划项、失败与未运行项
都会保留在 `summary.json` 中。

两个分支均有真实执行参考时，计算配对响应 MSE：
`mean(((预测A - 预测B) - (真值A - 真值B))²) / 255²`，排除初始帧。
预测来自解码的 MP4 RGB，参考为原始 RGB。它是图像空间诊断，不是接触、轨迹或任务成功指标。
相同 seed 不保证扩散噪声轨迹完全相同。缺少分支真值时标为 **unknown**，绝不当作零误差。
多个 seed 是模型重复采样，不是多条独立的物理轨迹。

## 3. 导出统一离线报告

```bash
vlaworldeval report --source results/branches/summary.json \
  --source results/evaluation.json --output results/report
python -m http.server 8080 --directory results/report
```

第二个 source 仅在保存过 `evaluate` 报告时使用，否则省略；也接受原生回放的 `summary.json`。
可以直接打开 `index.html`，或启动 HTTP 服务。远程服务器需 SSH 转发 8080 端口。
分享时复制整个报告文件夹，不需要外部 CDN。

报告包含多视角视频同步播放/跳转、逐帧误差图、参考可用性、成本、协议与错误，以及机器可读的 `report.json`。
原生输出文件复制前检查校验和，未运行的计划项仍保留在分母中。
所有输出使用新目录，不覆盖已有证据。视频看起来合理不代表模型可靠，详见[指标边界](reliability.md)。

[2026-10-31: OpenVLA / WorldGym parity receipts](../../configs/native_parity_evidence_v1.json).
