# 模型支持与验证证据

[使用手册](README.md) / 验证证据 · [English](../en/support.md)

## 分开回答五个问题

| 问题 | 所需证据 |
| --- | --- |
| 已注册吗？ | 有效的组合清单与运行绑定 |
| 环境能安装吗？ | 全新环境安装、依赖检查、实际版本记录 |
| 两个真实模型跑通过吗？ | 完成闭环、VLA 与 WM 都实际调用、保存产物通过检查、工作进程正常退出 |
| 与官方实现一致吗？ | 输入／转换核验，以及独立限定范围的数值一致性实验 |
| 能可靠评价策略吗？ | 合适的物理真值验证；支持表不认证这一点 |

执行 `environment/core/bin/python -m cli support` 获取机器可读记录。
README 表格由[版本化证据目录](../../configs/support_evidence_v1.json)生成。
目录记录报告和产物哈希、实际执行协议、模型路径、依赖版本与历史运行标识。
它是精简凭证，不包含原始实验视频和日志。

## 状态含义

- **`historical_pass`**：历史执行通过，当前仍保留的产物已复核；不代表当前代码或另一台机器通过。
- **`not_verified`**：该项能力、该精确组合版本没有合格证据。
- **`contract changed`**：同 ID 的执行协议发生变化。比较时补齐默认字段；纯描述性 metadata 变化不影响判断。
- **`other_contract_evidence`**：同名 adapter 的其他版本有历史证据；只能展示，不能沿用为新版本通过。

例如，Ctrl-World v1 的闭环记录不能认证现在的 v2 时间协议。
环境注册表里的 `ready` 仅是配置声明，不能替代全新安装凭证；权重路径存在也不代表校验过完整 checkpoint。
旧凭证会明确这些限制。

使用 `evaluate --check-only` 检查**当前机器**的资源、环境和数据；使用 `evaluate --execute` 产生新的运行证据。
未单独验证的官方数值一致性和物理可靠性仍为未验证。
由 action target 推导的 state 是假设，不是真实测得的未来机器人状态。

## 更新表格

完成一次公共框架真实运行后：

```bash
environment/core/bin/python scripts/acceptance/support_evidence.py \
  --report results/delivery/my_first_rollout/evaluate.stdout
environment/core/bin/python scripts/acceptance/support_evidence.py --check
```

导入器拒绝仅预检查、只有真实动作回放、未完成或产物不合格的记录，并保留精确组合版本。
两份 README 从同一数据生成；`--check` 只检查，不修改文件。
原始日志与视频留在 `results/`，不打包发布。历史记录不能证明当前源码；全新安装样例会额外记录实际安装 wheel 的 SHA256。

[全新安装真实样例](first-rollout.md) · [指标解释](reliability.md)

使用 `--fresh-report results/delivery/my_first_rollout/report.json` 附加全新安装凭证。CLI 会比较已测试 wheel 的框架源码哈希与当前源码；`matches_tested_wheel` 仅表示代码相同，不认证当前资源、依赖或硬件。

`official_input_alignment=declared_only_not_certified` 表示相机／动作协议已有声明，但尚未独立认证与官方输入完全对齐，也不表示数值一致。`last_tested_attempt` 优先显示最近导入的全新环境运行标识。

## 原生世界模型验收

以下证据与 VLA 组合认证分开，参见[复现命令](evaluation-tools.md)和
[带校验和的验收凭证](../../configs/world_model_acceptance_v1.json)。

| 模型／协议 | 已注册 | 真实原生回放 | 限定范围的原生入口一致性 | 长回放覆盖 |
| --- | --- | --- | --- | --- |
| Ctrl-World native v2 | 是 | 通过 | 两轮，模型条件与解码帧完全一致 | 2 条官方公开轨迹，26 次调用，各视角 33／72 帧 |
| Cosmos DROID 社区 20k-EMA | 是 | 通过 | 两轮，有效采样参数与解码帧完全一致 | 3 条开发轨迹，62 次调用，465／241／289 帧 |

一致性检查共享已加载的权重，不包含加载器独立认证。Cosmos 另外对全部 992 个动作与公开 DROID 数据集方法
进行转换比较，最大差异约 1.2e-7，在预先规定的绝对／相对 1e-6 容差内；这是数值一致，不是动作逐位相同。
真实未来图像只参与评分。

`support` 单独返回 `runtime_acceptance` 和 `current_adapter_sources_match`。
后者只比较凭证列出的 Adapter 源文件，不认证当前安装的资源或全部框架代码。
这些检查不升级任何 VLA 组合的认证状态，不证明物理动作跟随可靠，也不构成独立留出集效果排名。
其他已注册世界模型保留原有证据状态，不因此获得新的原生入口一致性认证。

## 查找世界模型回放入口

```bash
environment/core/bin/python -m cli world-models
environment/core/bin/python -m cli world-models --model ctrl_world
```

这个只读入口覆盖模型目录中的全部 checkpoint 配置，汇总回放配置、输入输出、训练域、
运行时资产绑定、历史原生验收记录和具体 VLA 组合证据。它不会加载模型或下载文件。
缺少运行入口、缺少公开验收记录都会明确标注；这不等于模型效果失败。

| 入口情况 | Checkpoint 配置 |
| --- | --- |
| 已有统一回放配置 | GE-Sim 2.0 action 模式、A2World pretrained、WorldGym、Ctrl-World、Cosmos DROID 20k EMA、Cosmos DROID joint best5600、DreamDojo GR1、IWS PushT、PlayWorld SVD、DreamTrue 三视角、IRASim LanguageTable |
| 已有适配器，尚缺统一回放配置 | τ₀-WM Simulator、GigaWorld-1、基础版 Cosmos3-Nano |
| 已有精简原生一致性／长回放验收记录 | Ctrl-World v2 和 Cosmos DROID 20k EMA；以记录中的适用范围、源码匹配状态为准 |

有回放配置不代表已通过官方示例、完整轨迹或 VLA 闭环验证。
分别查看 `native_acceptance_evidence` 和 `closed_loop_evidence`。
旧版回放配置保留用于复现，必须显式选择协议；基础 Cosmos 不能继承 DROID 权重的验证结果。

首次运行先按[安装指南](installation.md)和[模型协议](models.md)准备环境与权重，
再依照[工作流](workflows.md)准备匹配的输入和独立参考数据包。
将命令模板中的 `CASE`、`MY_NEW_RUN` 替换为已准备的数据目录和新结果目录，先检查准入，再执行推理。
参考视频只参与生成后的评分，不作为模型的未来观测输入。最后使用输出提示中的 `report` 命令生成离线报告。
对于尚未支持的数据格式，入口不会伪造数据转换命令。
