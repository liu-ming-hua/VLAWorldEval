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
