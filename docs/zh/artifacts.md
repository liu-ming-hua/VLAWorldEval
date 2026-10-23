# 输出格式与结果查看

[手册首页](README.md) / 产物 · [English](../en/artifacts.md)

## 先找到实际 job 路径

`evaluate` 返回报告内含每个 job 的产物位置，不要凭日期猜 attempt 目录。
VLA 闭环通常包含：

```text
job/
├── rollout.mp4           # 连续总览
├── views/*.mp4           # 具名相机的连续视频
├── actions.npz          # 原生动作与桥接后的动作
├── states.npz           # 实际存在的状态输入、mask 与来源
├── metadata.json        # 维度、耗时、协议与视角映射
└── status.json          # 生命周期与完成状态
```

具体文件随模式而不同。原生回放报告引用自己的视频、summary、动作证据与可选配对指标；
请以返回路径为准，不假设它完全沿用闭环目录结构。
没有状态数据不意味着可以将其视为物理预测状态。

## 不加载模型即可检查

```bash
environment/core/bin/python -m cli results --root results/experiments
environment/core/bin/python -m cli inspect --artifact results/first_demo
```

`inspect` 检查结构和视频诊断，不判定任务成功。
`regression-report --report REPORT.json` 复核执行报告，不是世界模型的科学效果分数。

读取 NPZ 时先检查 NumPy keys，再查看 shape、dtype 与有限值；
不要假定所有 adapter 的数组键名或动作维度相同。

## GitHub 与远程服务器播放

README 使用 GIF 预览链接 MP4，避免 GitHub 对本地 HTML video 标签的渲染限制。
GIF 为减小体积进行了降采样，MP4 保留该次保存视频的 15 FPS。

自己的产物可下载 MP4 到电脑打开；服务器本地路径不是浏览器 URL。
如需临时网页访问，请仅绑定 localhost 并通过隧道转发，
不要将整个项目、数据或凭据暴露到公网 HTTP 服务。

## 结果保留

`results/` 被 Git 忽略且不进入公开分发。
`cli cleanup-results` 预览保留策略，`--apply` 可恢复地归档符合条件的生成产物，
不能操作权重、数据和环境。
公开展示视频是 `docs/assets/` 下经过精确白名单限定的少量例外，并附来源记录。

[评价流程](workflows.md) · [排障指南](troubleshooting.md)
