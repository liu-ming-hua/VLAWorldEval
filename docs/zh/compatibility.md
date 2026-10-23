# 组合兼容性

[手册首页](README.md) / 选择组合 · [English](../en/compatibility.md)

## 从显式组合开始

当前 16 条组合如下。B 是目录定义的基础支持路径；E 是实验桥接路径。
这不是性能或任务成功排序，也不保证当前机器已安装所需资产。
同一模型的多卡配置只是资源变体，不另算模型组合。

| Pair | Tier | Configuration |
| --- | :---: | --- |
| OpenVLA × WorldGym | B | [`openvla_worldgym_v1.json`](../../configs/experiments/openvla_worldgym_v1.json) |
| SpatialVLA × WorldGym | B | [`spatialvla_worldgym_v1.json`](../../configs/experiments/spatialvla_worldgym_v1.json) |
| Octo × WorldGym | B | [`octo_worldgym_v1.json`](../../configs/experiments/octo_worldgym_v1.json) |
| RT-1-X × WorldGym | B | [`rt1_x_worldgym_v1.json`](../../configs/experiments/rt1_x_worldgym_v1.json) |
| GR00T N1.5 × Cosmos EEF FD | B | [`engineering_groot_cosmos_v1.json`](../../configs/experiments/engineering_groot_cosmos_v1.json) |
| GR00T N1.5 × Ctrl | E | [`groot_n15_ctrl_world_droid_v1.json`](../../configs/experiments/groot_n15_ctrl_world_droid_v1.json) |
| GR00T N1.7 base × Cosmos EEF FD | E | [`groot_n17_cosmos_droid_3gpu_v1.json`](../../configs/experiments/groot_n17_cosmos_droid_3gpu_v1.json) |
| GR00T N1.7 base × Ctrl | E | [`groot_n17_ctrl_world_droid_v1.json`](../../configs/experiments/groot_n17_ctrl_world_droid_v1.json) |
| GR00T N1.7-DROID × Cosmos EEF FD | E | [`groot_n17_droid_cosmos_v1.json`](../../configs/experiments/groot_n17_droid_cosmos_v1.json) |
| GR00T N1.7-DROID × Ctrl | E | [`groot_n17_droid_ctrl_v1.json`](../../configs/experiments/groot_n17_droid_ctrl_v1.json) |
| π0 DROID × Cosmos EEF FD | E | [`pi0_cosmos_droid_v1.json`](../../configs/experiments/pi0_cosmos_droid_v1.json) |
| π0 DROID × Ctrl | E | [`pi0_ctrl_world_droid_v1.json`](../../configs/experiments/pi0_ctrl_world_droid_v1.json) |
| π0.5 DROID × Cosmos EEF FD | E | [`pi05_cosmos_droid_v1.json`](../../configs/experiments/pi05_cosmos_droid_v1.json) |
| π0.5 DROID × Ctrl | E | [`pi05_ctrl_world_droid_v1.json`](../../configs/experiments/pi05_ctrl_world_droid_v1.json) |
| π0-FAST DROID × Cosmos EEF FD | E | [`pi0_fast_droid_cosmos_v1.json`](../../configs/experiments/pi0_fast_droid_cosmos_v1.json) |
| π0-FAST DROID × Ctrl | E | [`pi0_fast_droid_ctrl_v1.json`](../../configs/experiments/pi0_fast_droid_ctrl_v1.json) |

## 接入前必须匹配

| 条件 | 要核对的内容 |
| --- | --- |
| 机器人与动作 | EEF / 关节；delta / absolute；坐标系；旋转表示；夹爪方向与单位 |
| 视角 | 相机身份、顺序、分辨率、crop；腕部与外部视角不可互换 |
| 时间 | 动作频率、视频频率、chunk、已有上下文、重复首帧和末尾 padding |
| 归一化 | checkpoint 对应统计量；已转换动作不得重复归一化 |
| 策略额外输入 | 需要状态或图像历史时必须显式提供，不能从“视频输出”推断存在状态 |
| 数据域 | 原生训练域与 checkpoint 专属输入要求 |

Bridge 路径使用 7D 到 WorldGym 10D 的明确映射；不能把同样的 padding 用于任意机器人。
DROID 路径包含 FK、时间对齐与绝对/相对动作转换，且当前非测量状态反馈受到假设限制。
真实测量缺失时必须保留来源标签；不能称为 WM 预测状态或真机执行结果。

## 目录等级怎么读

- **supported**：只针对明确列出的基础组合，不代表物理可靠。
- **native_replay**：支持模型原生条件回放，不代表能与现有 VLA 配对。
- **experimental**：可能仅包含启动、特定转换或资源尚不完整的入口。
- **incompatible**：相对于当前机器人 action loop 不兼容，不表示其原生推理无效。

`cli models` 查看模型目录；`cli capabilities` 查看显式配对；
`cli runtimes` 查看构造和数据入口。登记新模型不能自动扩大这些配对。

[模型输入输出](models.md) · [运行方法](workflows.md) · [新增适配器](extending.md)
