# 模型参考：输入、输出与训练域

[手册首页](README.md) / 模型参考 · [English](../en/models.md)

张量以项目 adapter 实际协议为准，不是同名模型所有 checkpoint 的能力。
`T` 为时间长度，`V` 为视角数；RGB 为 uint8 HWC，模型内部转换由适配器负责。
动作宽度相同不代表坐标系、控制语义、统计量或机器人相同。

## VLA 接口

下表保留上游动作术语，避免将 relative EEF、absolute target 和 joint velocity 混用。

| 模型 | 图像与指令输入 | Adapter 动作输出 | 机器人状态输入 | 训练 / 使用域 |
| --- | --- | --- | --- | --- |
| OpenVLA | RGB + 指令 | [T,10] ← 原生 7D | — | OXE / Bridge |
| SpatialVLA | RGB + 指令 | [T,10] ← 原生 7D | — | OXE + RH20T |
| Octo | RGB 历史 + 指令 | [T,10] ← 原生 7D | — | OXE / Bridge |
| RT-1-X | RGB 256×320 + USE512 / 指令 | [1,10] ← 原生动作字典 | — | OXE / Bridge |
| GR00T N1.5 | 三路 DROID RGB + 指令 | [16,7] 相对 EEF + 夹爪 | EEF 位姿 + 夹爪 | OXE-DROID |
| GR00T N1.7 base | 两路视角 × 历史 / 当前帧 + 指令 | [40,17] 解码后的绝对目标 | EEF9 + 关节7 + 夹爪1 | 通用底座 / OXE-DROID |
| GR00T N1.7-DROID | 两路当前 RGB + 指令 | [40,17] 解码后的绝对目标 | EEF9 + 关节7 + 夹爪1 | DROID 专用后训练 |
| π0 DROID | 外部视角 + 腕部视角 + 指令 | [10,8] 关节速度 + 夹爪 | 关节7 + 夹爪1 | DROID |
| π0.5 DROID | 外部视角 + 腕部视角 + 指令 | [15,8] 关节速度 + 夹爪 | 关节7 + 夹爪1 | DROID / RoboArena |
| π0-FAST DROID | 外部视角 + 腕部视角 + 指令 | [10,8] 关节速度 + 夹爪 | 关节7 + 夹爪1 | DROID |
| SmolVLA base | 具名 RGB 视角 + 指令 | [50,6] | 6D | 社区数据底座；需要机器人专用微调 |
| Cosmos DROID Policy | 三路 DROID RGB + 指令 | [32,8] 绝对关节 / 夹爪 | 关节7 + 夹爪1 | DROID 后训练；未注册 WM 配对 |
| DreamZero DROID | 三视角；首轮 1 帧，后续 4 帧 + 指令 | [24,8] 绝对关节 / 夹爪 | 关节7 + 夹爪1 | DROID；原生 GPU 验证尚未完成 |

## 世界模型接口

绝大多数模型只输出视频；GE-Sim 的 Pose-Expert 是模型特定例外，不能作为通用状态接口的承诺。
Ctrl 的 inclusive 窗口包含已有上下文，不能把每次 5 帧都计作 5 个新未来帧。

| Checkpoint | 观测条件 | 动作条件 | 输出 | 训练 / 输入域 |
| --- | --- | --- | --- | --- |
| WorldGym | 单路 RGB 256² | [T,10]；核心为 Bridge 7D | RGB 视频；无状态输出 | 九个 OXE 数据集；明确包含 Bridge V2 / RT-1；完整混合未公开 |
| Ctrl-World | 三路 RGB 192×320，稀疏历史 | [5,7] 归一化绝对 EEF | 包含上下文的 5 帧 × 3 视角视频；无状态输出 | DROID |
| Cosmos EEF FD · 20k EMA | 三视角画布 540×640 → 528×640 | [16,10] 相对 OpenCV EEF rot6D + 翻转夹爪 | 15 FPS 下 16 个未来帧；无状态输出 | 社区 DROID 后训练；不是 NVIDIA 通用底座权重 |
| Cosmos joint FD · best5600 | 三视角画布 + 当前状态8 | [32,8] 绝对关节 / 夹爪 | 32 个未来帧；无状态输出头 | 独立 DROID 关节版本 |
| Cosmos3-Nano base | RGB + 数据域专属条件 | 依数据域；不是 EEF FD 后训练 | 视频；FD 接口不输出状态 | 通用预训练底座 |
| A2World | 两路或三路同步 RGB | [20,14] 原生机器人协议 | 多视角 RGB 256²；无状态输出 | 210 万以上轨迹 / 20 多种机器人形态 |
| GE-Sim 2.0 | 三路标定 G01 视角 + FK 资产 | [T,16] 绝对关节 | 三视角 RGB + Pose-Expert [T,16] | G01 / OmniPicker 发布版本 |
| τ₀-WM Simulator | 图像 [V,3,192,256] | [T,20] 双臂相对 EEF API；内部为 26D | 终帧视角 / 视频；无状态输出 | 公开全身模拟器；不是 TacoPlay 7D |
| DreamDojo 2B GR1 | GR1 初始 RGB + 原生条件 | 29 个归一化关节，编码到 384 个槽位 | 含上下文的 49 帧原生窗口；无状态输出 | 人类视频预训练 + GR1 后训练 |
| IWS PushT cam1 | RGB 128² + 初始世界坐标 XY | [T,4] 官方 FK 得到的绝对世界坐标 XY | RGB 128²；无状态输出 | 任务专用真实 ALOHA PushT |
| PlayWorld SVD | 任务原生图像 / 条件 | 任务专属原生编码 | 视频；无状态输出 | DROID 初始化 + robot-play 后训练 |
| DreamTrue | 经标定的多视角渲染几何 | 结构化几何，而非裸动作向量 | 多视角 RGB；无状态输出 | AgiBot / DROID / RoboMIND / RoboTwin |
| IRASim Language-Table | 原生初始 RGB / latent | Language-Table 平面 XY | RGB 视频；无状态输出 | Language-Table Frame-Ada 公开权重 |
| Giga-World-1 FunCtrl | 初始图像 + Plücker / Ray-Map 控制视频 | 控制视频，而非机器人动作张量 | RGB 视频；无状态输出 | 混合视频 / 机器人 / 第一人称语料；Toydata 仅为演示 |

## 选择数据时的基本判断

- WorldGym 发布的 Bridge 示例是 9 train + 1 test，不是完整 OXE 训练混合或论文完整测试集。
- DROID 后训练模型需要 DROID 的视角、机器人与动作协议；任意机械臂视频不是等价输入。
- IWS 必须使用真实观测关节经官方 FK 得到的 world XY，不能用存储命令替代观测位置。
- Giga-World-1 的公开 FunCtrl 入口不接受裸机器人 action；Toydata 不验证 VLA 动作条件闭环。
- τ₀ 的 TacoPlay 7D 数据与公开 20D API checkpoint 不同，不能补零伪装匹配。
- N1.7 base 与 DROID 专用后训练、Cosmos EEF FD 与 joint FD、PlayWorld SVD 与 AR 版本必须分开。

## 环境与官方来源

详细机器可读契约在[模型目录](../../configs/model_catalog.json)，构造参数在[运行注册表](../../configs/runtime_registry.json)。
下载入口见[安装指南](installation.md)，原生评价条件见[操作指南](workflows.md)。

| ID | 环境键 | 官方来源 |
| --- | --- | --- |
| `openvla` | `vla.openvla` | [上游](https://openvla.github.io/) |
| `spatialvla` | `vla.spatialvla` | [上游](https://github.com/SpatialVLA/SpatialVLA) |
| `octo` | `vla.octo` | [上游](https://github.com/octo-models/octo) |
| `rt1_x` | `vla.rt1` | [上游](https://github.com/google-deepmind/open_x_embodiment) |
| `groot_n15` | `vla.groot_n15` | [上游](https://github.com/NVIDIA/Isaac-GR00T) |
| `groot_n17` | `vla.groot_n17` | [上游](https://github.com/NVIDIA/Isaac-GR00T) |
| `pi0_droid` | `vla.openpi` | [上游](https://github.com/Physical-Intelligence/openpi) |
| `pi05_droid` | `vla.openpi` | [上游](https://github.com/Physical-Intelligence/openpi) |
| `smolvla` | `vla.smolvla` | [上游](https://github.com/huggingface/lerobot) |
| `groot_n17_droid` | `vla.groot_n17` | [上游](https://github.com/NVIDIA/Isaac-GR00T) |
| `pi0_fast_droid` | `vla.openpi` | [上游](https://github.com/Physical-Intelligence/openpi) |
| `cosmos_policy_droid` | `vla.cosmos_policy_droid` | [上游](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) |
| `dreamzero_droid` | `vla.dreamzero` | [上游](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) |
| `worldgym` | `world_models.worldgym` | [上游](https://arxiv.org/html/2506.00613v3) |
| `ctrl_world` | `world_models.ctrl_world` | [上游](https://github.com/Robert-gyj/Ctrl-World) |
| `cosmos3_nano` | `world_models.cosmos3_nano` | [上游](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/action) |
| `cosmos3_nano_droid_fd_20k_ema` | `world_models.cosmos3_nano_droid_fd` | [上游](https://huggingface.co/jere-mybao/cosmos3-nano-droid-forward-dynamics) |
| `cosmos3_droid_joint_fd_best5600` | `world_models.cosmos3_droid_joint_fd` | [上游](https://huggingface.co/anshks/cosmos3-nano-droid-fd-wm) |
| `a2world` | `world_models.a2world` | [上游](https://github.com/LogosRoboticsGroup/A2World) |
| `ge_sim_v2` | `world_models.ge_sim_v2` | [上游](https://github.com/AgibotTech/GE-Sim-V2) |
| `tau0_wm_simulator` | `world_models.tau0_wm_simulator` | [上游](https://github.com/sii-research/tau-0-wm) |
| `dreamdojo_2b_gr1` | `world_models.dreamdojo` | [上游](https://github.com/NVIDIA/DreamDojo) |
| `iws_pusht_cam1` | `world_models.iws` | [上游](https://github.com/WangYixuan12/interactive_world_sim) |
| `playworld_svd` | `world_models.playworld` | [上游](https://github.com/irom-princeton/open-world) |
| `dreamtrue_threeview_v1` | `world_models.dreamtrue` | [上游](https://github.com/brave-eai/DreamTrue) |
| `irasim_languagetable_frame_ada` | `world_models.irasim` | [上游](https://github.com/bytedance/IRASim) |
| `gigaworld1` | `world_models.gigaworld1` | [上游](https://github.com/open-gigaai/giga-world-1) |

[组合兼容性](compatibility.md) · [指标解释](reliability.md)
