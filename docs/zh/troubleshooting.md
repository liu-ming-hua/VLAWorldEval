# 排障指南

[手册首页](README.md) / 排障 · [English](../en/troubleshooting.md)

| 现象 | 优先检查 | 不应采用的处理 |
| --- | --- | --- |
| 依赖解析冲突 | 模型是否使用独立环境，环境键与依赖版本是否匹配 | 将所有 VLA extras 一次装进同一解释器 |
| CUDA 驱动不匹配 | 驱动与当前 Torch / CUDA 构建是否兼容 | 未协调就升级共享机器驱动 |
| 下载超时或网络不可达 | 代理是否清除，目标主机是否可达，镜像与认证是否适用于该主机 | 假定 HF 镜像覆盖 Drive / GCS / GitHub |
| 缺少网络结构代码 | 对应 setup 是否执行，上游源码身份是否匹配 | 将任意模型源码复制到核心代码 |
| GPU 准入失败 | 物理 GPU ID、空余显存、是否声明多卡、两个 Worker 的预算 | 把显存上限当作模型压缩 |
| 缺少状态 / 历史 / 视角 | 原生协议和 pair manifest 的要求 | 伪造测量状态或复制相机身份 |
| 视频静止或花屏 | RGB 布局、归一化、动作坐标和频率、上下文、逐视角 VAE | 直接归因为模型无效或盲目放大动作 |
| 输出目录已存在 | 使用新的版本目录 | 删除证据后覆盖重跑 |
| 推理慢 | 分开统计加载、去噪、自回归依赖与评分开销，验证常驻 / batch 能力 | 不记录就降低科学配置 |
| MP4 播放不了 | ffprobe、编码与像素格式，先复制到本机查看 | 把远程文件路径当网页链接 |

## 常用只读检查

```bash
environment/core/bin/python -m cli validate
environment/core/bin/python -m cli environments
environment/core/bin/python -m cli env-audit --environment vla.openvla --dependencies
environment/core/bin/python -m cli inspect --artifact results/first_demo
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/openvla_worldgym_v1.json --experiment loop_smoke --check-only
```

CLI 返回结构化错误；需要堆栈时，在子命令前添加 `--debug`。
反馈问题时附配置、准确错误、模型版本和环境审计，不要附 token、私钥、
私有数据帧或其他用户的个人路径。

[安装指南](installation.md) · [模型协议](models.md) · [视频查看](artifacts.md)
