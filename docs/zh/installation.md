# 安装与资产管理

[手册首页](README.md) / 安装 · [English](../en/installation.md)

## 分开管理三个层次

| 层次 | 路径 | 职责 |
| --- | --- | --- |
| 核心 | `environment/core` | CLI、协议、调度与产物检查 |
| 模型环境 | `environment/vla/`、`environment/world_models/` | 各模型官方推理依赖；可通过注册表覆盖位置 |
| 模型资产 | `checkpoints/`、`datasets/`、`third_party/` | 权重、数据与独立获取的上游源码 |

以下命令从项目根目录执行。不要把所有模型装进一个环境：
例如 OpenVLA 与 SpatialVLA 使用的 Transformers 版本冲突。

```bash
bash scripts/setup/core.sh
# 可选：开发工具与 DROID Parquet 支持。
CORE_DEV=1 CORE_DROID=1 bash scripts/setup/core.sh
environment/core/bin/python -m cli environments
environment/core/bin/python -m cli models
```

核心安装需要本机已有 Python 3.11；必要时设置 `CORE_PYTHON=/absolute/path/to/python3.11`，
脚本不会悄悄下载 Python。

## 网络约定

安装入口复用 [common.sh](../../scripts/setup/common.sh)，关闭代理变量，将缓存保留在项目内。
默认使用 PyPI 镜像与 `HF_ENDPOINT=https://hf-mirror.com`。
可显式覆盖 `PYPI_MIRROR` / `HF_ENDPOINT`。镜像不绕过认证或模型许可；
GitHub 源码、Google Drive 和 GCS 资源仍从其实际托管位置获取。

不要把整个 venv 直接复制到其他机器当作可移植环境：
解释器链接、shebang、二进制扩展和 CUDA 版本都可能与主机绑定。
请按脚本与依赖文件重建，可复用 uv 缓存降低开销。

## 按需安装

| 模型系列 | 环境入口 | 权重入口 |
| --- | --- | --- |
| OpenVLA / SpatialVLA / Octo | `scripts/setup/vla_envs.sh MODEL` | `scripts/download/vla_hf.sh MODEL` |
| RT-1-X | `scripts/setup/vla_envs.sh rt1` | `scripts/download/rt1_x.sh` |
| GR00T N1.5 / N1.7 | `scripts/setup/groot_n15.sh` / `groot_n17.sh` | `scripts/download/vla_hf.sh`，先查看模型名称选项 |
| OpenPI DROID 系列 | `scripts/setup/openpi.sh` | `scripts/download/openpi.sh` / `droid_specialists.sh` |
| SmolVLA / DreamZero | `scripts/setup/smolvla.sh` / `dreamzero.sh` | 先检查对应脚本接受的选择项 |
| WorldGym | `scripts/setup/worldgym.sh` | 安装入口调用 checkpoint helper，官方 Drive 托管 |
| Cosmos DROID EEF FD | `scripts/setup/cosmos3_nano_droid_fd.sh` | `scripts/download/cosmos3_nano_droid_fd.sh` |
| Ctrl / A2World / GE-Sim / τ₀ | 对应 `scripts/setup/*.sh` | 安装入口与声明的官方资产 |
| IWS / IRASim / DreamDojo / DreamTrue / PlayWorld | 对应 `scripts/setup/*.sh` | 对应 `scripts/download/*.py --help` |

这是入口索引，不要求运行全部安装器。[环境脚本](../../scripts/setup/) · [下载脚本](../../scripts/download/) ·
[各模型协议](models.md)。下载前请检查选项和磁盘空间。

## 路径与环境选择

[environment/registry.toml](../../environment/registry.toml) 将 adapter 的环境键映射到解释器。
共享配置使用项目相对路径。已有外部环境时，可将 `VLAWORLDEVAL_ENVIRONMENTS` 指向个人 JSON：

```json
{"schema_version": 1, "paths": {"vla.openvla": "/absolute/path/to/openvla_env"}}
```

覆盖目录必须存在且是绝对路径。不要提交个人路径、密钥或 token。
指定路径不代表其中依赖兼容。

```bash
environment/core/bin/python -m cli env-audit --environment vla.openvla --dependencies
environment/core/bin/python -m cli verify-assets --directory checkpoints/vla/openvla-7b
```

## 使用 wheel

安装本地构建的 wheel 后，可运行 `vlaworldeval init --directory /absolute/new_workspace`，
并设置 `VLAWORLDEVAL_ROOT`。该操作仅初始化空工作区，不下载资产，不覆盖非空目录。
环境安装和上游补丁仍从源码 checkout 执行。

[快速开始](quickstart.md) · [排障指南](troubleshooting.md)
