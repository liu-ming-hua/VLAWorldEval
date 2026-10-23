# 第一个闭环

[手册首页](README.md) / 快速开始 · [English](../en/quickstart.md)

## 1 · 检查轻量核心

在源码根目录执行，需要 Python 3.11、uv 和 FFmpeg：

```bash
bash scripts/setup/core.sh
environment/core/bin/python -m cli demo --output results/first_demo --chunks 30
environment/core/bin/python -m cli inspect --artifact results/first_demo
environment/core/bin/python -m cli capabilities
```

预期得到连续合成视频、动作数组和可检查产物。该过程验证框架，不代表预训练模型效果或任务成功。
每次请选择尚不存在的输出目录。

## 2 · 准备一个真实组合

以 OpenVLA + WorldGym 为例：

```bash
bash scripts/setup/vla_envs.sh openvla
bash scripts/download/vla_hf.sh openvla
bash scripts/setup/worldgym.sh
environment/core/bin/python scripts/download/worldgym_data.py
```

WorldGym 的安装入口也会准备独立获取的运行源码。公开权重托管于 Google Drive；
VAE 需要遵循上游访问条款。HF 镜像不能代替这些托管位置或授权。下载前请阅读[安装指南](installation.md)。

## 3 · 选择设备

```bash
environment/core/bin/python -m cli configure \
  --config configs/experiments/openvla_worldgym_v1.json \
  --output configs/experiments/my_openvla.json \
  --vla-gpus 0 --wm-gpus 1 --budget-gib 20 --vla-budget-gib 18
```

显存预算是准入与分配器上限，不是模型实际需求估计。
只有声明支持分布式的适配器才接受多个 GPU；不要擅自占用其他人的计算资源。

## 4 · 先检查，再执行

```bash
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/my_openvla.json \
  --experiment loop_smoke --check-only
environment/core/bin/python -m cli evaluate \
  --config configs/experiments/my_openvla.json \
  --experiment loop_smoke --execute
```

check-only 检查配置、资产、环境和真实数据窗口，不加载模型。
execute 额外检查当前 GPU 资源。使用返回的产物路径执行 `cli inspect --artifact PATH`。

## 继续阅读

[选择其他组合](compatibility.md) · [长闭环与真实动作回放](workflows.md) · [理解输出](artifacts.md)
