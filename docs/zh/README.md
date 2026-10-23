# 使用手册

[项目首页](../../README_CN.md) · [English](../en/README.md)

**从第一个闭环到接入新模型。** 这里说明框架的使用方法与适用边界，不记录实验流水账。

## 开始使用

| 目标 | 阅读 |
| --- | --- |
| 不下载权重、不使用 GPU 即可运行 | [快速开始](quickstart.md) |
| 安装一个模型与相关资产 | [安装与镜像](installation.md) |
| 选择合理的 VLA / WM 组合 | [组合兼容性](compatibility.md) |
| 核对输入输出、训练域与状态要求 | [模型参考](models.md) |

## 操作指南

- [运行闭环或真实动作回放](workflows.md)
- [查看视频与动作数组](artifacts.md)
- [处理环境、协议和执行错误](troubleshooting.md)
- [在不复制评估循环的前提下接入新模型](extending.md)

## 原理与维护

- [架构及目录职责](architecture.md)
- [指标解读与可靠性边界](reliability.md)
- [构建、检查与发布](release.md)

### 阅读约定

模型登记不等于存在可执行配对；可执行配对也不代表物理正确。
`--check-only` 和 `--admit-only` 不会生成视频。每次请选择新输出目录。
命令默认从源码根目录运行，模型环境只有显式安装时才会创建。

声明来源：[模型目录](../../configs/model_catalog.json)、
[运行注册表](../../configs/runtime_registry.json)、
[环境注册表](../../environment/registry.toml)。
