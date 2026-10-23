# 新增模型或配对

[手册首页](README.md) / 扩展 · [English](../en/extending.md)

## 接入清单

1. **先明确原生协议。** 固定源码和权重版本，记录机器人、训练域、视角、图像布局、
   动作语义、历史与时间频率。
2. **新增一个 adapter**，放入 `adapters/vla/` 或 `adapters/world_models/`，
   保留官方推理方式，模型库延迟导入。
3. 在[环境注册表](../../environment/registry.toml)声明独立键，
   安装下载入口复用代理清除、项目缓存和校验 helper。
4. 更新[模型目录](../../configs/model_catalog.json)；模型登记不会自动启用 VLA × WM 配对。
5. 在[运行注册表](../../configs/runtime_registry.json)绑定构造参数、必要资产、分布式能力与数据入口。
6. 新组合必须包含[pair manifest](../../configs/pairs/) 和[实验配置](../../configs/experiments/)，
   对动作 / 视角桥接的单位、坐标系、频率、旋转与夹爪约定分别测试。
7. **先无权重契约测试，再原生样例验证。** 维度、官方源码一致性、生命周期与资产身份，
   不应混同为模型科学可靠性。
8. 更新两种语言的使用手册与组合表。仅启动成功的模型不能宣称完成闭环配对。

## 适配接口

先阅读 [base.py](../../adapters/base.py)、[contracts.py](../../evaluation/contracts.py)，
选择原生协议真正相近的现有 adapter 作为参照，而不只是动作维度相同。
共享 evaluator 与 native replay，不在模型脚本中复制另一套 rollout 编排。

通用转换放入 `adapters/actions/` 或 `adapters/observations/`；
插件使用显式模块 / 类引用。未知机器人协议应拒绝执行，而不是静默补维。

## 分阶段校验

```bash
CORE_DEV=1 bash scripts/setup/core.sh
bash scripts/check_project.sh
environment/core/bin/python -m cli models
environment/core/bin/python -m cli runtimes
environment/core/bin/python -m cli capabilities
```

无权重测试不能下载模型或初始化 CUDA。
真实模型测试必须指定独立环境、权重和显式 GPU 执行。
在策略比较前，先完成原生域真实动作回放与 reference 对照。

## 贡献边界

权重、环境、上游 clone、缓存和完整结果不能提交 Git。
共享配置使用项目相对路径，个人路径通过显式 override 注入。
参考 [CONTRIBUTING](../../CONTRIBUTING.md)、[发布规则](release.md)和[可靠性边界](reliability.md)。
