# 构建与发布

[手册首页](README.md) / 发布 · [English](../en/release.md)

## 校验源码

```bash
bash scripts/check_project.sh
environment/core/bin/python scripts/acceptance/public_delivery.py \
  --offline --output results/delivery/my_release_check
```

请选择新目录。离线交付检查要求构建与核心依赖已在缓存中；
它在原 checkout 之外验证源码安装、wheel 资源和真实 CPU Worker。
不安装所有 GPU 模型，不发布远程仓库，也不验证机器人动力学。

## 导出干净的源码快照

```bash
environment/core/bin/python scripts/acceptance/export_source.py \
  --sdist results/delivery/my_release_check/distributions/vlaworldeval-0.1.0.tar.gz \
  --output dist/github/my_release
```

导出器拒绝危险归档路径、运行时资产和疑似凭据，记录文件哈希且拒绝覆盖已有目录。
只有 `docs/assets/` 下经过精确白名单限定的展示素材可以进入分发。
有限模式扫描不等于完整的安全或法律审计。

若工作目录继承了上游 Git 历史，应将审阅后的快照用于新的干净仓库，
不要 force-add 被忽略的权重、结果、私有数据或上游源码。
以上命令不会创建远程仓库、上传文件或改写本地历史。

## 构建文档网站

网站采用 Sphinx + MyST + PyData 主题，直接复用 `docs/en/` 与 `docs/zh/` 的 Markdown 手册，
不需要另写一套网页内容，也不应手动编辑生成的 HTML。

```bash
bash scripts/setup/docs.sh
environment/tools/docs/bin/python -m sphinx -W --keep-going -b html docs .cache/site/html
environment/core/bin/python scripts/checks/site.py --directory .cache/site/html
environment/tools/docs/bin/python -m http.server 8000 --directory .cache/site/html
```

打开 `http://localhost:8000/`；服务器环境需先通过 SSH 将 8000 端口转发到自己的电脑。
文档环境单独存放在 `environment/tools/docs/`，安装时关闭代理、使用配置的 PyPI 镜像；
不依赖模型权重或 Torch 等模型包，生成网页保存在 `.cache/`。

GitHub 仓库首次需选择 **Settings → Pages → Source → GitHub Actions**。
`documentation-pages` 工作流在 PR 中构建、校验网站，在 `main` 上发布到项目 GitHub Pages。
若尚未启用 Pages，工作流保留验证后的网页产物并提示设置步骤，不会把构建成功误报为部署成功。
设置完成后可以从 **Actions → documentation-pages → Run workflow** 手动触发发布。

全文检索同时索引中英文，语言切换保留当前所在的手册页面。

## 许可边界

原创框架贡献采用 [MIT](../../LICENSE)，具体见[许可范围](../../LICENSE_SCOPE.md)。
独立获取的模型代码、权重、数据、可选评价器及派生演示素材保留各自适用条款。
公开可下载不自动等于可以重新分发。

WorldGym 的运行实现通过 [worldgym_runtime.sh](../../scripts/setup/worldgym_runtime.sh)
单独获取，不包含在框架分发中。已检查的固定上游树没有根 LICENSE，
本框架不会将其重新授权为 MIT。这是分发边界，不是使用已有模型前必须询问作者。

正式发布前请复核源码许可、数据 / 展示素材授权、模型条款与敏感信息。
标题横幅是 AI 生成的品牌图，框架 SVG 是代码绘制的示意图，都不是实验证据；
真实演示来源见[素材清单](../assets/provenance.json)。

[贡献指南](../../CONTRIBUTING.md) · [手册首页](README.md)
