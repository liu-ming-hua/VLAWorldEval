# Build & release

[Manual](README.md) / Release · [简体中文](../zh/release.md)

## Validate the source

```bash
bash scripts/check_project.sh
environment/core/bin/python scripts/acceptance/public_delivery.py \
  --offline --output results/delivery/my_release_check
```

Choose a fresh output name. The offline delivery gate requires cached build/core dependencies;
it tests source installation, wheel resources and actual CPU workers away from this checkout.
It does not install every GPU stack, publish a repository or validate robot physics.

## Export a clean source snapshot

```bash
environment/core/bin/python scripts/acceptance/export_source.py \
  --sdist results/delivery/my_release_check/distributions/vlaworldeval-0.1.0.tar.gz \
  --output dist/github/my_release
```

The exporter rejects unsafe archive paths, runtime payloads and credential-like patterns,
writes file hashes and refuses to overwrite destinations.
Only specifically allowlisted presentation media under `docs/assets/` are included.
A bounded pattern scan is not a complete security or legal audit.

For a workspace with inherited upstream Git history, publish only the reviewed export into a
clean repository. Do not force-add ignored weights, results, private datasets or upstream source.
The commands above create no remote, upload nothing and do not rewrite local history.

## Build the documentation website

The website uses Sphinx + MyST + the PyData theme. The Markdown manuals remain
the single source; edit `docs/en/` and `docs/zh/`, not generated HTML.

```bash
bash scripts/setup/docs.sh
environment/tools/docs/bin/python -m sphinx -W --keep-going -b html docs .cache/site/html
environment/core/bin/python scripts/checks/site.py --directory .cache/site/html
environment/tools/docs/bin/python -m http.server 8000 --directory .cache/site/html
```

Open `http://localhost:8000/`. On a remote server, forward port 8000 with SSH first.
The dedicated documentation environment disables proxies and uses the configured PyPI mirror;
no model dependencies or weights are required. Generated HTML stays under `.cache/`.

For GitHub Pages, select **Settings → Pages → Source → GitHub Actions** once.
The `documentation-pages` workflow builds and validates both languages on pull requests,
and publishes `main` to the project Pages URL. If Pages is disabled, the workflow
keeps the validated artifact and reports the required setting instead of claiming deployment.
Use **Actions → documentation-pages → Run workflow** after enabling Pages.

Search indexes both English and Chinese text. Language switches preserve the current manual page.

## License boundary

Original framework contributions use [MIT](../../LICENSE); see [license scope](../../LICENSE_SCOPE.md).
Separately obtained model code, weights, data, optional evaluators and derived demo media retain
their own applicable terms. Being publicly downloadable does not automatically grant redistribution.

WorldGym runtime implementations are obtained separately by
[scripts/setup/worldgym_runtime.sh](../../scripts/setup/worldgym_runtime.sh) and excluded from
framework distributions. The inspected pinned upstream tree has no root LICENSE file;
the framework does not relicense it as MIT. This is a distribution boundary, not a requirement
to ask authors before running an already obtained model.

Before publishing: review source licensing, dataset/media permissions, model terms and secrets.
The branded banner is AI-generated presentation artwork; the framework SVG is code-native.
Neither represents experimental evidence. Demo provenance is in [assets](../assets/provenance.json).

[Contributing](../../CONTRIBUTING.md) · [Documentation index](README.md)
