"""The website remains portable, bilingual and independent of model packages."""
from pathlib import Path
import re
import runpy

import pytest

from scripts.acceptance.export_source import allowed
from scripts.checks.site import validate_site


ROOT = Path(__file__).resolve().parents[1]


def test_site_nav_covers_the_existing_manual_without_a_parallel_content_tree():
    config = runpy.run_path(str(ROOT / "docs/conf.py"))
    assert config["html_theme"] == "pydata_sphinx_theme"
    assert config["html_search_language"] == "zh"
    nav = (ROOT / "docs/_templates/manual-nav.html").read_text()
    entries = set(re.findall(r"\('([A-Za-z_]+)',", nav))
    for language in ("en", "zh"):
        assert entries == {p.stem for p in (ROOT / "docs" / language).glob("*.md")}
    assert "pydata-sphinx-theme==" in (ROOT / "environment/requirements/docs.txt").read_text()


def test_site_rewrites_only_repository_links_not_manual_pages_or_images():
    rewrite = runpy.run_path(str(ROOT / "docs/conf.py"))["repository_links"]
    assert rewrite("[Next](workflows.md)", "en/quickstart") == "[Next](workflows.md)"
    assert rewrite("![Diagram](../assets/framework.svg)", "en/architecture") == "![Diagram](../assets/framework.svg)"
    text = rewrite("[Config](../../configs/runtime_registry.json#pairs)", "en/README")
    assert text == "[Config](https://github.com/liu-ming-hua/VLAWorldEval/blob/main/configs/runtime_registry.json#pairs)"
    assert rewrite("[External](https://example.org)", "en/README") == "[External](https://example.org)"


@pytest.mark.parametrize("name", ["docs/conf.py", "docs/index.rst", "docs/_static/manual.css",
                                 "docs/_templates/landing.html", ".github/workflows/docs.yml"])
def test_source_export_includes_website_sources_but_not_generated_runtime(name):
    assert allowed(Path(name))
    assert not allowed(Path("docs/assets/private.html"))
    assert not allowed(Path("environment/tools/docs/bin/python"))
    assert not allowed(Path(".cache/site/html/index.html"))


def site_fixture(tmp_path):
    (tmp_path / "index.html").write_text('<h1 id="welcome">Welcome</h1><a href="guide.html#start">Guide</a>')
    (tmp_path / "guide.html").write_text('<h1 id="start">Guide</h1><a href="index.html#welcome">Home</a>')
    (tmp_path / "searchindex.js").write_text("Search.setIndex({})")
    return tmp_path


def test_site_validator_checks_files_and_anchors_without_network(tmp_path):
    directory = site_fixture(tmp_path)
    (directory / "_static").mkdir()
    (directory / "_static/macros.html").write_text('<a href="{{ unrendered_theme_macro }}">Template</a>')
    assert validate_site(directory)["internal_links"] == 2
    (directory / "guide.html").write_text('<a href="index.html#missing">Missing anchor</a>')
    with pytest.raises(ValueError, match="Missing anchor"):
        validate_site(directory)
    (directory / "guide.html").write_text('<img src="missing.png">')
    with pytest.raises(ValueError, match="missing.png"):
        validate_site(directory)


def test_site_validator_rejects_missing_search_index(tmp_path):
    (tmp_path / "index.html").write_text("Homepage")
    with pytest.raises(ValueError, match="search index"):
        validate_site(tmp_path)


def test_docs_workflow_keeps_deployment_separate_from_build_success():
    workflow = (ROOT / ".github/workflows/docs.yml").read_text()
    assert "needs: build" in workflow and "id-token: write" in workflow
    assert "pull_request" in workflow and "workflow_dispatch" in workflow
    assert "pages.outputs.enabled == 'true'" in workflow
    assert "--keep-going" in workflow and "scripts/checks/site.py" in workflow
    assert "pip install -e ." not in workflow
