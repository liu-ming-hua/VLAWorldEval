"""Single-source bilingual manual, with project-relative GitHub Pages output."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from urllib.parse import urlsplit


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
REPOSITORY = "https://github.com/liu-ming-hua/VLAWorldEval"
project = "VLAWorldEval"
author = "VLAWorldEval contributors"
copyright = "2026, VLAWorldEval contributors"
release = "0.1.0"
extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["README.md", "assets/**"]
templates_path = ["_templates"]
myst_enable_extensions = ["colon_fence", "deflist", "html_image"]
myst_heading_anchors = 5
html_search_language = "zh"  # Chinese segmentation plus English stemming.
html_theme = "pydata_sphinx_theme"
html_title = "VLAWorldEval · Documentation"
html_baseurl = "https://liu-ming-hua.github.io/VLAWorldEval/"
html_static_path = ["_static"]
html_css_files = ["manual.css"]
html_show_sourcelink = False
html_show_sphinx = False
html_theme_options = {
    "logo": {"text": "VLAWorldEval"},
    "github_url": REPOSITORY,
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["language-switch", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "secondary_sidebar_items": ["page-toc"],
    "show_prev_next": False,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_sidebars = {"**": ["search-field.html", "manual-nav.html"]}


def repository_links(text, docname):
    """Keep GitHub Markdown usable; source-only links become repository URLs."""
    source = DOCS / (docname + ".md")

    def rewrite(match):
        target = match.group(2)
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path or target.startswith("/"):
            return match.group(0)
        resolved = (source.parent / url.path).resolve()
        # Sphinx resolves manual pages and images itself. Repository configs,
        # root READMEs and source directories are not website documents.
        if resolved.is_relative_to(DOCS) and resolved.suffix in {".md", ".svg", ".png", ".gif"}:
            return match.group(0)
        if resolved.is_relative_to(ROOT):
            relative = resolved.relative_to(ROOT).as_posix()
            route = "tree" if resolved.is_dir() else "blob"
            fragment = "#" + url.fragment if url.fragment else ""
            return f"{match.group(1)}{REPOSITORY}/{route}/main/{relative}{fragment})"
        return match.group(0)

    return re.sub(r"(\[[^\]\n]*\]\()([^\s)]+)\)", rewrite, text)


def source_read(app, docname, source):
    if docname.startswith(("en/", "zh/")):
        source[0] = repository_links(source[0], docname)


def page_context(app, pagename, templatename, context, doctree):
    context["manual_language"] = "zh" if pagename.startswith("zh/") else "en"
    context["html_lang"] = "zh-CN" if context["manual_language"] == "zh" else "en"
    context["language"] = context["html_lang"]
    if pagename in {"index", "en/README", "zh/README"}:
        context["body"] = app.builder.templates.render("landing.html", context) + context["body"]


def copy_reviewed_media(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    assets = json.loads((DOCS / "assets/provenance.json").read_text())["assets"]
    for item in assets:
        source = ROOT / item["path"]
        if source.is_symlink() or not source.resolve().is_relative_to(DOCS / "assets"):
            raise ValueError("Unsafe presentation media path")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Presentation media checksum mismatch: " + item["path"])
    names = [item["path"] for item in assets]
    names += ["docs/assets/provenance.json", "docs/assets/reference-results.json"]
    for name in names:
        source = ROOT / name
        if source.is_symlink() or not source.resolve().is_relative_to(DOCS / "assets"):
            raise ValueError("Unsafe presentation media path")
        destination = Path(app.outdir) / source.relative_to(DOCS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (Path(app.outdir) / ".nojekyll").touch()


def setup(app):
    app.connect("source-read", source_read)
    app.connect("html-page-context", page_context)
    app.connect("build-finished", copy_reviewed_media)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
